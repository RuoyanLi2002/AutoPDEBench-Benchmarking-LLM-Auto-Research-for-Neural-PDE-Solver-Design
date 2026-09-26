import os
import random
import yaml
import glob
import pyarrow as pa
import numpy as np
import torch
from torch.utils.data import ConcatDataset, DataLoader, Sampler, TensorDataset
from types import SimpleNamespace



# -----------------------------
# Config loading
# -----------------------------

class Config(SimpleNamespace):
    """Namespace supporting nested attribute access (e.g. cfg.data.dataset_root)."""
    pass


def _dict_to_namespace(d):
    if isinstance(d, dict):
        return Config(**{k: _dict_to_namespace(v) for k, v in d.items()})
    elif isinstance(d, list):
        return [_dict_to_namespace(v) for v in d]
    else:
        return d


def to_dict(ns):
    """Recursively convert a Config namespace back to a plain dict.

    Useful for unpacking config sub-sections into class constructors,
    e.g. `Adam(model.parameters(), **to_dict(cfg.training.optimizer.params))`.
    """
    if isinstance(ns, SimpleNamespace):
        return {k: to_dict(v) for k, v in vars(ns).items()}
    elif isinstance(ns, list):
        return [to_dict(v) for v in ns]
    else:
        return ns


def _deep_merge(base, override):
    """Recursively merge `override` into `base`. Values in `override` win on conflict."""
    result = dict(base)
    for k, v in override.items():
        if k in result and isinstance(result[k], dict) and isinstance(v, dict):
            result[k] = _deep_merge(result[k], v)
        else:
            result[k] = v
    return result


def _resolve_base(config):
    """Recursively resolve `base_config` references inside a loaded config dict."""
    if "base_config" in config:
        base_path = config.pop("base_config")
        with open(base_path, "r") as f:
            base = yaml.safe_load(f) or {}
        base = _resolve_base(base)
        config = _deep_merge(base, config)
    return config


def load_config(config_path):
    """Load a YAML config file, resolving any `base_config` chain into a single namespace."""
    with open(config_path, "r") as f:
        config = yaml.safe_load(f) or {}
    config = _resolve_base(config)
    return _dict_to_namespace(config)


# -----------------------------
# Data loading
# -----------------------------


PDE_FILES = [
    "advection_diffusion_2d.npy",
    "burgers_2d.npy",
    "gray_scott_2d.npy",
    "navier_stokes_vorticity_2d.npy",
]
PDE_SUBDIRS = [os.path.splitext(f)[0] for f in PDE_FILES]
PDE_TO_ID = {name: i for i, name in enumerate(PDE_SUBDIRS)}
ID_TO_PDE = {i: name for name, i in PDE_TO_ID.items()}

SPLIT_RANGES = {
    "train": (0, 20),
    "valid": (20, 25),
    "test": (25, 30),
}


def process_npy_file(file_path):
    data = np.load(file_path).astype(np.float32)
    return torch.from_numpy(data)


class GroupedBatchSampler(Sampler):
    """Yields index batches drawn from a single group at a time.

    Groups correspond to channel counts, so every batch is homogeneous in
    `dim` and the default collate_fn can stack it into a regular tensor.
    Batches from different groups are interleaved (shuffled) during training.
    """

    def __init__(self, group_sizes, batch_size, shuffle, drop_last=False):
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.drop_last = drop_last
        self.groups = []
        offset = 0
        for size in group_sizes:
            self.groups.append(range(offset, offset + size))
            offset += size

    def __iter__(self):
        batches = []
        for grp in self.groups:
            idx = list(grp)
            if self.shuffle:
                random.shuffle(idx)
            for i in range(0, len(idx), self.batch_size):
                batch = idx[i : i + self.batch_size]
                if self.drop_last and len(batch) < self.batch_size:
                    continue
                batches.append(batch)
        if self.shuffle:
            random.shuffle(batches)
        return iter(batches)

    def __len__(self):
        n = 0
        for grp in self.groups:
            if self.drop_last:
                n += len(grp) // self.batch_size
            else:
                n += (len(grp) + self.batch_size - 1) // self.batch_size
        return n


def _make_grouped_dataloader(dataset, cfg, split):
    """Build a DataLoader whose batches never mix channel counts.

    `dataset` must be a ConcatDataset of channel-homogeneous TensorDatasets.
    """
    group_sizes = [len(d) for d in dataset.datasets]
    shuffle = (split == "train")
    batch_sampler = GroupedBatchSampler(
        group_sizes, cfg.training.batch_size, shuffle=shuffle
    )

    return DataLoader(dataset, batch_sampler=batch_sampler, pin_memory=True)


def load_single_dataset(cfg, split):
    if split not in SPLIT_RANGES:
        raise ValueError(f"split must be one of {list(SPLIT_RANGES)}, got '{split}'")
    start, end = SPLIT_RANGES[split]


    horizon = 5
    if horizon < 1:
        raise ValueError(f"cfg.data.horizon must be >= 1, got {horizon}")



    buckets = {}

    for fname in PDE_FILES:
        pde_name = os.path.splitext(fname)[0]
        pde_id = PDE_TO_ID[pde_name]
        f_path = os.path.join(cfg.data.dataset_root, fname)

        traj = process_npy_file(f_path)  # (N, T, dim, H, W)
        n_channels = traj.shape[2]

        traj = traj[start:end]  # (n_split, T, dim, H, W)
        if traj.shape[1] <= horizon:
            raise ValueError(
                f"{pde_name}: trajectory length T={traj.shape[1]} must exceed "
                f"horizon={horizon} to form any (input, target) pairs."
            )

        bucket = buckets.setdefault(n_channels, {"x": [], "y": [], "pde": []})
        for n in range(traj.shape[0]):
            video_tensor = traj[n]
            
            x = video_tensor[:-horizon : cfg.data.split_interval]
            y = video_tensor[horizon :: cfg.data.split_interval]
            bucket["x"].append(x)
            bucket["y"].append(y)
            bucket["pde"].append(torch.full((x.shape[0],), pde_id, dtype=torch.long))


    group_datasets = []
    for n_channels in sorted(buckets):
        bucket = buckets[n_channels]
        full_x = torch.cat(bucket["x"], dim=0)
        full_y = torch.cat(bucket["y"], dim=0)
        full_pde = torch.cat(bucket["pde"], dim=0)
        print(f"[{split}] channels={n_channels}  "
              f"x: {tuple(full_x.shape)}  y: {tuple(full_y.shape)}  "
              f"pde: {tuple(full_pde.shape)}")
        group_datasets.append(TensorDataset(full_x, full_y, full_pde))

    dataset = ConcatDataset(group_datasets)
    return _make_grouped_dataloader(dataset, cfg, split)


def _load_split(cfg, split):
    file_path = f"{cfg.data.data_save_path}/{split}.pth"
    if os.path.exists(file_path):
        print(f"{split}.pth already exists. Load from {file_path}")
        all_data = torch.load(file_path)
        print(len(all_data))
        if isinstance(all_data, ConcatDataset):
            dataloader = _make_grouped_dataloader(all_data, cfg, split)
        else:
            print(
                f"WARNING: {file_path} is in the old (padded) format; "
                f"delete it to rebuild without channel padding."
            )
            shuffle = (split == "train")
            dataloader = DataLoader(
                all_data, batch_size=cfg.training.batch_size, shuffle=shuffle
            )
    else:
        print(f"{split}.pth does not exists. Create dataset")
        dataloader = load_single_dataset(cfg, split)
    return dataloader


def load_train(cfg):
    return _load_split(cfg, "train")


def load_valid(cfg):
    return _load_split(cfg, "valid")


def load_test(cfg):
    return _load_split(cfg, "test")