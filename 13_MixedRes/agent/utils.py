import os
import yaml
import glob
import pyarrow as pa
import numpy as np
import torch
from torch.utils.data import TensorDataset, Dataset, DataLoader
from types import SimpleNamespace
from collections import defaultdict




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


class MixedResDataset(Dataset):
    def __init__(self, root, split, split_interval=1):
        self.split_interval = split_interval
        split_dir = os.path.join(root, split)
        self.files = sorted(glob.glob(os.path.join(split_dir, "batch_*.npz")))
        if not self.files:
            raise FileNotFoundError(f"No batch_*.npz found in {split_dir}")

        
        
        self.index = []
        self.shapes = []
        
        self.meta = []  # list of (H, W, dx, dy)

        for fi, fpath in enumerate(self.files):
            with np.load(fpath) as d:
                vor = d["vor"]  # (bsize, T+1, H, W)
                bsize, n_times, H, W = vor.shape
                dx = float(d["dx"])
                dy = float(d["dy"])
            self.meta.append((H, W, dx, dy))

            
            last_input = n_times - 1 - self.split_interval
            for s in range(bsize):
                for t in range(0, last_input + 1, self.split_interval):
                    self.index.append((fi, s, t))
                    self.shapes.append((H, W))

    def __len__(self):
        return len(self.index)

    def _make_pos(self, fi, H, W):
        _, _, dx, dy = self.meta[fi]
        xs = np.arange(W, dtype=np.float32) * dx
        ys = np.arange(H, dtype=np.float32) * dy
        gx, gy = np.meshgrid(xs, ys, indexing="xy")  # both (H, W)
        return np.stack([gx, gy], axis=0)  # (2, H, W)

    def __getitem__(self, i):
        fi, s, t = self.index[i]
        fpath = self.files[fi]
        with np.load(fpath) as d:
            vor = d["vor"]  # (bsize, T+1, H, W)
            x = vor[s, t].astype(np.float32)
            y = vor[s, t + self.split_interval].astype(np.float32)

        H, W = x.shape
        pos = self._make_pos(fi, H, W)  # (2, H, W)

        x = torch.from_numpy(x).unsqueeze(0)   # (1, H, W)
        y = torch.from_numpy(y).unsqueeze(0)   # (1, H, W)
        pos = torch.from_numpy(pos)            # (2, H, W)
        return x, pos, y


class ShapeBatchSampler(torch.utils.data.Sampler):
    def __init__(self, shapes, batch_size, shuffle):
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.groups = defaultdict(list)
        for idx, hw in enumerate(shapes):
            self.groups[hw].append(idx)

    def __iter__(self):
        batches = []
        for hw, idxs in self.groups.items():
            idxs = list(idxs)
            if self.shuffle:
                np.random.shuffle(idxs)
            for k in range(0, len(idxs), self.batch_size):
                batches.append(idxs[k:k + self.batch_size])
        if self.shuffle:
            np.random.shuffle(batches)
        return iter(batches)

    def __len__(self):
        total = 0
        for idxs in self.groups.values():
            total += (len(idxs) + self.batch_size - 1) // self.batch_size
        return total


def _collate(samples):
    xs = torch.stack([s[0] for s in samples], dim=0)   # (B, 1, H, W)
    pos = torch.stack([s[1] for s in samples], dim=0)  # (B, 2, H, W)
    ys = torch.stack([s[2] for s in samples], dim=0)   # (B, 1, H, W)
    return xs, pos, ys


def _load_split(cfg, split):
    split_interval = getattr(cfg.data, "split_interval", 1)
    dataset = MixedResDataset(
        root=cfg.data.dataset_root,
        split=split,
        split_interval=split_interval,
    )
    shuffle = (split == "train_s")
    sampler = ShapeBatchSampler(dataset.shapes, cfg.training.batch_size, shuffle)
    dataloader = DataLoader(
        dataset,
        batch_sampler=sampler,
        collate_fn=_collate,
        pin_memory=True,
    )
    print(f"[{split}] {len(dataset)} samples, {len(sampler)} batches")
    return dataloader


def load_train(cfg):
    return _load_split(cfg, "train_s")


def load_valid(cfg):
    return _load_split(cfg, "valid_s")


def load_test(cfg):
    return _load_split(cfg, "test_s")