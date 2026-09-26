import os
import yaml
import glob
import pyarrow as pa
import numpy as np
import torch
from torch.utils.data import TensorDataset, DataLoader
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


import os
import numpy as np
import torch
from torch.utils.data import TensorDataset, DataLoader

BC_SUBDIRS = [
    "dirichlet_time_dependent",
    "neumann_time_dependent",
    "periodic_time_dependent",
]
BC_TO_ID = {name: i for i, name in enumerate(BC_SUBDIRS)}
ID_TO_BC = {i: name for name, i in BC_TO_ID.items()}

SPLIT_RANGES = {
    "train": (0, 20),
    "valid": (20, 25),
    "test": (25, 30),
}


def process_npy_file(file_path):
    data = np.load(file_path).astype(np.float32)
    return torch.from_numpy(data)


def load_single_dataset(cfg, split):
    if split not in SPLIT_RANGES:
        raise ValueError(f"split must be one of {list(SPLIT_RANGES)}, got '{split}'")
    start, end = SPLIT_RANGES[split]
    all_x = []
    all_y = []
    all_bc_type = []
    all_bc_vals = []
    for subdir in BC_SUBDIRS:
        bc_id = BC_TO_ID[subdir]
        f_path = os.path.join(cfg.data.dataset_root, subdir, "trajectories.npy")
        traj = process_npy_file(f_path)
        traj = traj[start:end]

        bc_path = os.path.join(cfg.data.dataset_root, subdir, "boundary_conditions.npy")
        if os.path.exists(bc_path):
            bc_raw = process_npy_file(bc_path)
            bc_raw = bc_raw[start:end]
        else:
            bc_raw = torch.zeros(traj.shape[0], traj.shape[1], 4)

        for n in range(traj.shape[0]):
            video_tensor = traj[n]
            bc_tensor = bc_raw[n]
            x = video_tensor[:-1:cfg.data.split_interval]
            y = video_tensor[1::cfg.data.split_interval]
            bc_vals = bc_tensor[:-1:cfg.data.split_interval]
            all_x.append(x)
            all_y.append(y)
            all_bc_type.append(torch.full((x.shape[0],), bc_id, dtype=torch.long))
            all_bc_vals.append(bc_vals)
    full_x = torch.cat(all_x, dim=0)
    full_y = torch.cat(all_y, dim=0)
    full_bc_type = torch.cat(all_bc_type, dim=0)
    full_bc_vals = torch.cat(all_bc_vals, dim=0)
    print(f"full_x: {full_x.shape}")
    print(f"full_y: {full_y.shape}")
    print(f"full_bc_type: {full_bc_type.shape}")
    print(f"full_bc_vals: {full_bc_vals.shape}")
    dataset = TensorDataset(full_x, full_y, full_bc_type, full_bc_vals)
    shuffle = (split == "train")
    dataloader = DataLoader(dataset, batch_size=cfg.training.batch_size, shuffle=shuffle, pin_memory=True)
    return dataloader


def _load_split(cfg, split):
    file_path = f"{cfg.data.data_save_path}/{split}.pth"
    if os.path.exists(file_path):
        print(f"{split}.pth already exists. Load from {file_path}")
        all_data = torch.load(file_path)
        print(len(all_data))
        shuffle = (split == "train")
        dataloader = DataLoader(all_data, batch_size=cfg.training.batch_size, shuffle=shuffle)
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