import os
import re
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


RE_MIN = 100.0
RE_MAX = 5000.0


def _parse_re(folder_path):
    """Extract the integer Re value from a folder like '.../Re_1300'."""
    name = os.path.basename(folder_path.rstrip("/"))
    m = re.search(r"Re_(\d+)", name)
    if m is None:
        raise ValueError(f"Could not parse Re value from {folder_path}")
    return float(m.group(1))


def _normalize_re(re_value):
    return (re_value - RE_MIN) / (RE_MAX - RE_MIN)


def process_npy_file(file_path):
    data = np.load(file_path)  # (20, 21, 128, 128)
    return torch.from_numpy(data.astype(np.float32))


def load_single_dataset(cfg, split):
    split_dir = {
        "train": "ns_train",
        "valid": "ns_valid",
        "test": "ns_test",
    }
    if split not in split_dir:
        raise NotImplementedError(split)

    root = os.path.join(cfg.data.dataset_root, split_dir[split])
    re_folders = sorted(glob.glob(os.path.join(root, "Re_*")))
    re_folders = [f for f in re_folders if os.path.isdir(f)]

    interval = cfg.data.split_interval
    horizon = 5

    all_x = []
    all_y = []
    all_param = []

    for folder in re_folders:
        re_value = _parse_re(folder)
        re_norm = _normalize_re(re_value)

        npy_files = sorted(glob.glob(os.path.join(folder, "*.npy")))
        for f_path in npy_files:
            video = process_npy_file(f_path)  # (n_traj, T, H, W)

            # x[t] -> y[t+horizon], stepping by split_interval along time.
            x = video[:, :-horizon:interval]   # (n_traj, n_t, H, W)
            y = video[:, horizon::interval]    # (n_traj, n_t, H, W)

            n_traj, n_t, H, W = x.shape
            x = x.reshape(n_traj * n_t, 1, H, W)
            y = y.reshape(n_traj * n_t, 1, H, W)

            param = torch.full((x.shape[0], 1), re_norm, dtype=torch.float32)

            all_x.append(x)
            all_y.append(y)
            all_param.append(param)

    full_x = torch.cat(all_x, dim=0)
    full_y = torch.cat(all_y, dim=0)
    full_param = torch.cat(all_param, dim=0)
    print(f"full_x: {full_x.shape}")
    print(f"full_y: {full_y.shape}")
    print(f"full_param: {full_param.shape}")

    dataset = TensorDataset(full_x, full_param, full_y)
    shuffle = (split == "train")
    dataloader = DataLoader(
        dataset,
        batch_size=cfg.training.batch_size,
        shuffle=shuffle,
        pin_memory=True,
    )
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