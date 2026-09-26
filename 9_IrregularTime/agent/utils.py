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


def process_npz_file(file_path, split_interval):
    data = np.load(file_path)
    vor = data["vor"].astype(np.float32)   # (B, T+1, H, W)
    t = data["t"].astype(np.float32)       # (T+1,)

    all_x, all_y, all_dt = [], [], []
    for b in range(vor.shape[0]):          # handles bsize > 1, not just [0]
        video = torch.from_numpy(vor[b])[:, None, :, :]   # (T+1, 1, H, W)
        x = video[:-1:split_interval]                     # (N, 1, H, W)
        y = video[1::split_interval]                      # (N, 1, H, W)
        dt = torch.from_numpy(
            t[1::split_interval] - t[:-1:split_interval]  # (N,)
        )
        assert x.shape[0] == y.shape[0] == dt.shape[0]
        all_x.append(x)
        all_y.append(y)
        all_dt.append(dt)

    return torch.cat(all_x), torch.cat(all_y), torch.cat(all_dt)


def load_single_dataset(cfg, split):
    split_dir = os.path.join(cfg.data.dataset_root, split)
    files = sorted(glob.glob(os.path.join(split_dir, "*.npz")))
    if not files:
        raise FileNotFoundError(f"No .npz files found in {split_dir}")

    all_x, all_y, all_dt = [], [], []
    for f_path in files:
        x, y, dt = process_npz_file(f_path, cfg.data.split_interval)
        all_x.append(x)
        all_y.append(y)
        all_dt.append(dt)

    full_x = torch.cat(all_x, dim=0)
    full_y = torch.cat(all_y, dim=0)
    full_dt = torch.cat(all_dt, dim=0)
    print(f"full_x: {full_x.shape} | full_y: {full_y.shape} | full_dt: {full_dt.shape}")

    dataset = TensorDataset(full_x, full_y, full_dt)
    shuffle = (split == "train")  # only shuffle training data
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
        # Sanity check: cached datasets created before dt was added only
        # contain (x, y) and must be regenerated.
        if isinstance(all_data, TensorDataset) and len(all_data.tensors) != 3:
            raise RuntimeError(
                f"{file_path} is a stale cache without dt. Delete it and rerun."
            )
        shuffle = (split == "train")
        dataloader = DataLoader(
            all_data,
            batch_size=cfg.training.batch_size,
            shuffle=shuffle,
            pin_memory=True,
        )
    else:
        print(f"{split}.pth does not exist. Create dataset")
        dataloader = load_single_dataset(cfg, split)
    return dataloader


def load_train(cfg):
    return _load_split(cfg, "train")


def load_valid(cfg):
    return _load_split(cfg, "valid")


def load_test(cfg):
    return _load_split(cfg, "test")