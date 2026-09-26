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



FILES = {
    "1d": "allen_cahn_1d.npy",
    "2d": "allen_cahn_2d.npy",
    "3d": "allen_cahn_3d.npy",
}


def _build_loader(cfg, filename, split):
    data = np.load(os.path.join(cfg.data.dataset_root, filename))
    # data: (num_traj, T, C, *spatial), C == 1

    if split == "train":
        data = data[:20]
    elif split == "valid":
        data = data[20:25]
    elif split == "test":
        data = data[25:30]
    else:
        raise NotImplementedError

    num_traj, T = data.shape[0], data.shape[1]
    # flatten channel + spatial dims -> num_discretization_points
    data = data.reshape(num_traj, T, -1)  # (num_traj, T, num_points)

    all_x, all_y = [], []
    horizon = 5
    for traj in data:
        num_timesteps = traj.shape[0]
        input_indices = np.arange(0, num_timesteps - horizon, cfg.data.split_interval)
        target_indices = input_indices + horizon

        all_x.append(traj[input_indices])
        all_y.append(traj[target_indices])

    all_x = np.concatenate(all_x, axis=0)
    all_y = np.concatenate(all_y, axis=0)

    # (N, num_points) -> (N, num_points, 1)
    all_x = torch.tensor(all_x, dtype=torch.float32).unsqueeze(-1)
    all_y = torch.tensor(all_y, dtype=torch.float32).unsqueeze(-1)

    print(f"[{split}] {filename} -> all_x: {all_x.shape}, all_y: {all_y.shape}")

    shuffle = (split == "train")
    dataset = TensorDataset(all_x, all_y)
    dataloader = DataLoader(
        dataset,
        batch_size=cfg.training.batch_size,
        shuffle=shuffle,
        num_workers=0,
        pin_memory=True,
    )

    return dataloader


def load_single_dataset(cfg, split):
    # returns a dict of dataloaders, one per dimensionality
    return {key: _build_loader(cfg, fname, split) for key, fname in FILES.items()}


def _load_split(cfg, split):
    file_path = f"{cfg.data.data_save_path}/{split}.pth"
    if os.path.exists(file_path):
        print(f"{split}.pth already exists. Load from {file_path}")
        all_data = torch.load(file_path)
        shuffle = (split == "train")
        dataloaders = {
            key: DataLoader(ds, batch_size=cfg.training.batch_size, shuffle=shuffle)
            for key, ds in all_data.items()
        }
    else:
        print(f"{split}.pth does not exists. Create dataset")
        dataloaders = load_single_dataset(cfg, split)

    return dataloaders


def load_train(cfg):
    return _load_split(cfg, "train")


def load_valid(cfg):
    return _load_split(cfg, "valid")


def load_test(cfg):
    return _load_split(cfg, "test")