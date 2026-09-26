import os
import re
import yaml
import h5py
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

def load_single_dataset(cfg, split):
    h5_files = []
    for f in os.listdir(cfg.data.dataset_root):
        match = re.match(r'plt_(\d+)\.h5', f)
        if match:
            h5_files.append((int(match.group(1)), f))

    h5_files.sort(key=lambda x: x[0])
    # print(f"h5_files: {h5_files}")

    if split == "train":
        h5_files = h5_files[:int(0.8 * len(h5_files))]
    elif split == "valid":
        h5_files = h5_files[int(0.8 * len(h5_files)):int(0.9 * len(h5_files))]
    elif split == "test":
        h5_files = h5_files[int(0.9 * len(h5_files)):]
    else:
        raise NotImplementedError

    all_x, all_tx = [], []
    all_y, all_ty = [], []
    for num, filename in h5_files:
        file_path = os.path.join(cfg.data.dataset_root, filename)

        with h5py.File(file_path, 'r') as hf:
            time = hf['time'][:]
            density = hf['density'][:]
            temperature = hf['temperature'][:]
            xVel = hf['xVel'][:]
            yVel = hf['yVel'][:]

            stacked = np.stack([temperature, density, xVel, yVel], axis=1)

            num_timesteps = stacked.shape[0]
            input_indices = np.arange(0, num_timesteps - 1, cfg.data.split_interval)
            target_indices = input_indices + 1

            all_x.append(stacked[input_indices])
            all_y.append(stacked[target_indices])

            all_tx.append(time[input_indices])
            all_ty.append(time[target_indices])

    all_x = np.concatenate(all_x, axis=0)
    all_y = np.concatenate(all_y, axis=0)
    all_tx = np.concatenate(all_tx, axis=0)
    all_ty = np.concatenate(all_ty, axis=0)

    all_x = torch.tensor(all_x, dtype=torch.float32)
    all_y = torch.tensor(all_y, dtype=torch.float32)
    all_tx = torch.tensor(all_tx, dtype=torch.float32)
    all_ty = torch.tensor(all_ty, dtype=torch.float32)

    all_tx = all_tx.unsqueeze(-1)
    all_ty = all_ty.unsqueeze(-1)

    print(f"all_x: {all_x.shape}")
    print(f"all_y: {all_y.shape}")
    print(f"all_tx: {all_tx.shape}")
    print(f"all_ty: {all_ty.shape}")

    shuffle = (split == "train")
    dataset = TensorDataset(all_x, all_tx, all_y, all_ty)
    dataloader = DataLoader(
        dataset,
        batch_size=cfg.training.batch_size,
        shuffle=shuffle,
        num_workers=0,
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