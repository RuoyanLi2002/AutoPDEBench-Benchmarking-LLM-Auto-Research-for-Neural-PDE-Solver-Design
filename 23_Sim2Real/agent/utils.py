import os
import yaml
import glob
import h5py
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




def process_h5_file(file_path, time_stride=1, max_frames=None):
    """Read one FSI h5 file -> (T, 2, 128, 128) tensor with channels (u, v)."""
    with h5py.File(file_path, "r") as f:
        u = f["measured_data/u"][...]  # (T, 128, 128) float32
        v = f["measured_data/v"][...]  # (T, 128, 128) float32

    combined = np.stack([u, v], axis=1)  # (T, 2, 128, 128)

    if time_stride > 1:
        combined = combined[::time_stride]
    if max_frames is not None:
        combined = combined[:max_frames]

    return torch.from_numpy(np.ascontiguousarray(combined))


def _get_files(cfg, split):
    """train/valid come from numerical (sim), test comes from real."""
    if split in ("train", "valid"):
        domain = "numerical"
    elif split == "test":
        domain = "real"
    else:
        raise NotImplementedError(split)

    files = sorted(glob.glob(os.path.join(cfg.data.dataset_root, domain, "*.h5")))
    assert len(files) > 0, f"No h5 files found under {cfg.data.dataset_root}/{domain}"

    if split == "train":
        files = files[: int(0.8 * len(files))]
    elif split == "valid":
        files = files[int(0.8 * len(files)):]

    max_files = getattr(cfg.data, "max_files", None)
    if max_files is not None:
        files = files[:max_files]

    return files


def load_single_dataset(cfg, split):
    files = _get_files(cfg, split)[:10]
    print(f"[{split}] using {len(files)} files")

    time_stride = getattr(cfg.data, "time_stride", 1)
    max_frames = getattr(cfg.data, "max_frames", None)

    all_x, all_y = [], []
    for f_path in files:
        video_tensor = process_h5_file(f_path, time_stride=time_stride, max_frames=max_frames)

        x = video_tensor[:-1:cfg.data.split_interval]
        y = video_tensor[1::cfg.data.split_interval]

        all_x.append(x)
        all_y.append(y)

    full_x = torch.cat(all_x, dim=0)
    full_y = torch.cat(all_y, dim=0)
    print(f"[{split}] full_x: {full_x.shape}")
    print(f"[{split}] full_y: {full_y.shape}")

    dataset = TensorDataset(full_x, full_y)
    shuffle = (split == "train")
    dataloader = DataLoader(dataset, batch_size=cfg.training.batch_size,
                            shuffle=shuffle, pin_memory=True)
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
        print(f"{split}.pth does not exist. Create dataset")
        dataloader = load_single_dataset(cfg, split)

    return dataloader


def load_train(cfg):
    return _load_split(cfg, "train")


def load_valid(cfg):
    return _load_split(cfg, "valid")


def load_test(cfg):
    return _load_split(cfg, "test")