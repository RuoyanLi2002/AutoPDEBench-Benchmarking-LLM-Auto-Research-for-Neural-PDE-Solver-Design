import os
import re
import yaml
import glob
import h5py
import pyarrow as pa
import numpy as np
import torch
from torch.utils.data import Dataset, TensorDataset, DataLoader
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




def _natural_key(path):
    """Sort iteration_0, iteration_1, ..., iteration_10 numerically, not lexicographically."""
    m = re.search(r"iteration_(\d+)_5d\.h5$", os.path.basename(path))
    return int(m.group(1)) if m else -1


class Gyro5DOneStepDataset(Dataset):
    def __init__(self, files, split_interval=1):
        assert len(files) > 0, "No .h5 files given to dataset"
        self.files = list(files)
        self.split_interval = split_interval
        self._handles = None


        self.index = []
        for fi, path in enumerate(self.files):
            with h5py.File(path, "r") as h5:
                T = h5["trajectory"].shape[0]
            for t in range(0, T - 1, self.split_interval):
                self.index.append((fi, t))


    def _get_traj(self, fi):
        if self._handles is None:
            self._handles = [None] * len(self.files)
        if self._handles[fi] is None:
            self._handles[fi] = h5py.File(self.files[fi], "r")
        return self._handles[fi]["trajectory"]

    def __getstate__(self):
        state = self.__dict__.copy()
        state["_handles"] = None
        return state

    
    def __len__(self):
        return len(self.index)

    def __getitem__(self, idx):
        fi, t = self.index[idx]
        traj = self._get_traj(fi)
        
        x = torch.from_numpy(np.asarray(traj[t], dtype=np.float32))
        y = torch.from_numpy(np.asarray(traj[t + 1], dtype=np.float32))
        return x, y

    def close(self):
        if self._handles:
            for h in self._handles:
                if h is not None:
                    h.close()
            self._handles = None


def _split_files(cfg, split):
    files = sorted(
        glob.glob(os.path.join(cfg.data.dataset_root, "*.h5")),
        key=_natural_key,
    )



    
    if split == "train":
        files = files[:8]
    elif split == "valid":
        files = files[8:9]
    elif split == "test":
        files = files[9:10]
    else:
        raise NotImplementedError(split)
    return files


def _load_split(cfg, split):
    files = _split_files(cfg, split)
    print(f"[{split}] using {len(files)} trajectories:")
    for f in files:
        print(f"    {f}")

    split_interval = getattr(cfg.data, "split_interval", 1)
    dataset = Gyro5DOneStepDataset(files, split_interval=split_interval)
    print(f"[{split}] {len(dataset)} one-step samples")

    shuffle = (split == "train")
    num_workers = getattr(cfg.data, "num_workers", 4)
    dataloader = DataLoader(
        dataset,
        batch_size=cfg.training.batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=True,
        persistent_workers=(num_workers > 0),
        prefetch_factor=2 if num_workers > 0 else None,
        drop_last=(split == "train"),
    )
    return dataloader


def load_train(cfg):
    return _load_split(cfg, "train")


def load_valid(cfg):
    return _load_split(cfg, "valid")


def load_test(cfg):
    return _load_split(cfg, "test")