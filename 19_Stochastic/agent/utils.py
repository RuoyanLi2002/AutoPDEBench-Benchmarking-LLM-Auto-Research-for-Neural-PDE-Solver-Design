import os
import yaml
import glob
import h5py
import pyarrow as pa
import numpy as np
import torch
from torch.utils.data import TensorDataset, Dataset, DataLoader
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




class SPDEDataset(Dataset):
    def __init__(self, sol, W, horizon=20, noise_mode="full"):
        assert sol.shape == W.shape, f"sol {sol.shape} vs W {W.shape}"
        assert noise_mode in ("full", "single")
        self.sol = sol            # (N, T, H, W) float32 tensors
        self.W = W
        self.horizon = horizon
        self.noise_mode = noise_mode

        N, T = sol.shape[0], sol.shape[1]
        self.pairs_per_traj = T - horizon
        assert self.pairs_per_traj > 0, "horizon >= trajectory length"
        self.length = N * self.pairs_per_traj

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        n = idx // self.pairs_per_traj
        t = idx % self.pairs_per_traj
        k = self.horizon

        u_t = self.sol[n, t].unsqueeze(0)                 # (1, H, W)
        if self.noise_mode == "full":
            noise = self.W[n, t:t + k]                    # (k, H, W)
        else:
            noise = self.W[n, t].unsqueeze(0)             # (1, H, W)

        x = torch.cat([u_t, noise], dim=0)                # (1+k, H, W) or (2, H, W)
        y = self.sol[n, t + k].unsqueeze(0)               # (1, H, W)
        return x, y


def load_h5_split(cfg, split):
    file_path = os.path.join(cfg.data.dataset_root, f"{split}.h5")
    with h5py.File(file_path, "r") as f:
        sol = torch.from_numpy(f["sol"][:]).float()       # (N, T, H, W)
        W = torch.from_numpy(f["W"][:]).float()

    if getattr(cfg.data, "drop_periodic_endpoint", True):
        sol = sol[..., :-1, :-1]
        W = W[..., :-1, :-1]

    horizon = 20
    noise_mode = "full"
    dataset = SPDEDataset(sol, W, horizon=horizon, noise_mode=noise_mode)

    print(f"[{split}] sol: {tuple(sol.shape)}, W: {tuple(W.shape)}, "
          f"horizon={horizon}, noise_mode={noise_mode}, "
          f"num_pairs={len(dataset)}, x_channels={1 + (horizon if noise_mode == 'full' else 1)}")

    shuffle = (split == "train")
    dataloader = DataLoader(
        dataset,
        batch_size=cfg.training.batch_size,
        shuffle=shuffle,
        pin_memory=True,
        num_workers=getattr(cfg.training, "num_workers", 0),
    )
    return dataloader


def _load_split(cfg, split):
    return load_h5_split(cfg, split)


def load_train(cfg):
    return _load_split(cfg, "train")


def load_valid(cfg):
    return _load_split(cfg, "valid")


def load_test(cfg):
    return _load_split(cfg, "test")