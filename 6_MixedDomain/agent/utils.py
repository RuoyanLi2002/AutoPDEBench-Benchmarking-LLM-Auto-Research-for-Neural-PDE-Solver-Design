import os
import yaml
import glob
import pyarrow as pa
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader, Sampler
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





XMIN, XMAX = 8.002590257127906e-10, 0.11999999731779099
YMIN, YMAX = -0.0062500000931322575, 0.08624999970197678
HISTORY = 5


def _grid_positions(h, w):
    xs = torch.linspace(XMIN, XMAX, h)
    ys = torch.linspace(YMIN, YMAX, w)
    gx, gy = torch.meshgrid(xs, ys, indexing="ij")
    return torch.stack([gx, gy], dim=-1).reshape(-1, 2)


class MixedDomainDataset(Dataset):
    def __init__(self, root, timesteps):
        mesh_pos = torch.from_numpy(np.load(os.path.join(root, "eulerian_mesh_nodes.npy"))).float()
        mesh_vel = torch.from_numpy(np.load(os.path.join(root, "eulerian_velocity_irregular.npy"))).float()
        grid_vel = torch.from_numpy(np.load(os.path.join(root, "eulerian_velocity.npy"))).float()
        part_pos = torch.from_numpy(np.load(os.path.join(root, "particle_pos.npy"))).float()
        part_type = torch.from_numpy(np.load(os.path.join(root, "particle_type.npy"))).float()

        T, _, H, W = grid_vel.shape
        grid_vel = grid_vel.permute(0, 2, 3, 1).reshape(T, H * W, 2)
        grid_pos = _grid_positions(H, W)

        part_vel = torch.zeros_like(part_pos)
        part_vel[1:] = part_pos[1:] - part_pos[:-1]

        self.domains = [
            dict(pos=mesh_pos, vel=mesh_vel, node_type=torch.zeros(mesh_pos.shape[0], 1), lagrangian=False),
            dict(pos=grid_pos, vel=grid_vel, node_type=torch.ones(grid_pos.shape[0], 1), lagrangian=False),
            dict(pos=part_pos, vel=part_vel, node_type=3.0 - part_type, lagrangian=True),
        ]
        self.timesteps = list(timesteps)
        self.group_sizes = [len(self.timesteps)] * len(self.domains)

    def __len__(self):
        return len(self.domains) * len(self.timesteps)

    def __getitem__(self, idx):
        d, i = divmod(idx, len(self.timesteps))
        t = self.timesteps[i]
        dom = self.domains[d]
        pos = dom["pos"][t] if dom["lagrangian"] else dom["pos"]
        hist = dom["vel"][t - HISTORY + 1 : t + 1]
        hist = hist.permute(1, 0, 2).reshape(pos.shape[0], 2 * HISTORY)
        x = torch.cat([pos, hist, dom["node_type"]], dim=-1)
        y = dom["vel"][t + 1]
        return x, y


class DomainBatchSampler(Sampler):
    def __init__(self, group_sizes, batch_size, shuffle):
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.groups = []
        start = 0
        for g in group_sizes:
            self.groups.append(torch.arange(start, start + g))
            start += g

    def __iter__(self):
        batches = []
        for g in self.groups:
            order = g[torch.randperm(len(g))] if self.shuffle else g
            for i in range(0, len(order), self.batch_size):
                batches.append(order[i : i + self.batch_size].tolist())
        if self.shuffle:
            batches = [batches[i] for i in torch.randperm(len(batches))]
        yield from batches

    def __len__(self):
        return sum((len(g) + self.batch_size - 1) // self.batch_size for g in self.groups)


def _split_timesteps(T, stride, split):
    ts = list(range(HISTORY, T - 1, stride))
    n = len(ts)
    if split == "train":
        return ts[: int(0.8 * n)]
    elif split == "valid":
        return ts[int(0.8 * n) : int(0.9 * n)]
    elif split == "test":
        return ts[int(0.9 * n) :]
    else:
        raise NotImplementedError


def _load_split(cfg, split):
    grid_vel = np.load(os.path.join(cfg.data.dataset_root, "eulerian_velocity.npy"), mmap_mode="r")
    T = grid_vel.shape[0]
    timesteps = _split_timesteps(T, cfg.data.split_interval, split)
    dataset = MixedDomainDataset(cfg.data.dataset_root, timesteps)
    sampler = DomainBatchSampler(dataset.group_sizes, cfg.training.batch_size, shuffle=(split == "train"))
    dataloader = DataLoader(dataset, batch_sampler=sampler, pin_memory=True)
    return dataloader


def load_train(cfg):
    return _load_split(cfg, "train")


def load_valid(cfg):
    return _load_split(cfg, "valid")


def load_test(cfg):
    return _load_split(cfg, "test")