import os
import yaml
import glob
import os.path as osp
import re
import h5py
import pyvista as pv
import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset
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





VEL_MEAN = (-0.05811476, -7.05567631)
VEL_STD = (14.46373118,8.72693759)

SPLIT_RANGES = {"train": (0, 10000), "valid": (10000, 11000), "test": (11000, 12000)}
KEEP_AXES = [0, 2]  # drop y (2D simulation in the x-z plane)


def vel_normalizer(vel):
    vel_mean = torch.tensor(VEL_MEAN, device=vel.device, dtype=vel.dtype)
    vel_std = torch.tensor(VEL_STD, device=vel.device, dtype=vel.dtype)
    return (vel - vel_mean) / vel_std


def vel_unnormalizer(normalized_vel):
    vel_mean = torch.tensor(VEL_MEAN, device=normalized_vel.device, dtype=normalized_vel.dtype)
    vel_std = torch.tensor(VEL_STD, device=normalized_vel.device, dtype=normalized_vel.dtype)
    return (normalized_vel * vel_std) + vel_mean


def _sorted_files(data_dir):
    key = lambda p: int(re.search(r"_(\d+)\.vtk$", p).group(1))
    return sorted(glob.glob(f"{data_dir}/PartFluid_*.vtk"), key=key)


def _read_frame(path):
    mesh = pv.read(path)
    idp = np.asarray(mesh.point_data["Idp"]).astype(np.int64)
    order = np.argsort(idp)  # sort by Idp so per-frame lookup can use searchsorted
    return {
        "idp": np.ascontiguousarray(idp[order]),
        "pos": np.ascontiguousarray(np.asarray(mesh.points, dtype=np.float32)[order][:, KEEP_AXES]),
        "vel": np.ascontiguousarray(np.asarray(mesh.point_data["Vel"], dtype=np.float32)[order][:, KEEP_AXES]),
    }


def _load_frames(cfg, split):
    """Read all frames of a split into memory (list of dicts with variable N).
    Cached to {data_save_path}/{split}_frames.pt so the VTK parse happens once."""
    cache = osp.join(cfg.data.data_save_path, f"{split}_frames.pt")
    if os.path.exists(cache):
        print(f"Loading cached frames from {cache}")
        return torch.load(cache)

    files = _sorted_files(cfg.data.dataset_root)
    lo, hi = SPLIT_RANGES[split]
    if len(files) < hi:
        raise RuntimeError(f"Need {hi} files for split '{split}', found {len(files)}")

    print(f"[{split}] reading {hi - lo} VTK files ...")
    frames = []
    for t, path in enumerate(files[lo:hi]):
        frames.append(_read_frame(path))
        if (t + 1) % 500 == 0:
            print(f"  [{split}] {t + 1}/{hi - lo}", flush=True)

    os.makedirs(cfg.data.data_save_path, exist_ok=True)
    torch.save(frames, cache)
    print(f"[{split}] cached {len(frames)} frames to {cache}")
    return frames


class ParticleWindowDataset(Dataset):
    def __init__(self, frames, cfg, split):
        self.frames = frames
        self.interval = cfg.data.split_interval
        self.starts = list(range(0, len(frames) - 1, self.interval))

    def __len__(self):
        return len(self.starts)

    def __getitem__(self, i):
        s = self.starts[i]
        in_frame = self.frames[s]
        out_frame = self.frames[s + 1]

        vel = torch.from_numpy(in_frame["vel"])                  # (N_in, 2)

        x = torch.cat(
            [torch.from_numpy(in_frame["pos"]), vel_normalizer(vel)], dim=-1
        )                                                        # (N_in, 4)

        x_out = torch.from_numpy(out_frame["pos"])               # (N_out, 2)
        y = vel_normalizer(torch.from_numpy(out_frame["vel"]))   # (N_out, 2)

        return {"x": x, "x_out": x_out, "y": y}


def collate_padded(batch):
    """Pad variable-length samples to the batch max; return masks."""
    B = len(batch)
    n_in = max(b["x"].shape[0] for b in batch)
    n_out = max(b["x_out"].shape[0] for b in batch)
    F = batch[0]["x"].shape[-1]

    x = torch.zeros(B, n_in, F)
    in_mask = torch.zeros(B, n_in)
    x_out = torch.zeros(B, n_out, 2)
    y = torch.zeros(B, n_out, 2)
    out_mask = torch.zeros(B, n_out)

    for b, s in enumerate(batch):
        ni, no = s["x"].shape[0], s["x_out"].shape[0]
        x[b, :ni] = s["x"]
        in_mask[b, :ni] = 1.0
        x_out[b, :no] = s["x_out"]
        y[b, :no] = s["y"]
        out_mask[b, :no] = 1.0

    return {"x": x, "in_mask": in_mask, "x_out": x_out, "y": y, "out_mask": out_mask}


# ---------------------------------------------------------------------------
# loaders
# ---------------------------------------------------------------------------

def _load_split(cfg, split):
    frames = _load_frames(cfg, split)
    dataset = ParticleWindowDataset(frames, cfg, split)
    print(f"[{split}] {len(dataset)} samples")
    return DataLoader(
        dataset,
        batch_size=cfg.training.batch_size,
        shuffle=(split == "train"),
        collate_fn=collate_padded,
        num_workers=getattr(cfg.training, "num_workers", 0),
        pin_memory=True,
    )


def load_train(cfg):
    return _load_split(cfg, "train")


def load_valid(cfg):
    return _load_split(cfg, "valid")


def load_test(cfg):
    return _load_split(cfg, "test")