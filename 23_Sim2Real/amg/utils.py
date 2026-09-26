import os
import glob

import h5py
import numpy as np
import torch
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader



_GRID_CACHE = {}


def make_unit_grid(spatial_shape):
    if spatial_shape not in _GRID_CACHE:
        axes = [torch.linspace(0.0, 1.0, s) for s in spatial_shape]
        mesh = torch.meshgrid(*axes, indexing="ij")          # tuple of [*spatial_shape]
        pos = torch.stack([m.reshape(-1) for m in mesh], dim=1)  # [N, d]
        _GRID_CACHE[spatial_shape] = pos.contiguous()
    return _GRID_CACHE[spatial_shape]


def field_to_graph(field, target_field=None):
    field = torch.as_tensor(field, dtype=torch.float32)
    num_channels = field.shape[0]
    spatial_shape = tuple(field.shape[1:])

    x = field.reshape(num_channels, -1).t().contiguous()     # [N, C]
    pos = make_unit_grid(spatial_shape)                      # [N, d]
    data = Data(x=x, pos=pos)

    if target_field is not None:
        target_field = torch.as_tensor(target_field, dtype=torch.float32)
        data.y = target_field.reshape(target_field.shape[0], -1).t().contiguous()  # [N, C]

    return data


def process_h5_file(file_path, time_stride=1, max_frames=None):
    """Read one FSI h5 file -> tensor [T, 2, 128, 128] with channels (u, v)."""
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
    # test: use all real files

    max_files = getattr(cfg.data, "max_files", None)
    if max_files is not None:
        files = files[:max_files]

    return files


def load_single_dataset(cfg, split):
    files = _get_files(cfg, split)
    print(f"[{split}] using {len(files)} files")

    time_stride = getattr(cfg.data, "time_stride", 1)
    max_frames = getattr(cfg.data, "max_frames", None)

    data_list = []
    for f_path in files:
        video_tensor = process_h5_file(f_path, time_stride=time_stride, max_frames=max_frames)
        num_timesteps = video_tensor.shape[0]

        input_indices = np.arange(0, num_timesteps - 1, cfg.data.split_interval)
        for t in input_indices:
            data_list.append(field_to_graph(video_tensor[t], video_tensor[t + 1]))

    print(f"[{split}] {len(data_list)} graphs")
    if len(data_list) > 0:
        print(f"  x:   {tuple(data_list[0].x.shape)}")
        print(f"  pos: {tuple(data_list[0].pos.shape)}")
        print(f"  y:   {tuple(data_list[0].y.shape)}")

    shuffle = (split == "train")
    dataloader = DataLoader(
        data_list,
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
        all_data = torch.load(file_path, weights_only=False)
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