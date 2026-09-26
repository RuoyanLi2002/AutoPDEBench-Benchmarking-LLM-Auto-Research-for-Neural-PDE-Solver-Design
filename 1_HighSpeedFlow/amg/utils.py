import os
import re

import h5py
import numpy as np
import torch
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader


_GRID_CACHE = {}


def make_unit_grid(spatial_shape):
    if spatial_shape not in _GRID_CACHE:
        axes = [torch.linspace(0.0, 1.0, s) for s in spatial_shape]
        mesh = torch.meshgrid(*axes, indexing="ij")
        pos = torch.stack([m.reshape(-1) for m in mesh], dim=1)
        _GRID_CACHE[spatial_shape] = pos.contiguous()
    return _GRID_CACHE[spatial_shape]


def field_to_graph(field, target_field=None):
    field = torch.as_tensor(field, dtype=torch.float32)
    num_channels = field.shape[0]
    spatial_shape = tuple(field.shape[1:])

    x = field.reshape(num_channels, -1).t().contiguous()
    pos = make_unit_grid(spatial_shape)
    data = Data(x=x, pos=pos)

    if target_field is not None:
        target_field = torch.as_tensor(target_field, dtype=torch.float32)
        data.y = target_field.reshape(target_field.shape[0], -1).t().contiguous()

    return data


def load_single_dataset(cfg, split):
    h5_files = []
    for f in os.listdir(cfg.data.dataset_root):
        match = re.match(r'plt_(\d+)\.h5', f)
        if match:
            h5_files.append((int(match.group(1)), f))
    h5_files.sort(key=lambda x: x[0])

    if split == "train":
        h5_files = h5_files[:int(0.8 * len(h5_files))]
    elif split == "valid":
        h5_files = h5_files[int(0.8 * len(h5_files)):int(0.9 * len(h5_files))]
    elif split == "test":
        h5_files = h5_files[int(0.9 * len(h5_files)):]
    else:
        raise NotImplementedError

    data_list = []
    for num, filename in h5_files:
        file_path = os.path.join(cfg.data.dataset_root, filename)
        with h5py.File(file_path, 'r') as hf:
            density = hf['density'][:]
            temperature = hf['temperature'][:]
            xVel = hf['xVel'][:]
            yVel = hf['yVel'][:]

        stacked = np.stack([temperature, density, xVel, yVel], axis=1)
        num_timesteps = stacked.shape[0]


        input_indices = np.arange(0, num_timesteps - 1, cfg.data.split_interval)
        for t in input_indices:
            data_list.append(field_to_graph(stacked[t], stacked[t + 1]))

    print(f"{split}: {len(data_list)} graphs")
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