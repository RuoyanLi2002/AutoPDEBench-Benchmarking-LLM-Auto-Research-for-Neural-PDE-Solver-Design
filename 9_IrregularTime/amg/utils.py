import os
import glob

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


def process_npz_file(file_path):
    """Read one .npz -> tensor [T, 1, H, W]."""
    data = np.load(file_path)
    vor = data["vor"].astype(np.float32)[0]                  # drop leading dim
    vor = vor[:, None, :, :]                                 # add channel dim -> [T, 1, H, W]
    return torch.from_numpy(vor)


def load_single_dataset(cfg, split):
    if split not in ("train", "valid", "test"):
        raise NotImplementedError(split)

    split_dir = os.path.join(cfg.data.dataset_root, split)
    files = sorted(glob.glob(os.path.join(split_dir, "*.npz")))

    interval = cfg.data.split_interval

    data_list = []
    for f_path in files:
        video = process_npz_file(f_path)                     # [T, 1, H, W]

        inputs = video[:-1:interval]                         # [n, 1, H, W]
        targets = video[1::interval]                         # [n, 1, H, W]

        for inp, tgt in zip(inputs, targets):
            data_list.append(field_to_graph(inp, tgt))       # inp/tgt are [1, H, W]

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