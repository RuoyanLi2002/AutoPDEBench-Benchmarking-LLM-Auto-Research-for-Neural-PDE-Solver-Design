import os

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


def _make_mask(H, W, generator=None):
    n_pixels = H * W
    n_visible = n_pixels // 2
    perm = torch.argsort(torch.rand(n_pixels, generator=generator))
    mask = torch.zeros(n_pixels)
    mask[perm[:n_visible]] = 1.0
    return mask.view(1, H, W)


def process_npy_file(file_path):
    data = np.load(file_path)
    return torch.from_numpy(data.astype(np.float32))


def field_to_graph(field, mask, target_field):
    field = torch.as_tensor(field, dtype=torch.float32)
    mask = torch.as_tensor(mask, dtype=torch.float32)
    num_channels = field.shape[0]
    spatial_shape = tuple(field.shape[1:])

    x_observed = field * mask
    x = x_observed.reshape(num_channels, -1).t().contiguous()
    pos = make_unit_grid(spatial_shape)

    data = Data(x=x, pos=pos)
    data.mask = mask.reshape(1, -1).t().contiguous()

    target_field = torch.as_tensor(target_field, dtype=torch.float32)
    data.y = target_field.reshape(target_field.shape[0], -1).t().contiguous()

    return data


def load_single_dataset(cfg, split):
    split_dir = {
        "train": "train",
        "valid": "valid",
        "test": "test",
    }
    if split not in split_dir:
        raise NotImplementedError(split)

    file_path = os.path.join(cfg.data.dataset_root, split_dir[split], f"{split_dir[split]}.npy")

    interval = cfg.data.split_interval
    horizon = 1

    video = process_npy_file(file_path)

    x = video[:, :-horizon:interval]
    y = video[:, horizon::interval]

    n_traj, n_t, H, W = x.shape
    x = x.reshape(n_traj * n_t, 1, H, W)
    y = y.reshape(n_traj * n_t, 1, H, W)

    is_train = (split == "train")

    data_list = []
    for idx in range(x.shape[0]):
        if is_train:
            mask = _make_mask(H, W)
        else:
            g = torch.Generator().manual_seed(idx)
            mask = _make_mask(H, W, generator=g)
        data_list.append(field_to_graph(x[idx], mask, y[idx]))

    print(f"{split}: {len(data_list)} graphs")
    if len(data_list) > 0:
        print(f"  x:    {tuple(data_list[0].x.shape)}")
        print(f"  pos:  {tuple(data_list[0].pos.shape)}")
        print(f"  mask: {tuple(data_list[0].mask.shape)}")
        print(f"  y:    {tuple(data_list[0].y.shape)}")

    dataloader = DataLoader(
        data_list,
        batch_size=cfg.training.batch_size,
        shuffle=is_train,
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