import os

import numpy as np
import torch
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader


PDE_FILES = [
    "advection_diffusion_2d.npy",
    "burgers_2d.npy",
    "gray_scott_2d.npy",
    "navier_stokes_vorticity_2d.npy",
]

PDE_SUBDIRS = [os.path.splitext(f)[0] for f in PDE_FILES]

PDE_TO_ID = {name: i for i, name in enumerate(PDE_SUBDIRS)}
ID_TO_PDE = {i: name for name, i in PDE_TO_ID.items()}

SPLIT_RANGES = {
    "train": (0, 20),
    "valid": (20, 25),
    "test": (25, 30),
}


_GRID_CACHE = {}


def make_unit_grid(spatial_shape):
    if spatial_shape not in _GRID_CACHE:
        axes = [torch.linspace(0.0, 1.0, s) for s in spatial_shape]
        mesh = torch.meshgrid(*axes, indexing="ij")
        pos = torch.stack([m.reshape(-1) for m in mesh], dim=1)
        _GRID_CACHE[spatial_shape] = pos.contiguous()
    return _GRID_CACHE[spatial_shape]


def field_to_graph(field, target_field=None, pde_id=None):
    field = torch.as_tensor(field, dtype=torch.float32)
    num_channels = field.shape[0]
    spatial_shape = tuple(field.shape[1:])

    x = field.reshape(num_channels, -1).t().contiguous()
    pos = make_unit_grid(spatial_shape)
    data = Data(x=x, pos=pos)

    if target_field is not None:
        target_field = torch.as_tensor(target_field, dtype=torch.float32)
        data.y = target_field.reshape(target_field.shape[0], -1).t().contiguous()

    if pde_id is not None:
        data.pde = torch.tensor([pde_id], dtype=torch.long)

    return data


def process_npy_file(file_path):
    data = np.load(file_path).astype(np.float32)
    return torch.from_numpy(data)


def load_single_dataset(cfg, split):
    if split not in SPLIT_RANGES:
        raise ValueError(f"split must be one of {list(SPLIT_RANGES)}, got '{split}'")
    start, end = SPLIT_RANGES[split]

    interval = cfg.data.split_interval



    horizon = 5
    if horizon < 1:
        raise ValueError(f"cfg.data.horizon must be >= 1, got {horizon}")

    data_list = []
    for fname in PDE_FILES:
        pde_name = os.path.splitext(fname)[0]
        pde_id = PDE_TO_ID[pde_name]
        f_path = os.path.join(cfg.data.dataset_root, fname)
        traj = process_npy_file(f_path)                      # (N, T, dim, H, W)

        if traj.shape[2] == 1:
            pad = torch.zeros_like(traj)
            traj = torch.cat([traj, pad], dim=2)             # (N, T, 2, H, W)

        traj = traj[start:end]                               # (n_split, T, dim, H, W)

        if traj.shape[1] <= horizon:
            raise ValueError(
                f"{pde_name}: trajectory length T={traj.shape[1]} must exceed "
                f"horizon={horizon} to form any (input, target) pairs."
            )

        for n in range(traj.shape[0]):
            video_tensor = traj[n]                           # (T, dim, H, W)

            inputs = video_tensor[:-horizon:interval]        # [n, dim, H, W]
            targets = video_tensor[horizon::interval]        # [n, dim, H, W]

            for inp, tgt in zip(inputs, targets):
                data_list.append(field_to_graph(inp, tgt, pde_id))  # inp/tgt are [dim, H, W]

    print(f"{split}: {len(data_list)} graphs")
    if len(data_list) > 0:
        print(f"  x:   {tuple(data_list[0].x.shape)}")
        print(f"  pos: {tuple(data_list[0].pos.shape)}")
        print(f"  y:   {tuple(data_list[0].y.shape)}")
        print(f"  pde: {tuple(data_list[0].pde.shape)}")

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