import os
import numpy as np
import torch
from torch_geometric.data import Data, DataLoader


def _grid_pos(grid_size):
    """
    Node coordinates for the 1D grid, normalized to [0, 1].
    Returns: (grid_size, 1)
    """
    return torch.linspace(0.0, 1.0, grid_size, dtype=torch.float32).unsqueeze(-1)


def load_single_dataset(cfg, split):
    data = np.load(os.path.join(cfg.data.dataset_root, "ks.npy"))

    if split == "train":
        data = data[:100]
    elif split == "valid":
        data = data[100:110]
    elif split == "test":
        data = data[110:120]
    else:
        raise NotImplementedError

    # squeeze channel dim -> (num_traj, T, GridSize)
    data = data[:, :, 0, :]

    grid_size = data.shape[-1]
    pos = _grid_pos(grid_size)  # (GridSize, 1)

    all_data = []
    for traj in data:
        num_timesteps = traj.shape[0]
        input_indices = np.arange(0, num_timesteps - 1, cfg.data.split_interval)
        target_indices = input_indices + 1

        x_traj = torch.tensor(traj[input_indices], dtype=torch.float32).unsqueeze(-1)
        y_traj = torch.tensor(traj[target_indices], dtype=torch.float32).unsqueeze(-1)

        for x_i, y_i in zip(x_traj, y_traj):
            all_data.append(Data(x=x_i, y=y_i, pos=pos))  # (GridSize, 1) each

    print(f"all_data: {len(all_data)}  (GridSize: {grid_size})")

    shuffle = (split == "train")
    dataloader = DataLoader(
        all_data,
        batch_size=cfg.training.batch_size,
        shuffle=shuffle,
        num_workers=0,
        pin_memory=True,
    )

    return dataloader


def load_trajectories(cfg, split):
    data = np.load(os.path.join(cfg.data.dataset_root, "ks.npy"))

    if split == "train":
        data = data[:100]
    elif split == "valid":
        data = data[100:110]
    elif split == "test":
        data = data[110:120]
    else:
        raise NotImplementedError

    data = data[:, :, 0, :]
    return torch.tensor(data, dtype=torch.float32)


def _load_split(cfg, split):
    file_path = f"{cfg.data.data_save_path}/{split}.pth"
    if os.path.exists(file_path):
        print(f"{split}.pth already exists. Load from {file_path}")
        all_data = torch.load(file_path)
        print(len(all_data))
        shuffle = (split == "train")
        dataloader = DataLoader(all_data, batch_size=cfg.training.batch_size, shuffle=shuffle)
    else:
        print(f"{split}.pth does not exists. Create dataset")
        dataloader = load_single_dataset(cfg, split)

    return dataloader


def load_train(cfg):
    return _load_split(cfg, "train")


def load_valid(cfg):
    return _load_split(cfg, "valid")


def load_test(cfg):
    return _load_split(cfg, "test")