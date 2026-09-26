import os
import numpy as np
import torch
from torch.utils.data import TensorDataset, DataLoader


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


def process_npy_file(file_path):
    data = np.load(file_path).astype(np.float32)
    return torch.from_numpy(data)


def load_single_dataset(cfg, split):
    if split not in SPLIT_RANGES:
        raise ValueError(f"split must be one of {list(SPLIT_RANGES)}, got '{split}'")
    start, end = SPLIT_RANGES[split]


    horizon = 5
    if horizon < 1:
        raise ValueError(f"cfg.data.horizon must be >= 1, got {horizon}")

    all_x = []
    all_y = []
    all_pde = []

    for fname in PDE_FILES:
        pde_name = os.path.splitext(fname)[0]
        pde_id = PDE_TO_ID[pde_name]
        f_path = os.path.join(cfg.data.dataset_root, fname)
        traj = process_npy_file(f_path)  # (N, T, dim, H, W)

        if traj.shape[2] == 1:
            pad = torch.zeros_like(traj)
            traj = torch.cat([traj, pad], dim=2)  # (N, T, 2, H, W)

        traj = traj[start:end]  # (n_split, T, dim, H, W)

        if traj.shape[1] <= horizon:
            raise ValueError(
                f"{pde_name}: trajectory length T={traj.shape[1]} must exceed "
                f"horizon={horizon} to form any (input, target) pairs."
            )

        for n in range(traj.shape[0]):
            video_tensor = traj[n]  # (T, dim, H, W)

            x = video_tensor[:-horizon:cfg.data.split_interval]
            y = video_tensor[horizon::cfg.data.split_interval]

            all_x.append(x)
            all_y.append(y)
            all_pde.append(torch.full((x.shape[0],), pde_id, dtype=torch.long))

    full_x = torch.cat(all_x, dim=0)
    full_y = torch.cat(all_y, dim=0)
    full_pde = torch.cat(all_pde, dim=0)
    print(f"full_x: {full_x.shape}")
    print(f"full_y: {full_y.shape}")
    print(f"full_pde: {full_pde.shape}")

    dataset = TensorDataset(full_x, full_y, full_pde)
    shuffle = (split == "train")
    dataloader = DataLoader(dataset, batch_size=cfg.training.batch_size, shuffle=shuffle, pin_memory=True)

    return dataloader


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