import os
import numpy as np
import torch
from torch.utils.data import TensorDataset, DataLoader


BC_SUBDIRS = [
    "dirichlet_time_dependent",
    "neumann_time_dependent",
    "periodic_time_dependent",
]

BC_TO_ID = {name: i for i, name in enumerate(BC_SUBDIRS)}
ID_TO_BC = {i: name for name, i in BC_TO_ID.items()}

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

    all_x = []
    all_y = []
    all_bc = []

    for subdir in BC_SUBDIRS:
        bc_id = BC_TO_ID[subdir]
        f_path = os.path.join(cfg.data.dataset_root, subdir, "trajectories.npy")
        traj = process_npy_file(f_path)  # (N, T, dim, H, W)

        traj = traj[start:end]  # (n_split, T, dim, H, W)

        for n in range(traj.shape[0]):
            video_tensor = traj[n]  # (T, dim, H, W)

            x = video_tensor[:-1:cfg.data.split_interval]
            y = video_tensor[1::cfg.data.split_interval]

            all_x.append(x)
            all_y.append(y)
            all_bc.append(torch.full((x.shape[0],), bc_id, dtype=torch.long))

    full_x = torch.cat(all_x, dim=0)
    full_y = torch.cat(all_y, dim=0)
    full_bc = torch.cat(all_bc, dim=0)
    print(f"full_x: {full_x.shape}")
    print(f"full_y: {full_y.shape}")
    print(f"full_bc: {full_bc.shape}")

    dataset = TensorDataset(full_x, full_y, full_bc)
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