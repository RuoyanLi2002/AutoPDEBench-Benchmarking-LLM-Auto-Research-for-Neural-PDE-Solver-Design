import os
import glob
import h5py
import numpy as np
import torch
from torch.utils.data import TensorDataset, DataLoader


def _stack_vector(arr):
    """Return a vector field with the size-3 component axis at position 1.

    Handles both (..., 3, nx, ny, nz) and (..., nx, ny, nz, 3) layouts.
    `arr` here is per-trajectory: (T, 3, nx, ny, nz) or (T, nx, ny, nz, 3).
    """
    if arr.shape[1] == 3:
        return arr
    if arr.shape[-1] == 3:
        return np.moveaxis(arr, -1, 1)
    raise ValueError(f"Cannot find vector (size-3) axis in shape {arr.shape}")


def process_trajectory(density, magnetic_field, velocity):
    """Combine the three fields for ONE trajectory into (T, C, nx, ny, nz).

    Channels: density (1) + magnetic_field (3) + velocity (3) = 7.
    """
    density = np.asarray(density, dtype=np.float32)
    magnetic_field = _stack_vector(np.asarray(magnetic_field, dtype=np.float32))
    velocity = _stack_vector(np.asarray(velocity, dtype=np.float32))

    # Scalar density gets a channel dim: (T, nx, ny, nz) -> (T, 1, nx, ny, nz)
    density = density[:, None, ...]

    combined = np.concatenate([density, magnetic_field, velocity], axis=1)  # (T, 7, nx, ny, nz)
    return torch.from_numpy(combined)


def make_pairs_from_file(file_path, split_interval):
    """Read every trajectory in a file and return per-trajectory (x, y) pairs.

    Next-step pairing is done WITHIN each trajectory so that no input/target
    pair ever straddles a trajectory boundary.
    """
    xs, ys = [], []
    with h5py.File(file_path, "r") as f:
        density_ds = f["t0_fields"]["density"]
        magnetic_ds = f["t1_fields"]["magnetic_field"]
        velocity_ds = f["t1_fields"]["velocity"]

        num_traj = density_ds.shape[0]  # leading axis is the trajectory index
        for traj in range(num_traj):
            video_tensor = process_trajectory(
                density_ds[traj, ...],
                magnetic_ds[traj, ...],
                velocity_ds[traj, ...],
            )  # (T, C, nx, ny, nz)
            print(f"video_tensor: {video_tensor.shape}")

            x = video_tensor[:-1:split_interval]   # state at t
            y = video_tensor[1::split_interval]    # state at t+1
            xs.append(x)
            ys.append(y)

    return xs, ys


def load_single_dataset(cfg, split):
    files = sorted(glob.glob(os.path.join(cfg.data.dataset_root, split, "*.hdf5")))
    if len(files) == 0:
        raise FileNotFoundError(
            f"No .hdf5 files found in {os.path.join(cfg.data.dataset_root, split)}"
        )

    all_x = []
    all_y = []

    for f_path in files:
        xs, ys = make_pairs_from_file(f_path, cfg.data.split_interval)
        all_x.extend(xs)
        all_y.extend(ys)

    full_x = torch.cat(all_x, dim=0)
    full_y = torch.cat(all_y, dim=0)
    print(f"full_x: {full_x.shape}")
    print(f"full_y: {full_y.shape}")

    dataset = TensorDataset(full_x, full_y)
    shuffle = (split == "train")
    dataloader = DataLoader(
        dataset,
        batch_size=cfg.training.batch_size,
        shuffle=shuffle,
        pin_memory=True,
    )
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