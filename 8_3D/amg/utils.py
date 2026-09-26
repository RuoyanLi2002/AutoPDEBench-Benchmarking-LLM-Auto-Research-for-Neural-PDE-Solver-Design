import os
import glob

import h5py
import numpy as np
import torch
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader



_GRID_CACHE = {}


def make_unit_grid(spatial_shape):
    """Equally-spaced coordinate grid over the [0, 1]^d unit domain.

    spatial_shape is the field's spatial size, e.g. (nx, ny, nz). Returns a
    tensor of shape [prod(spatial_shape), len(spatial_shape)] whose row
    ordering matches a C-order (row-major) flatten of the field, so it lines
    up with the node features built below.
    """
    if spatial_shape not in _GRID_CACHE:
        axes = [torch.linspace(0.0, 1.0, s) for s in spatial_shape]
        mesh = torch.meshgrid(*axes, indexing="ij")              # tuple of [*spatial_shape]
        pos = torch.stack([m.reshape(-1) for m in mesh], dim=1)  # [N, d]
        _GRID_CACHE[spatial_shape] = pos.contiguous()
    return _GRID_CACHE[spatial_shape]


def field_to_graph(field, target_field=None):
    """Convert one snapshot [C, *spatial] -> Data(x=[N, C], pos=[N, d]).

    If target_field (the next frame) is given, attach it as y=[N, C]. PyG
    concatenates y along the node dimension during batching, so it stays
    row-aligned with x.
    """
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


def _stack_vector(arr):
    """Return a vector field with the size-3 component axis at position 1.

    Handles both (T, 3, nx, ny, nz) and (T, nx, ny, nz, 3) layouts.
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

    
    density = density[:, None, ...]

    combined = np.concatenate([density, magnetic_field, velocity], axis=1)
    return torch.from_numpy(combined)


def make_graphs_from_file(file_path, split_interval):
    """Read every trajectory in a file and return a list of Data objects.

    Next-step pairing is done WITHIN each trajectory so that no input/target
    pair ever straddles a trajectory boundary.
    """
    data_list = []
    with h5py.File(file_path, "r") as f:
        density_ds = f["t0_fields"]["density"]
        magnetic_ds = f["t1_fields"]["magnetic_field"]
        velocity_ds = f["t1_fields"]["velocity"]

        num_traj = density_ds.shape[0]
        for traj in range(num_traj):
            video_tensor = process_trajectory(
                density_ds[traj, ...],
                magnetic_ds[traj, ...],
                velocity_ds[traj, ...],
            )

            num_timesteps = video_tensor.shape[0]
            input_indices = np.arange(0, num_timesteps - 1, split_interval)
            for t in input_indices:
                data_list.append(field_to_graph(video_tensor[t], video_tensor[t + 1]))

    return data_list


def load_single_dataset(cfg, split):
    files = sorted(glob.glob(os.path.join(cfg.data.dataset_root, split, "*.hdf5")))
    if len(files) == 0:
        raise FileNotFoundError(
            f"No .hdf5 files found in {os.path.join(cfg.data.dataset_root, split)}"
        )

    data_list = []
    for f_path in files:
        data_list.extend(make_graphs_from_file(f_path, cfg.data.split_interval))

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