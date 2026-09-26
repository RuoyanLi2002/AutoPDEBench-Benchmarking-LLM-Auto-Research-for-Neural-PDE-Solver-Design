import os
import glob

import pyarrow as pa
import numpy as np
import torch
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader


_GRID_CACHE = {}


def make_unit_grid(spatial_shape):
    """Equally-spaced coordinate grid over the [0, 1]^d unit domain.

    spatial_shape is the field's spatial size, e.g. (H, W). Returns a tensor of
    shape [prod(spatial_shape), len(spatial_shape)] whose row ordering matches a
    C-order (row-major) flatten of the field, so it lines up with the node
    features built below.
    """
    if spatial_shape not in _GRID_CACHE:
        axes = [torch.linspace(0.0, 1.0, s) for s in spatial_shape]
        mesh = torch.meshgrid(*axes, indexing="ij")          # tuple of [*spatial_shape]
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


def process_arrow_file(file_path):
    """Read one arrow file -> tensor [num_timesteps, 2, H, W] (channels = u, v)."""
    with pa.memory_map(file_path, 'rb') as source:
        table = pa.ipc.open_stream(source).read_all()
        row = table.to_pylist()[0]

    T, H, W = row['shape_t'], row['shape_h'], row['shape_w']

    u = np.frombuffer(row['u'], dtype=np.float32).reshape(T, H, W)
    v = np.frombuffer(row['v'], dtype=np.float32).reshape(T, H, W)

    combined = np.stack([u, v], axis=1)                      # [T, 2, H, W]
    return torch.from_numpy(combined)


def load_single_dataset(cfg, split):
    files = sorted(glob.glob(os.path.join(cfg.data.dataset_root, "*.arrow")))

    if split == "train":
        files = files[:int(0.8 * len(files))][:5]
    elif split == "valid":
        files = files[int(0.8 * len(files)):int(0.9 * len(files))][:1]
    elif split == "test":
        files = files[int(0.9 * len(files)):][:1]
    else:
        raise NotImplementedError

    data_list = []
    for f_path in files:
        stacked = process_arrow_file(f_path)
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