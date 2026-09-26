import os
import glob
import re

import numpy as np
import torch
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader


_GRID_CACHE = {}


RE_MIN = 100.0
RE_MAX = 5000.0


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


def field_to_graph(field, target_field=None, param=None):
    """Convert one snapshot [C, *spatial] -> Data(x=[N, C], pos=[N, d]).

    If target_field (the t+horizon frame) is given, attach it as y=[N, C]. PyG
    concatenates y along the node dimension during batching, so it stays
    row-aligned with x.

    If param (the normalized Re scalar) is given, attach it as a graph-level
    attribute of shape [1, 1]. PyG concatenates graph-level attrs along dim 0
    during batching, giving param shape [batch_size, 1].
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

    if param is not None:
        data.param = torch.as_tensor([[param]], dtype=torch.float32)

    return data


def _parse_re(folder_path):
    """Extract the integer Re value from a folder like '.../Re_1300'."""
    name = os.path.basename(folder_path.rstrip("/"))
    m = re.search(r"Re_(\d+)", name)
    if m is None:
        raise ValueError(f"Could not parse Re value from {folder_path}")
    return float(m.group(1))


def _normalize_re(re_value):
    return (re_value - RE_MIN) / (RE_MAX - RE_MIN)


def process_npy_file(file_path):
    """Read one Re_*.npy -> tensor [n_traj, T, H, W]."""
    data = np.load(file_path)                                # (20, 21, 128, 128)
    return torch.from_numpy(data.astype(np.float32))


def load_single_dataset(cfg, split):
    # Each split lives in its own root: ns_train / ns_valid / ns_test.
    split_dir = {
        "train": "ns_train",
        "valid": "ns_valid",
        "test": "ns_test",
    }
    if split not in split_dir:
        raise NotImplementedError(split)

    root = os.path.join(cfg.data.dataset_root, split_dir[split])
    re_folders = sorted(glob.glob(os.path.join(root, "Re_*")))
    re_folders = [f for f in re_folders if os.path.isdir(f)]

    interval = cfg.data.split_interval
    horizon = 5  # predict t+5 from t

    data_list = []
    for folder in re_folders:
        re_value = _parse_re(folder)
        re_norm = _normalize_re(re_value)

        npy_files = sorted(glob.glob(os.path.join(folder, "*.npy")))
        for f_path in npy_files:
            video = process_npy_file(f_path)                 # [n_traj, T, H, W]
            n_traj, num_timesteps = video.shape[0], video.shape[1]


            input_indices = np.arange(0, num_timesteps - horizon, interval)
            for traj in range(n_traj):
                for t in input_indices:
                    inp = video[traj, t].unsqueeze(0)            # [1, H, W]
                    tgt = video[traj, t + horizon].unsqueeze(0)  # [1, H, W]
                    data_list.append(field_to_graph(inp, tgt, param=re_norm))

    print(f"{split}: {len(data_list)} graphs")
    if len(data_list) > 0:
        print(f"  x:     {tuple(data_list[0].x.shape)}")
        print(f"  pos:   {tuple(data_list[0].pos.shape)}")
        print(f"  y:     {tuple(data_list[0].y.shape)}")
        print(f"  param: {tuple(data_list[0].param.shape)}")

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