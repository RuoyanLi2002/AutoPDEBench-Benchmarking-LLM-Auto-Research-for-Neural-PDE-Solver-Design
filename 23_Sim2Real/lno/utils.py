import os
import glob
import h5py
import numpy as np
import torch
from torch.utils.data import TensorDataset, DataLoader


def process_h5_file(file_path, time_stride=1, max_frames=None):
    """Read one FSI h5 file -> (T, 2, 128, 128) tensor with channels (u, v)."""
    with h5py.File(file_path, "r") as f:
        u = f["measured_data/u"][...]  # (T, 128, 128) float32
        v = f["measured_data/v"][...]  # (T, 128, 128) float32

    combined = np.stack([u, v], axis=1)  # (T, 2, 128, 128)

    if time_stride > 1:
        combined = combined[::time_stride]
    if max_frames is not None:
        combined = combined[:max_frames]

    return torch.from_numpy(np.ascontiguousarray(combined))


def _get_files(cfg, split):
    """train/valid come from numerical (sim), test comes from real."""
    if split in ("train", "valid"):
        domain = "numerical"
    elif split == "test":
        domain = "real"
    else:
        raise NotImplementedError(split)

    files = sorted(glob.glob(os.path.join(cfg.data.dataset_root, domain, "*.h5")))
    assert len(files) > 0, f"No h5 files found under {cfg.data.dataset_root}/{domain}"

    if split == "train":
        files = files[: int(0.8 * len(files))]
    elif split == "valid":
        files = files[int(0.8 * len(files)):]

    max_files = getattr(cfg.data, "max_files", None)
    if max_files is not None:
        files = files[:max_files]

    return files


def load_single_dataset(cfg, split):
    files = _get_files(cfg, split)[:10]
    print(f"[{split}] using {len(files)} files")

    time_stride = getattr(cfg.data, "time_stride", 1)
    max_frames = getattr(cfg.data, "max_frames", None)

    all_x, all_y = [], []
    for f_path in files:
        video_tensor = process_h5_file(f_path, time_stride=time_stride, max_frames=max_frames)

        x = video_tensor[:-1:cfg.data.split_interval]
        y = video_tensor[1::cfg.data.split_interval]

        all_x.append(x)
        all_y.append(y)

    full_x = torch.cat(all_x, dim=0)
    full_y = torch.cat(all_y, dim=0)
    print(f"[{split}] full_x: {full_x.shape}")
    print(f"[{split}] full_y: {full_y.shape}")

    dataset = TensorDataset(full_x, full_y)
    shuffle = (split == "train")
    dataloader = DataLoader(dataset, batch_size=cfg.training.batch_size,
                            shuffle=shuffle, pin_memory=True)
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
        print(f"{split}.pth does not exist. Create dataset")
        dataloader = load_single_dataset(cfg, split)

    return dataloader


def load_train(cfg):
    return _load_split(cfg, "train")


def load_valid(cfg):
    return _load_split(cfg, "valid")


def load_test(cfg):
    return _load_split(cfg, "test")