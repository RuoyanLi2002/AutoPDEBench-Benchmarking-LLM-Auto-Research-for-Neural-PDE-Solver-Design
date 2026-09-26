import os
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader


def process_npy_file(file_path):
    data = np.load(file_path)  # (n_traj, T, H, W)
    return torch.from_numpy(data.astype(np.float32))


def _make_mask(H, W, generator=None):
    """Random binary mask with EXACTLY 50% of cells visible."""
    n_pixels = H * W
    n_visible = n_pixels // 2  # 8192 of 16384
    perm = torch.argsort(torch.rand(n_pixels, generator=generator))
    mask = torch.zeros(n_pixels)
    mask[perm[:n_visible]] = 1.0
    return mask.view(1, H, W)


class PartialObsDataset(Dataset):
    def __init__(self, x, y, resample, seed=0):
        assert x.shape == y.shape
        self.x = x  # (N, 1, H, W)
        self.y = y  # (N, 1, H, W)
        self.resample = resample
        self.seed = seed

    def __len__(self):
        return self.x.shape[0]

    def __getitem__(self, idx):
        _, H, W = self.x[idx].shape
        if self.resample:
            mask = _make_mask(H, W)  # new mask every access -> every epoch
        else:
            g = torch.Generator().manual_seed(self.seed + idx)  # fixed per sample
            mask = _make_mask(H, W, generator=g)
        x_observed = self.x[idx] * mask
        return x_observed, mask, self.y[idx]


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

    video = process_npy_file(file_path)  # (n_traj, T, H, W)

    # x[t] -> y[t+horizon], stepping by split_interval along time.
    x = video[:, :-horizon:interval]   # (n_traj, n_t, H, W)
    y = video[:, horizon::interval]    # (n_traj, n_t, H, W)

    n_traj, n_t, H, W = x.shape
    x = x.reshape(n_traj * n_t, 1, H, W)
    y = y.reshape(n_traj * n_t, 1, H, W)
    print(f"x: {x.shape}")
    print(f"y: {y.shape}")

    is_train = (split == "train")
    dataset = PartialObsDataset(x, y, resample=is_train)
    dataloader = DataLoader(
        dataset,
        batch_size=cfg.training.batch_size,
        shuffle=is_train,
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