import os
import glob
import numpy as np
import torch
from torch.utils.data import TensorDataset, DataLoader


def process_npz_file(file_path):
    data = np.load(file_path)
    vor = data["vor"].astype(np.float32)[0]
    vor = vor[:, None, :, :]
    return torch.from_numpy(vor)


def load_single_dataset(cfg, split):
    split_dir = os.path.join(cfg.data.dataset_root, split)
    files = sorted(glob.glob(os.path.join(split_dir, "*.npz")))
    # print(f"files: {files}")

    if split == "train":
        files = files
    elif split == "valid":
        files = files
    elif split == "test":
        files = files
    else:
        raise NotImplementedError

    all_x = []
    all_y = []

    for f_path in files:
        video_tensor = process_npz_file(f_path)

        x = video_tensor[:-1:cfg.data.split_interval]
        y = video_tensor[1::cfg.data.split_interval]

        all_x.append(x)
        all_y.append(y)

    full_x = torch.cat(all_x, dim=0)
    full_y = torch.cat(all_y, dim=0)
    print(f"full_x: {full_x.shape}")
    print(f"full_y: {full_y.shape}")

    dataset = TensorDataset(full_x, full_y)
    dataloader = DataLoader(dataset, batch_size=cfg.training.batch_size, shuffle=True, pin_memory=True)

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