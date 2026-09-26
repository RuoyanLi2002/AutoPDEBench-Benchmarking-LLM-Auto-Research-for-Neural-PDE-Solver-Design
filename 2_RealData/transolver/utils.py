import os
import glob
import pyarrow as pa
import numpy as np
import torch
from torch.utils.data import TensorDataset, DataLoader


def process_arrow_file(file_path):
    with pa.memory_map(file_path, 'rb') as source:
        table = pa.ipc.open_stream(source).read_all()
        row = table.to_pylist()[0] 

    T, H, W = row['shape_t'], row['shape_h'], row['shape_w']
    
    u = np.frombuffer(row['u'], dtype=np.float32).reshape(T, H, W)
    v = np.frombuffer(row['v'], dtype=np.float32).reshape(T, H, W)
    
    combined = np.stack([u, v], axis=1) 
    return torch.from_numpy(combined)

def load_single_dataset(cfg, split):
    files = sorted(glob.glob(os.path.join(cfg.data.dataset_root, "*.arrow")))
    # print(f"files: {files}")

    if split == "train":
        files = files[:int(0.8*len(files))][:5]
    elif split == "valid":
        files = files[int(0.8*len(files)):int(0.9*len(files))][0:1]
    elif split == "test":
        files = files[int(0.9*len(files)):][0:1]
    else:
        raise NotImplementedError
    
    all_x = []
    all_y = []

    for f_path in files:
        video_tensor = process_arrow_file(f_path)

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