import os
import yaml
import glob
import pyarrow as pa
import numpy as np
import torch
from torch.utils.data import TensorDataset, DataLoader
from types import SimpleNamespace




# -----------------------------
# Config loading
# -----------------------------

class Config(SimpleNamespace):
    """Namespace supporting nested attribute access (e.g. cfg.data.dataset_root)."""
    pass


def _dict_to_namespace(d):
    if isinstance(d, dict):
        return Config(**{k: _dict_to_namespace(v) for k, v in d.items()})
    elif isinstance(d, list):
        return [_dict_to_namespace(v) for v in d]
    else:
        return d


def to_dict(ns):
    """Recursively convert a Config namespace back to a plain dict.

    Useful for unpacking config sub-sections into class constructors,
    e.g. `Adam(model.parameters(), **to_dict(cfg.training.optimizer.params))`.
    """
    if isinstance(ns, SimpleNamespace):
        return {k: to_dict(v) for k, v in vars(ns).items()}
    elif isinstance(ns, list):
        return [to_dict(v) for v in ns]
    else:
        return ns


def _deep_merge(base, override):
    """Recursively merge `override` into `base`. Values in `override` win on conflict."""
    result = dict(base)
    for k, v in override.items():
        if k in result and isinstance(result[k], dict) and isinstance(v, dict):
            result[k] = _deep_merge(result[k], v)
        else:
            result[k] = v
    return result


def _resolve_base(config):
    """Recursively resolve `base_config` references inside a loaded config dict."""
    if "base_config" in config:
        base_path = config.pop("base_config")
        with open(base_path, "r") as f:
            base = yaml.safe_load(f) or {}
        base = _resolve_base(base)
        config = _deep_merge(base, config)
    return config


def load_config(config_path):
    """Load a YAML config file, resolving any `base_config` chain into a single namespace."""
    with open(config_path, "r") as f:
        config = yaml.safe_load(f) or {}
    config = _resolve_base(config)
    return _dict_to_namespace(config)


# -----------------------------
# Data loading
# -----------------------------


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