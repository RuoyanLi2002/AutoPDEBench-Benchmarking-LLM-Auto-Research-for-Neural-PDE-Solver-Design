import os
import yaml
import glob
import os.path as osp
import h5py
import numpy as np
import torch
from torch_geometric.data import Data, DataLoader
from types import SimpleNamespace

from scipy.spatial import cKDTree

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




noise_std = 6.7e-4

VEL_MEAN = torch.tensor([
    [-2.5686059e-06, -3.1346712e-08],
    [ 1.1389641e-05,  1.5440160e-06],
], dtype=torch.float32)

VEL_STD = torch.tensor([
    [ 1.0721595e-03,  1.0510616e-03],
    [ 1.7509022e-03,  6.6960320e-04],
], dtype=torch.float32)


def vel_normalizer(vel, particle_type):
    """
    vel:           (..., N, 2)
    particle_type: (N,) long tensor with values in {0, 1}
    """
    mean = VEL_MEAN.to(device=vel.device, dtype=vel.dtype)[particle_type]  # (N, 2)
    std  = VEL_STD.to(device=vel.device, dtype=vel.dtype)[particle_type]   # (N, 2)

    return (vel - mean) / std


def vel_unnormalizer(normalized_vel, particle_type):
    """
    Input: (T, num_particles, 2)
    Output: (T, num_particles, 2)
    """
    mean = VEL_MEAN.to(device=normalized_vel.device, dtype=normalized_vel.dtype)[particle_type]  # (N, 2)
    std  = VEL_STD.to(device=normalized_vel.device, dtype=normalized_vel.dtype)[particle_type]   # (N, 2)

    return (normalized_vel * std) + mean


def _time_diff(input_sequence):
    return input_sequence[:, 1:] - input_sequence[:, :-1]


def _get_random_walk_noise(position_sequence, noise_std_last_step):
    """Returns random-walk noise in the velocity applied to the position."""
    velocity_sequence = _time_diff(position_sequence)
    num_velocities = velocity_sequence.shape[1]
    velocity_sequence_noise = torch.randn(list(velocity_sequence.shape)) * (
        noise_std_last_step / num_velocities ** 0.5
    )
    velocity_sequence_noise = torch.cumsum(velocity_sequence_noise, dim=1)
    position_sequence_noise = torch.cat([
        torch.zeros_like(velocity_sequence_noise[:, 0:1]),
        torch.cumsum(velocity_sequence_noise, dim=1)], dim=1)
    return position_sequence_noise


def _create_edge(second_last_frame, connectivity_radius):
    pos = second_last_frame.detach().cpu().numpy()
    tree = cKDTree(pos)
    pairs = tree.query_pairs(connectivity_radius, output_type="ndarray")
    pairs = np.concatenate([pairs, pairs[:, ::-1]], axis=0)
    src = torch.from_numpy(pairs[:, 0]).long()
    dst = torch.from_numpy(pairs[:, 1]).long()
    edge_index = torch.stack([src, dst], dim=0)

    rel = second_last_frame[src] - second_last_frame[dst]
    dist = torch.norm(rel, dim=-1, keepdim=True)
    edge_attr = torch.cat([rel, dist], dim=-1)
    return edge_index, edge_attr


def _create_subsequences(position, particle_information, sloshing_motion, cfg, split):
    seq_length = cfg.data.seq_length
    interval = cfg.data.split_interval if (split == "train") else 10

    if not torch.is_tensor(particle_information):
        particle_information = torch.as_tensor(particle_information)

    ptype = (particle_information.squeeze(-1).long()).to(position.device)

    if not torch.is_tensor(sloshing_motion):
        sloshing_motion = torch.as_tensor(sloshing_motion)
    sloshing_motion = sloshing_motion.to(position.device, dtype=position.dtype)

    num_frames, num_particles, _ = position.shape
    ls_data = []

    start_idx = 0
    while (start_idx + seq_length) < num_frames:
        subseq = position[start_idx:start_idx + seq_length]  # (seq_length, n_p, 2)

        x = subseq[:-1]                  # (seq_length-1, n_p, 2)
        y = subseq[-1] - subseq[-2]
        y = vel_normalizer(y, ptype)            # (n_p, 2)

        sampled_noise = _get_random_walk_noise(
            x.permute(1, 0, 2), noise_std_last_step=noise_std)
        sampled_noise = sampled_noise.permute(1, 0, 2)
        noised_x = x + sampled_noise

        velocity = noised_x[1:, :, :] - noised_x[:-1, :, :]
        velocity = vel_normalizer(velocity, ptype)
        velocity = velocity.permute(1, 0, 2).contiguous()
        velocity = velocity.view(num_particles, -1)

        
        
        slosh_window = sloshing_motion[start_idx:start_idx + seq_length]  # (seq_length, 6)
        slosh_feat = slosh_window.reshape(-1)                             # (seq_length*6,)
        slosh_feat = slosh_feat.unsqueeze(0).expand(num_particles, -1)    # (n_p, seq_length*6)

        node_feature = torch.cat(
            [x[-1, :, :], velocity, particle_information, slosh_feat], dim=-1)

        data = Data(
            x=node_feature.unsqueeze(0),
            pos=x[-1, :, :].unsqueeze(0),
            y=y.unsqueeze(0),
            particle_information=particle_information.unsqueeze(0),
        )
        ls_data.append(data)

        start_idx += interval

    return ls_data


def load_single_dataset(cfg, split):
    split_file = {
        "train": "train.h5",
        "valid": "valid.h5",
        "test": "test.h5",
    }
    if split not in split_file:
        raise NotImplementedError(split)

    p = osp.join(cfg.data.dataset_root, split_file[split])

    all_data = []
    with h5py.File(p, "r") as f:
        particle_information = torch.from_numpy(f["particle_type"][:]) - 1.0
        position = torch.from_numpy(f["positions"][:])
        sloshing_motion = torch.from_numpy(f["sloshing_motion"][:])

        ls_data = _create_subsequences(position, particle_information, sloshing_motion, cfg, split)
        all_data = all_data + ls_data

    print(f"all_data: {len(all_data)}")

    shuffle = (split == "train")
    dataloader = DataLoader(
        all_data,
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