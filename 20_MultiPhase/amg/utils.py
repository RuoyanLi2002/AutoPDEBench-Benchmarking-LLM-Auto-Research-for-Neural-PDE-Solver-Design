import os
import glob
import os.path as osp

import h5py
import numpy as np
import torch
from torch_geometric.data import Data, DataLoader

from scipy.spatial import cKDTree


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




def _create_subsequences(position, particle_information, sloshing_motion, cfg):
    seq_length = cfg.data.seq_length
    interval = cfg.data.split_interval

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
            x=node_feature.float(),
            pos=x[-1, :, :].float(),
            y=y.float(),
            particle_information=particle_information.float(),
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

        ls_data = _create_subsequences(position, particle_information, sloshing_motion, cfg)
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