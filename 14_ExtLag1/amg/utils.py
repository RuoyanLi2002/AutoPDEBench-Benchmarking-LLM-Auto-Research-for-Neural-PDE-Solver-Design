import os
import glob
import os.path as osp

import h5py
import numpy as np
import torch
from torch_geometric.data import Data, DataLoader


noise_std = 6.7e-4

VEL_MEAN = torch.tensor([3.5897982e-05, 1.7713952e-05], dtype=torch.float32)
VEL_STD  = torch.tensor([7.0120855e-02, 4.1210721e-02], dtype=torch.float32)


def vel_normalizer(vel):
    """
    Input:  (..., 2)
    Output: (..., 2)
    """
    vel_mean = VEL_MEAN.to(device=vel.device, dtype=vel.dtype)
    vel_std  = VEL_STD.to(device=vel.device, dtype=vel.dtype)
    return (vel - vel_mean) / vel_std


def vel_unnormalizer(normalized_vel):
    """
    Input:  (..., 2)
    Output: (..., 2)
    """
    vel_mean = VEL_MEAN.to(device=normalized_vel.device, dtype=normalized_vel.dtype)
    vel_std  = VEL_STD.to(device=normalized_vel.device, dtype=normalized_vel.dtype)
    return (normalized_vel * vel_std) + vel_mean


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


def _create_subsequences(position, sloshing_motion, cfg):
    """
    position:        (num_frames, num_particles, 2)
    sloshing_motion: (num_frames, 6)   -- global per-frame signal
    """
    seq_length = cfg.data.seq_length
    interval = cfg.data.split_interval

    num_frames, num_particles, _ = position.shape
    ls_data = []

    start_idx = 0
    while (start_idx + seq_length) < num_frames:
        subseq = position[start_idx:start_idx + seq_length]      # (seq_length, n_p, 2)
        motion_win = sloshing_motion[start_idx:start_idx + seq_length]  # (seq_length, 6)

        x = subseq[:-1]                  # (seq_length-1, n_p, 2)
        y = subseq[-1] - subseq[-2]
        y = vel_normalizer(y)            # (n_p, 2)

        sampled_noise = _get_random_walk_noise(
            x.permute(1, 0, 2), noise_std_last_step=noise_std)
        sampled_noise = sampled_noise.permute(1, 0, 2)
        noised_x = x + sampled_noise

        velocity = noised_x[1:, :, :] - noised_x[:-1, :, :]
        velocity = vel_normalizer(velocity)
        velocity = velocity.permute(1, 0, 2).contiguous()
        velocity = velocity.view(num_particles, -1)   # (n_p, 2*(seq_length-2))


       
        motion_hist = motion_win.reshape(-1)                     # (6*seq_length,)
        motion_hist = motion_hist.to(dtype=x.dtype)
        motion_feat = motion_hist.unsqueeze(0).expand(num_particles, -1)  # (n_p, 6*seq_length)

        node_feature = torch.cat([x[-1, :, :], velocity, motion_feat], dim=-1)

        data = Data(x=node_feature.float(), pos=x[-1, :, :].float(), y=y.float())
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
        print(list(f.keys()))

        position        = torch.from_numpy(f["particle_pos"][:])       # (T, N, 2)
        sloshing_motion = torch.from_numpy(f["sloshing_motion"][:])    # (T, 6)
        print(f"particle_pos: {tuple(position.shape)}  "
              f"sloshing_motion: {tuple(sloshing_motion.shape)}")

        ls_data = _create_subsequences(position, sloshing_motion, cfg)
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