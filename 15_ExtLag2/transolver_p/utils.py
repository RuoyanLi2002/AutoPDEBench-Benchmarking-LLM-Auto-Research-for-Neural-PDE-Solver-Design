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
    [ 7.84766056e-06, -2.61507342e-06],
    [-1.34246962e-06,  0.00000000e+00],
], dtype=torch.float32)

VEL_STD = torch.tensor([
    [1.93005700e-02, 9.02444000e-03],
    [9.19804000e-03, 1.00000000e+00],
], dtype=torch.float32)

_STD_EPS = 1e-8
VEL_STD = torch.clamp(VEL_STD, min=_STD_EPS)


def vel_normalizer(vel, node_type):
    mean = VEL_MEAN.to(device=vel.device, dtype=vel.dtype)[node_type]  # (N, 2)
    std  = VEL_STD.to(device=vel.device, dtype=vel.dtype)[node_type]   # (N, 2)
    return (vel - mean) / std


def vel_unnormalizer(normalized_vel, node_type):
    mean = VEL_MEAN.to(device=normalized_vel.device, dtype=normalized_vel.dtype)[node_type]
    std  = VEL_STD.to(device=normalized_vel.device, dtype=normalized_vel.dtype)[node_type]
    return (normalized_vel * std) + mean


def _time_diff(input_sequence):
    return input_sequence[:, 1:] - input_sequence[:, :-1]


def _get_random_walk_noise(position_sequence, noise_std_last_step):
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


def _create_subsequences(position, particle_type, cfg):
    seq_length = cfg.data.seq_length
    interval = cfg.data.split_interval

    num_frames, num_particles, _ = position.shape
    ls_data = []

    is_piston = (particle_type == 1)
    piston_col = is_piston.to(torch.float32).unsqueeze(-1)   # (n_p, 1) flag feature

    start_idx = 0
    while (start_idx + seq_length) < num_frames:
        subseq = position[start_idx:start_idx + seq_length]  # (seq_length, n_p, 2)

        x = subseq[:-1]                  # (seq_length-1, n_p, 2)
        y = subseq[-1] - subseq[-2]      # (n_p, 2)  raw t+1 velocity
        y_norm = vel_normalizer(y, particle_type)            # (n_p, 2)

        sampled_noise = _get_random_walk_noise(
            x.permute(1, 0, 2), noise_std_last_step=noise_std)
        sampled_noise = sampled_noise.permute(1, 0, 2)
        noised_x = x + sampled_noise

        velocity = noised_x[1:, :, :] - noised_x[:-1, :, :]  # (seq_length-2, n_p, 2)
        velocity = vel_normalizer(velocity, particle_type)
        velocity = velocity.permute(1, 0, 2).contiguous()
        velocity = velocity.view(num_particles, -1)          # (n_p, 2*(seq_length-2))


        next_vel = torch.zeros_like(y_norm)                  # (n_p, 2)
        next_vel[is_piston] = y_norm[is_piston]

        node_feature = torch.cat(
            [x[-1, :, :], velocity, next_vel, piston_col], dim=-1)

        data = Data(
            x=node_feature.unsqueeze(0),
            pos=x[-1, :, :].unsqueeze(0),
            y=y_norm.unsqueeze(0),
            node_type=particle_type.unsqueeze(0),   # (1, n_p) for fluid-only loss
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
        print(list(f.keys()))

        position      = torch.from_numpy(f["particle_pos"][:])                  # (T, N, 2)
        particle_type = torch.from_numpy(
            np.asarray(f["particle_type"][:]).reshape(-1).astype(np.int64))     # (N,)
        print(f"particle_pos: {tuple(position.shape)}  "
              f"particle_type: {tuple(particle_type.shape)}  "
              f"(fluid={int((particle_type==0).sum())}, "
              f"piston={int((particle_type==1).sum())})")

        ls_data = _create_subsequences(position, particle_type, cfg)
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