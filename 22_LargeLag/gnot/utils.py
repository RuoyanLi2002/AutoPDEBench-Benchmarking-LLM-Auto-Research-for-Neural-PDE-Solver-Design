import glob
import os
import os.path as osp
from collections import OrderedDict

import numpy as np
import torch
from torch.utils.data import Dataset
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader

import pyvista as pv


FAR_AWAY = 5.0

FLUID_TYPE = 1.0
SOLID_TYPE = 0.0


_VEL_MEAN = [0.62270163, 0.0212611]
_VEL_STD = [4.13231391, 0.81710397]


def vel_normalizer(vel):
    vel_mean = torch.tensor(_VEL_MEAN, device=vel.device, dtype=vel.dtype)
    vel_std = torch.tensor(_VEL_STD, device=vel.device, dtype=vel.dtype)
    return (vel - vel_mean) / vel_std


def vel_unnormalizer(normalized_vel):
    vel_mean = torch.tensor(_VEL_MEAN, device=normalized_vel.device,
                            dtype=normalized_vel.dtype)
    vel_std = torch.tensor(_VEL_STD, device=normalized_vel.device,
                           dtype=normalized_vel.dtype)
    return normalized_vel * vel_std + vel_mean



def _read_fluid_frame(path):
    mesh = pv.read(path)
    pos = np.asarray(mesh.points)[:, [0, 2]].astype(np.float32)
    vel = np.asarray(mesh["Vel"])[:, [0, 2]].astype(np.float32)
    idp = np.asarray(mesh["Idp"]).astype(np.int64)

    finite = np.isfinite(pos).all(axis=1) & np.isfinite(vel).all(axis=1)
    idp, pos, vel = idp[finite], pos[finite], vel[finite]

    order = np.argsort(idp)
    return idp[order], pos[order], vel[order]


def _read_solid_frame(path):
    mesh = pv.read(path)
    pos = np.asarray(mesh.points)[:, [0, 2]].astype(np.float32)
    idp = np.asarray(mesh["Idp"]).astype(np.int64)

    finite = np.isfinite(pos).all(axis=1)
    idp, pos = idp[finite], pos[finite]

    order = np.argsort(idp)
    return idp[order], pos[order]



def _nearest_fill_velocity(vel, mask):
    N, T = mask.shape
    idx = np.broadcast_to(np.arange(T), (N, T))
    
    fwd = np.maximum.accumulate(np.where(mask, idx, -1), axis=1)
    bwd = np.minimum.accumulate(np.where(mask, idx, T)[:, ::-1], axis=1)[:, ::-1]

    big = np.iinfo(np.int64).max
    dist_f = np.where(fwd >= 0, idx - fwd, big)
    dist_b = np.where(bwd < T, bwd - idx, big)

    take = np.where(dist_b < dist_f, bwd, fwd)
    take = np.clip(take, 0, T - 1)
    return np.take_along_axis(vel, take[:, :, None], axis=1)




class CylinderFlowDataset(Dataset):
    def __init__(self, cfg, frame_ids, split):
        super().__init__()
        self.root = cfg.data.dataset_root
        self.seq_length = cfg.data.seq_length
        self.interval = cfg.data.split_interval
        self.frame_ids = list(frame_ids)


       
        self.cache_dir = getattr(cfg.data, "cache_dir", None)
        if self.cache_dir is not None:
            os.makedirs(self.cache_dir, exist_ok=True)


        
        self.sample_cache_dir = getattr(cfg.data, "sample_cache_dir", None)
        if self.sample_cache_dir is not None:
            os.makedirs(self.sample_cache_dir, exist_ok=True)



        n_windows = len(self.frame_ids) - self.seq_length
        self.window_starts = list(range(0, n_windows, self.interval))


        self._cache = OrderedDict()
        self._cache_size = self.seq_length + 4


    def _fluid_path(self, fid):
        return osp.join(self.root, f"PartFluid_{fid:04d}.vtk")

    def _solid_path(self, fid):
        return osp.join(self.root, f"PartMoving_{fid:04d}.vtk")

    def _cached_load(self, kind, fid, reader, path):
        key = (kind, fid)
        if key in self._cache:
            self._cache.move_to_end(key)
            return self._cache[key]

        frame = None
        if self.cache_dir is not None:
            npz_path = osp.join(self.cache_dir, f"{kind}_{fid:04d}.npz")
            if osp.exists(npz_path):
                with np.load(npz_path) as z:
                    frame = tuple(z[k] for k in z.files)

        if frame is None:
            frame = reader(path)
            if self.cache_dir is not None:
                tmp = osp.join(self.cache_dir,
                               f"{kind}_{fid:04d}.{os.getpid()}.tmp.npz")
                np.savez(tmp, *frame)
                os.replace(tmp, npz_path)

        self._cache[key] = frame
        while len(self._cache) > self._cache_size:
            self._cache.popitem(last=False)
        return frame

    def _get_fluid(self, fid):
        return self._cached_load("fluid", fid, _read_fluid_frame,
                                 self._fluid_path(fid))

    def _get_solid(self, fid):
        return self._cached_load("solid", fid, _read_solid_frame,
                                 self._solid_path(fid))


    def __len__(self):
        return len(self.window_starts)

    def _sample_cache_path(self, start):
        first_fid = self.frame_ids[start]
        return osp.join(self.sample_cache_dir,
                        f"sample_{first_fid:04d}_L{self.seq_length}.npz")

    def __getitem__(self, index):
        start = self.window_starts[index]


        if self.sample_cache_dir is not None:
            p = self._sample_cache_path(start)
            if osp.exists(p):
                with np.load(p) as z:
                    return Data(x=torch.from_numpy(z["x"]),
                                pos=torch.from_numpy(z["pos"]),
                                y=torch.from_numpy(z["y"]),
                                y_mask=torch.from_numpy(z["y_mask"]),
                                num_nodes=int(z["x"].shape[0]))

        data = self._build_sample(start)

        if self.sample_cache_dir is not None:
            tmp = osp.join(self.sample_cache_dir,
                           f".{os.getpid()}_{start}.tmp.npz")
            np.savez(tmp, x=data.x.numpy(), pos=data.pos.numpy(),
                     y=data.y.numpy(), y_mask=data.y_mask.numpy())
            os.replace(tmp, p)
        return data

    def _build_sample(self, start):
        L = self.seq_length
        input_ids = self.frame_ids[start:start + L]
        target_id = self.frame_ids[start + L]

        frames = [self._get_fluid(fid) for fid in input_ids]
        t_idp, _, t_vel = self._get_fluid(target_id)

 
        all_idp = np.unique(np.concatenate([f[0] for f in frames]))
        n_fluid = len(all_idp)
        T = L + 1

        vel_seq = np.zeros((n_fluid, T, 2), dtype=np.float32)
        present = np.zeros((n_fluid, T), dtype=bool)
        last_pos = np.full((n_fluid, 2), FAR_AWAY, dtype=np.float32)

        for t, (idp, pos, vel) in enumerate(frames):
            rows = np.searchsorted(all_idp, idp)
            vel_seq[rows, t] = vel
            present[rows, t] = True
            if t == L - 1:
                last_pos[rows] = pos


        loc = np.searchsorted(all_idp, t_idp)
        valid = (loc < n_fluid)
        valid[valid] &= (all_idp[loc[valid]] == t_idp[valid])
        vel_seq[loc[valid], L] = t_vel[valid]
        present[loc[valid], L] = True

       
        present_target = present[:, L].copy()

 
        vel_seq = _nearest_fill_velocity(vel_seq, present)
        vel_hist = vel_seq[:, :L]
        y_fluid = vel_seq[:, L]


        y_mask_fluid = present_target


        _, s_pos = self._get_solid(input_ids[-1])
        n_solid = len(s_pos)
        s_vel_hist = np.zeros((n_solid, L, 2), dtype=np.float32)
        s_y = np.zeros((n_solid, 2), dtype=np.float32)
        s_y_mask = np.zeros(n_solid, dtype=bool)


        vel_hist = torch.from_numpy(vel_hist)          # (n_f, L, 2)
        s_vel_hist = torch.from_numpy(s_vel_hist)      # (n_s, L, 2)

        pos_all = torch.cat([torch.from_numpy(last_pos),
                             torch.from_numpy(s_pos)], dim=0)     # (n, 2)
        vel_all = vel_normalizer(torch.cat([vel_hist, s_vel_hist], dim=0))
        vel_all = vel_all.reshape(n_fluid + n_solid, -1)          # (n, 2L)
        type_all = torch.cat([
            torch.full((n_fluid, 1), FLUID_TYPE),
            torch.full((n_solid, 1), SOLID_TYPE)], dim=0)         # (n, 1)

        node_feature = torch.cat([pos_all, vel_all, type_all], dim=-1)

        y = vel_normalizer(torch.cat([torch.from_numpy(y_fluid),
                                      torch.from_numpy(s_y)], dim=0))
        y_mask = torch.cat([torch.from_numpy(y_mask_fluid),
                            torch.from_numpy(s_y_mask)], dim=0)

        data = Data(x=node_feature, pos=pos_all, y=y, y_mask=y_mask,
                    num_nodes=n_fluid + n_solid)
        return data





N_TRAIN, N_VALID, N_TEST = 1500, 250, 250


def _split_frame_ids(cfg, split):
    files = sorted(glob.glob(osp.join(cfg.data.dataset_root, "PartFluid_*.vtk")))
    if not files:
        raise FileNotFoundError(
            f"No PartFluid_*.vtk found in {cfg.data.dataset_root}")
    ids = [int(osp.basename(f).split("_")[1].split(".")[0]) for f in files]

    if split == "train":
        return ids[:N_TRAIN]
    if split == "valid":
        return ids[N_TRAIN:N_TRAIN + N_VALID]
    if split == "test":
        return ids[N_TRAIN + N_VALID:N_TRAIN + N_VALID + N_TEST]
    raise NotImplementedError(split)


def _load_split(cfg, split):
    frame_ids = _split_frame_ids(cfg, split)
    dataset = CylinderFlowDataset(cfg, frame_ids, split)
    print(f"[{split}] {len(frame_ids)} frames -> {len(dataset)} windows "
          f"(seq_length={cfg.data.seq_length}, interval={cfg.data.split_interval})")

    shuffle = (split == "train")
    num_workers = getattr(cfg.data, "num_workers", 2)
    dataloader = DataLoader(
        dataset,
        batch_size=cfg.training.batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=True,
        persistent_workers=num_workers > 0,
        prefetch_factor=getattr(cfg.data, "prefetch_factor", 4)
        if num_workers > 0 else None,
    )
    return dataloader


def load_train(cfg):
    return _load_split(cfg, "train")


def load_valid(cfg):
    return _load_split(cfg, "valid")


def load_test(cfg):
    return _load_split(cfg, "test")