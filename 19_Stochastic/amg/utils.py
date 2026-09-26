import os

import h5py
import torch
from torch.utils.data import Dataset
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader



_GRID_CACHE = {}


def make_unit_grid(spatial_shape):
    if spatial_shape not in _GRID_CACHE:
        axes = [torch.linspace(0.0, 1.0, s) for s in spatial_shape]
        mesh = torch.meshgrid(*axes, indexing="ij")          # tuple of [*spatial_shape]
        pos = torch.stack([m.reshape(-1) for m in mesh], dim=1)  # [N, d]
        _GRID_CACHE[spatial_shape] = pos.contiguous()
    return _GRID_CACHE[spatial_shape]


def field_to_graph(field, target_field=None):
    field = torch.as_tensor(field, dtype=torch.float32)
    num_channels = field.shape[0]
    spatial_shape = tuple(field.shape[1:])

    x = field.reshape(num_channels, -1).t().contiguous()     # [N, C]
    pos = make_unit_grid(spatial_shape)                      # [N, d]
    data = Data(x=x, pos=pos)

    if target_field is not None:
        target_field = torch.as_tensor(target_field, dtype=torch.float32)
        data.y = target_field.reshape(target_field.shape[0], -1).t().contiguous()  # [N, C_out]

    return data


class SPDEGraphDataset(Dataset):
    def __init__(self, sol, W, horizon=20, noise_mode="full"):
        assert sol.shape == W.shape, f"sol {sol.shape} vs W {W.shape}"
        assert noise_mode in ("full", "single")
        self.sol = sol
        self.W = W
        self.horizon = horizon
        self.noise_mode = noise_mode

        N, T = sol.shape[0], sol.shape[1]
        self.pairs_per_traj = T - horizon
        assert self.pairs_per_traj > 0, "horizon >= trajectory length"
        self.length = N * self.pairs_per_traj

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        n = idx // self.pairs_per_traj
        t = idx % self.pairs_per_traj
        k = self.horizon

        u_t = self.sol[n, t].unsqueeze(0)                 # (1, H, W)
        if self.noise_mode == "full":
            noise = self.W[n, t:t + k]                    # (k, H, W)
        else:
            noise = self.W[n, t].unsqueeze(0)             # (1, H, W)

        x = torch.cat([u_t, noise], dim=0)                # (1+k, H, W) or (2, H, W)
        y = self.sol[n, t + k].unsqueeze(0)               # (1, H, W)
        return field_to_graph(x, y)


def load_h5_split(cfg, split):
    file_path = os.path.join(cfg.data.dataset_root, f"{split}.h5")
    with h5py.File(file_path, "r") as f:
        sol = torch.from_numpy(f["sol"][:]).float()       # (N, T, H, W)
        W = torch.from_numpy(f["W"][:]).float()

    if getattr(cfg.data, "drop_periodic_endpoint", True):
        sol = sol[..., :-1, :-1]
        W = W[..., :-1, :-1]

    horizon = 20
    noise_mode = "full"
    dataset = SPDEGraphDataset(sol, W, horizon=horizon, noise_mode=noise_mode)

    sample = dataset[0]
    print(f"[{split}] sol: {tuple(sol.shape)}, W: {tuple(W.shape)}, "
          f"horizon={horizon}, noise_mode={noise_mode}, "
          f"num_graphs={len(dataset)}")
    print(f"  x:   {tuple(sample.x.shape)}")
    print(f"  pos: {tuple(sample.pos.shape)}")
    print(f"  y:   {tuple(sample.y.shape)}")

    shuffle = (split == "train")
    dataloader = DataLoader(
        dataset,
        batch_size=cfg.training.batch_size,
        shuffle=shuffle,
        pin_memory=True,
        num_workers=getattr(cfg.training, "num_workers", 0),
    )
    return dataloader


def _load_split(cfg, split):
    return load_h5_split(cfg, split)


def load_train(cfg):
    return _load_split(cfg, "train")


def load_valid(cfg):
    return _load_split(cfg, "valid")


def load_test(cfg):
    return _load_split(cfg, "test")