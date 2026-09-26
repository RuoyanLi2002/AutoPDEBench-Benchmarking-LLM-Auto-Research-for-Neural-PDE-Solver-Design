import os
import glob

import numpy as np
import torch
from torch_geometric.data import Data, Dataset
from torch_geometric.loader import DataLoader


def field_to_graph(field, pos, target_field=None):
    field = torch.as_tensor(field, dtype=torch.float32)
    num_channels = field.shape[0]

    x = field.reshape(num_channels, -1).t().contiguous()     # [N, C]
    pos = torch.as_tensor(pos, dtype=torch.float32)
    pos = pos.reshape(pos.shape[0], -1).t().contiguous()     # [N, d]
    data = Data(x=x, pos=pos)

    if target_field is not None:
        target_field = torch.as_tensor(target_field, dtype=torch.float32)
        data.y = target_field.reshape(target_field.shape[0], -1).t().contiguous()  # [N, C]

    return data


class MixedResGraphDataset(Dataset):
    def __init__(self, root, split, split_interval=1):
        super().__init__()
        self.split_interval = split_interval
        split_dir = os.path.join(root, split)
        self.files = sorted(glob.glob(os.path.join(split_dir, "batch_*.npz")))
        if not self.files:
            raise FileNotFoundError(f"No batch_*.npz found in {split_dir}")


        self.index = []
        self.meta = []  # list of (H, W, dx, dy)

        for fi, fpath in enumerate(self.files):
            with np.load(fpath) as d:
                vor = d["vor"]  # (bsize, T+1, H, W)
                bsize, n_times, H, W = vor.shape
                dx = float(d["dx"])
                dy = float(d["dy"])
            self.meta.append((H, W, dx, dy))

            last_input = n_times - 1 - self.split_interval
            for s in range(bsize):
                for t in range(0, last_input + 1, self.split_interval):
                    self.index.append((fi, s, t))

    def len(self):
        return len(self.index)

    def _make_pos(self, fi, H, W):
        _, _, dx, dy = self.meta[fi]
        xs = np.arange(W, dtype=np.float32) * dx
        ys = np.arange(H, dtype=np.float32) * dy
        gx, gy = np.meshgrid(xs, ys, indexing="xy")  # both (H, W)
        return np.stack([gx, gy], axis=0)  # (2, H, W)

    def get(self, i):
        fi, s, t = self.index[i]
        fpath = self.files[fi]
        with np.load(fpath) as d:
            vor = d["vor"]  # (bsize, T+1, H, W)
            x = vor[s, t].astype(np.float32)
            y = vor[s, t + self.split_interval].astype(np.float32)

        H, W = x.shape
        pos = self._make_pos(fi, H, W)         # (2, H, W)

        x = x[None, :, :]                      # (1, H, W)
        y = y[None, :, :]                      # (1, H, W)
        return field_to_graph(x, pos, y)


def _load_split(cfg, split):
    split_interval = getattr(cfg.data, "split_interval", 1)
    dataset = MixedResGraphDataset(
        root=cfg.data.dataset_root,
        split=split,
        split_interval=split_interval,
    )
    shuffle = (split == "train_s")
    dataloader = DataLoader(
        dataset,
        batch_size=cfg.training.batch_size,
        shuffle=shuffle,
        num_workers=0,
        pin_memory=True,
    )
    print(f"[{split}] {len(dataset)} graphs, batch_size {cfg.training.batch_size}")
    if len(dataset) > 0:
        g0 = dataset.get(0)
        print(f"  x:   {tuple(g0.x.shape)}")
        print(f"  pos: {tuple(g0.pos.shape)}")
        print(f"  y:   {tuple(g0.y.shape)}")
    return dataloader


def load_train(cfg):
    return _load_split(cfg, "train_s")


def load_valid(cfg):
    return _load_split(cfg, "valid_s")


def load_test(cfg):
    return _load_split(cfg, "test_s")