# losses.py
import torch


def compute_loss(out, y, x, pos=None):
    mse = torch.mean((out - y) ** 2)
    components = {
        "mse": float(mse.detach().cpu()),
    }
    return mse, components