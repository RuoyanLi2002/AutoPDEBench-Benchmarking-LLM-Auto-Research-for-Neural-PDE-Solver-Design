# losses.py
import torch


def compute_loss(out, y, x):
    loss = torch.mean((out - y) ** 2)
    components = {
        "mse": float(loss.detach().cpu()),
        "rmse": float(torch.sqrt(loss.detach()).cpu()),
    }
    return loss, components