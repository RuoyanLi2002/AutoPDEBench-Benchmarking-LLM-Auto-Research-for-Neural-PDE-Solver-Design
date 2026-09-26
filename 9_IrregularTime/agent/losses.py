# losses.py
import torch


def compute_loss(out, y, x, dt):
    loss = torch.mean((out - y) ** 2)
    components = {
        "mse": float(loss.detach().cpu()),
    }
    return loss, components