# losses.py
import torch
import torch.nn.functional as F


def compute_loss(out, y, x, tx, ty):
    loss = F.mse_loss(out, y)
    components = {
        "mse": float(loss.detach().cpu()),
        "rmse": float(torch.sqrt(loss.detach()).cpu()),
    }
    return loss, components