# losses.py
import torch.nn.functional as F


def compute_loss(out, y, x, attr):
    loss = F.mse_loss(out, y)
    components = {
        "mse": float(loss.detach().cpu()),
        "rmse": float(loss.detach().sqrt().cpu()),
    }
    return loss, components