# losses.py
import torch.nn.functional as F


def compute_loss(out, y, x, mask):
    loss = F.mse_loss(out, y)
    components = {
        "mse": float(loss.detach().cpu()),
    }
    return loss, components