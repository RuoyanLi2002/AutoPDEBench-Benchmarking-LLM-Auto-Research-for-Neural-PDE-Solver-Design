# losses.py
import torch.nn.functional as F


def compute_loss(out, y, x, pde):
    loss = F.mse_loss(out, y)
    components = {
        "mse": loss.item(),
    }
    return loss, components