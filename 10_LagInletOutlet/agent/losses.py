# losses.py
import torch


def masked_mse(pred, target, out_mask):
    """MSE over valid output particles only."""
    sq_err = (pred - target) ** 2

    if out_mask is None:
        return sq_err.mean()

    mask = out_mask.to(device=pred.device, dtype=pred.dtype).unsqueeze(-1)
    denom = (mask.sum() * pred.shape[-1]).clamp_min(1.0)
    return (sq_err * mask).sum() / denom


def compute_loss(out, y, out_mask):
    loss = masked_mse(out, y, out_mask)
    components = {
        "mse": float(loss.detach().cpu()),
        "rmse": float(torch.sqrt(loss.detach()).cpu()),
    }
    return loss, components