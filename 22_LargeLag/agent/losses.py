# losses.py
def compute_loss(out, y, x, loss_mask):
    """Masked MSE over valid particles and both predicted components.

    out, y:    (bs, N, 2)
    loss_mask: (bs, N) or (bs, N, 1)
    """
    if loss_mask.dim() == 3:
        loss_mask = loss_mask.squeeze(-1)

    valid = loss_mask.bool()
    squared_error = (out - y).pow(2)
    mask = valid.unsqueeze(-1).to(dtype=squared_error.dtype)

    denominator = (mask.sum() * squared_error.shape[-1]).clamp_min(1.0)
    loss = (squared_error * mask).sum() / denominator

    components = {
        "mse": float(loss.detach().cpu()),
        "rmse": float(loss.detach().sqrt().cpu()),
        "supervised_particles": int(valid.sum().item()),
    }
    return loss, components