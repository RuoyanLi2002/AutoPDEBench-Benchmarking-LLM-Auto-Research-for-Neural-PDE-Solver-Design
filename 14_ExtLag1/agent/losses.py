# losses.py
def compute_loss(out, y, x, attr=None):
    fluid_mask = x[..., 14] < 0.5

    if fluid_mask.any():
        diff = out - y
        loss = diff[fluid_mask].pow(2).mean()
    else:
        loss = (out - y).sum() * 0.0

    loss_value = float(loss.detach().cpu())
    components = {
        "loss": loss_value,
        "fluid_mse": loss_value,
    }
    return loss, components