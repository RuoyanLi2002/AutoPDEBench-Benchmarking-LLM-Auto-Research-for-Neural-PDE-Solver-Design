# losses.py
def compute_loss(out, y, x, param):
    loss = ((out - y) ** 2).mean()
    components = {
        "mse": float(loss.detach().cpu()),
    }
    return loss, components