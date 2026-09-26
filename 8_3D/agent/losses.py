# losses.py
def compute_loss(out, y, x):
    mse = ((out - y) ** 2).mean()
    components = {
        "mse": mse.detach().item(),
    }
    return mse, components