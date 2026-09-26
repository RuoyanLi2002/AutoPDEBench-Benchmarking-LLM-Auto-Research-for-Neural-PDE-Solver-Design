import torch

from utils import BC_SUBDIRS, BC_TO_ID


def _rmse(pred, target):
    rmse = torch.sqrt(torch.mean((pred - target) ** 2))
    return rmse


def _r2(pred, target, eps=1e-12):
    ybar = target.mean(0, keepdim=True)
    ss_res = torch.sum((pred - target) ** 2)
    ss_tot = torch.sum((target - ybar) ** 2)
    r2 = 1.0 - ss_res / (ss_tot + eps)
    return r2


def _frmse(pred, target):
    pred_F = torch.fft.fftn(pred, dim=[2, 3])
    target_F = torch.fft.fftn(target, dim=[2, 3])
    frmse = torch.sqrt(torch.mean((pred_F.abs() - target_F.abs()) ** 2))
    return frmse


def _boundary_mask(H, W, device):
    mask = torch.zeros(H, W, dtype=torch.bool, device=device)
    mask[0, :] = True
    mask[-1, :] = True
    mask[:, 0] = True
    mask[:, -1] = True
    return mask


def _boundary_rmse(pred, target):
    _, _, H, W = pred.shape
    mask = _boundary_mask(H, W, pred.device)
    diff2 = (pred - target) ** 2
    boundary_diff2 = diff2[..., mask]
    return torch.sqrt(boundary_diff2.mean())


def _mass_rmse(pred, target):
    pred_mass = pred.sum(dim=[2, 3])      # [N, C]
    target_mass = target.sum(dim=[2, 3])  # [N, C]
    return torch.sqrt(torch.mean((pred_mass - target_mass) ** 2))


def eval_metrics(pred, target, bc=None):
    rmse = _rmse(pred, target)
    r2 = _r2(pred, target)
    frmse = _frmse(pred, target)

    results = {
        "rmse": rmse.item(),
        "r2": r2.item(),
        "frmse": frmse.item(),
        "boundary_rmse": _boundary_rmse(pred, target).item(),
        "mass_rmse": _mass_rmse(pred, target).item(),
    }


    if bc is not None:
        bc = bc.to(pred.device)
        for name in BC_SUBDIRS:
            bc_id = BC_TO_ID[name]
            sel = bc == bc_id
            key = f"rmse_{name}"
            if sel.any():
                results[key] = _rmse(pred[sel], target[sel]).item()
            else:
                results[key] = float("nan")

    return results