import torch

# Channel layout (C=7): density (0), magnetic_field (1,2,3), velocity (4,5,6)
DENSITY_CHANNEL = 0
MAGNETIC_CHANNELS = (1, 2, 3)
VELOCITY_CHANNELS = (4, 5, 6)


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
    # 3D spatial FFT over (nx, ny, nz)
    pred_F = torch.fft.fftn(pred, dim=[2, 3, 4])
    target_F = torch.fft.fftn(target, dim=[2, 3, 4])
    frmse = torch.sqrt(torch.mean((pred_F.abs() - target_F.abs()) ** 2))
    return frmse


def _ke(pred, target, vel_channels=VELOCITY_CHANNELS):
    cu, cv, cw = vel_channels
    pred_k = 0.5 * (pred[:, cu] ** 2 + pred[:, cv] ** 2 + pred[:, cw] ** 2)
    target_k = 0.5 * (target[:, cu] ** 2 + target[:, cv] ** 2 + target[:, cw] ** 2)
    ke = (pred_k - target_k).abs().mean()
    return ke


def _me(pred, target, mag_channels=MAGNETIC_CHANNELS):
    bx, by, bz = mag_channels
    pred_m = 0.5 * (pred[:, bx] ** 2 + pred[:, by] ** 2 + pred[:, bz] ** 2)
    target_m = 0.5 * (target[:, bx] ** 2 + target[:, by] ** 2 + target[:, bz] ** 2)
    me = (pred_m - target_m).abs().mean()
    return me


def eval_metrics(pred, target):
    rmse = _rmse(pred, target)
    r2 = _r2(pred, target)
    frmse = _frmse(pred, target)
    ke = _ke(pred, target)
    me = _me(pred, target)

    return {
        "rmse": rmse.item(),
        "r2": r2.item(),
        "frmse": frmse.item(),
        "ke": ke.item(),
        "me": me.item(),
    }