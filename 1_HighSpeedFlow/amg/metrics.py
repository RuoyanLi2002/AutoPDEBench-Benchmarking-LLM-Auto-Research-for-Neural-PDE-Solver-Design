import torch

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

def _ke(pred, target, vel_channels=(2, 3)):
    cu, cv = vel_channels
    pred_k = 0.5 * (pred[:, cu] ** 2 + pred[:, cv] ** 2)
    target_k = 0.5 * (target[:, cu] ** 2 + target[:, cv] ** 2)
    ke = (pred_k - target_k).abs().mean()
    return ke

def eval_metrics(pred, target):
    rmse = _rmse(pred, target)
    r2 = _r2(pred, target)
    frmse = _frmse(pred, target)
    ke = _ke(pred, target)

    return {
        "rmse": rmse.item(),
        "r2": r2.item(),
        "frmse": frmse.item(),
        "ke": ke.item(),
    }