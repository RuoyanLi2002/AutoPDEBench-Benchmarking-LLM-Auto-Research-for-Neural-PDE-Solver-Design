import torch


def _rmse(pred, target):
    rmse = torch.sqrt(torch.mean((pred - target) ** 2))
    return rmse


def _frmse(pred, target):
    pred_F = torch.fft.fftn(pred, dim=[-2, -1])
    target_F = torch.fft.fftn(target, dim=[-2, -1])
    frmse = torch.sqrt(torch.mean((pred_F.abs() - target_F.abs()) ** 2))
    return frmse


def eval_metrics(pred, target):
    rmse = _rmse(pred, target)
    frmse = _frmse(pred, target)

    return {
        "rmse": rmse.item(),
        "frmse": frmse.item(),
    }
