import torch


def _rmse(pred, target):
    return torch.sqrt(torch.mean((pred - target) ** 2))


def _r2(pred, target, eps=1e-12):
    ybar = target.mean(0, keepdim=True)
    ss_res = torch.sum((pred - target) ** 2)
    ss_tot = torch.sum((target - ybar) ** 2)
    return 1.0 - ss_res / (ss_tot + eps)


def _frmse(pred, target):
    # Data is (N, num_points, C); spatial points live on dim=1.
    pred_F = torch.fft.fft(pred, dim=1)
    target_F = torch.fft.fft(target, dim=1)
    return torch.sqrt(torch.mean((pred_F.abs() - target_F.abs()) ** 2))


def eval_metrics(pred, target):
    rmse = _rmse(pred, target)
    r2 = _r2(pred, target)
    frmse = _frmse(pred, target)

    return {
        "rmse": rmse.item(),
        "r2": r2.item(),
        "frmse": frmse.item(),
    }