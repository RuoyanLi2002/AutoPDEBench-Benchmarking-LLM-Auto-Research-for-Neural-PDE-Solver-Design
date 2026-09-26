import torch


def instantaneous_ke(x, vel_channels=(2, 3)):
    cu, cv = vel_channels
    return 0.5 * (x[:, cu] ** 2 + x[:, cv] ** 2)  # [N, H, W]


def eval_metrics(pred, target, vel_channels=(2, 3), eps=1e-12):
    pred = pred[:, :c]
    target = target[:, :c]

    rmse = torch.sqrt(torch.mean((pred - target) ** 2))

    ybar = target.mean(0, keepdim=True)                 # [1, C, H, W]
    ss_res = torch.sum((pred - target) ** 2)
    ss_tot = torch.sum((target - ybar) ** 2)
    r2 = 1.0 - ss_res / (ss_tot + eps)

    pred_F = torch.fft.fftn(pred, dim=[2, 3])
    target_F = torch.fft.fftn(target, dim=[2, 3])
    frmse = torch.sqrt(torch.mean((pred_F.abs() - target_F.abs()) ** 2))

    ke_pred = instantaneous_ke(pred, vel_channels=vel_channels)
    ke_target = instantaneous_ke(target, vel_channels=vel_channels)
    ke = (ke_pred - ke_target).abs().mean()

    return {
        "rmse": rmse.item(),
        "r2": r2.item(),
        "frmse": frmse.item(),
        "ke": ke.item(),
    }