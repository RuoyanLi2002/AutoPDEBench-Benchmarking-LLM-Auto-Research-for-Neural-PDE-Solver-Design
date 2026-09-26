import torch


def rollout_metrics(pred, target, eps=1e-12):
    assert pred.shape == target.shape, f"{pred.shape} vs {target.shape}"

    err2 = (pred - target) ** 2

    
    rmse = torch.sqrt(err2.mean(dim=(0, 2)))


    
    pred_F = torch.fft.fft(pred, dim=-1)
    target_F = torch.fft.fft(target, dim=-1)
    frmse = torch.sqrt(((pred_F.abs() - target_F.abs()) ** 2).mean(dim=(0, 2)))

    return {"rmse": rmse, "frmse": frmse}