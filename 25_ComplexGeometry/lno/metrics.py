import torch


def _rmse(pred, target):
    return torch.sqrt(torch.mean((pred - target) ** 2))




def eval_metrics(pred, target, surf):
    rmse_volume = _rmse(pred[~surf], target[~surf])
    rmse_surface = _rmse(pred[surf], target[surf])
    return {
        "rmse_volume": rmse_volume.item(),
        "rmse_surface": rmse_surface.item(),
    }


def coefficient_metrics(pred_coefs, gt_coefs):
    pred = torch.as_tensor(pred_coefs, dtype=torch.float64)
    gt = torch.as_tensor(gt_coefs, dtype=torch.float64)

    results = {"rmse_coefficient": _rmse(pred, gt).item()}


    return results