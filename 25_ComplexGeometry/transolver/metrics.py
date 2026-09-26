import torch


def _rmse(pred, target):
    return torch.sqrt(torch.mean((pred - target) ** 2))





def eval_metrics(pred, target, surf):
    """Field metrics for one ShapeNetCar sample.

    Args:
        pred:   [N, C] predicted fields (velocity channels + pressure).
        target: [N, C] ground-truth fields.
        surf:   [N] bool mask, True for surface nodes.

    Returns dict with per-sample RMSE over volume (non-surface) nodes and
    surface nodes, all channels included (matching the legacy split).
    """
    rmse_volume = _rmse(pred[~surf], target[~surf])
    rmse_surface = _rmse(pred[surf], target[surf])
    return {
        "rmse_volume": rmse_volume.item(),
        "rmse_surface": rmse_surface.item(),
    }


def coefficient_metrics(pred_coefs, gt_coefs):
    """Drag-coefficient metrics across the whole split.

    Args:
        pred_coefs, gt_coefs: 1D sequences of per-sample drag coefficients.

    Returns dict with RMSE of the coefficient, and R^2 when it is
    computable (needs at least 2 samples with non-zero target variance).
    """
    pred = torch.as_tensor(pred_coefs, dtype=torch.float64)
    gt = torch.as_tensor(gt_coefs, dtype=torch.float64)

    results = {"rmse_coefficient": _rmse(pred, gt).item()}

    
    

    return results