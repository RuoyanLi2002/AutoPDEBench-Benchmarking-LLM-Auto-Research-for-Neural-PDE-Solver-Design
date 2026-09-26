import torch

DOMAIN_NAMES = ["mesh", "grid", "particle"]


def _rmse(pred, target):
    rmse = torch.sqrt(torch.mean((pred - target) ** 2))
    return rmse




def eval_metrics(preds, targets):
    per_domain_rmse = {}
    sq_err_sum = torch.tensor(0.0)
    n_elems = 0

    for name, pred, target in zip(DOMAIN_NAMES, preds, targets):
        per_domain_rmse[f"rmse_{name}"] = _rmse(pred, target).item()
        sq_err_sum = sq_err_sum + torch.sum((pred - target) ** 2)
        n_elems += target.numel()

    rmse = torch.sqrt(sq_err_sum / n_elems)

    return {
        "rmse": rmse.item(),
        **per_domain_rmse,
    }