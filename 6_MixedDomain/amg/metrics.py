import torch

DOMAIN_NAMES = ["mesh", "grid", "particle"]


def _rmse(pred, target):
    rmse = torch.sqrt(torch.mean((pred - target) ** 2))
    return rmse




def eval_metrics(preds, targets):
    per_domain = {}
    sq_err_sum = torch.tensor(0.0, dtype=torch.float64)
    n_elems = 0

    for name, pred, target in zip(DOMAIN_NAMES, preds, targets):
        if target.numel() == 0:
            per_domain[f"rmse_{name}"] = float("nan")
            per_domain[f"r2_{name}"] = float("nan")
            continue

        if pred.shape != target.shape:
            raise ValueError(
                f"domain '{name}': pred shape {tuple(pred.shape)} != "
                f"target shape {tuple(target.shape)}"
            )


        per_domain[f"rmse_{name}"] = _rmse(pred, target).item()

        sq_err_sum = sq_err_sum + torch.sum(
            (pred.double() - target.double()) ** 2
        )
        n_elems += target.numel()

    rmse = torch.sqrt(sq_err_sum / n_elems) if n_elems else torch.tensor(float("nan"))

    return {
        "rmse": rmse.item(),
        **per_domain,
    }