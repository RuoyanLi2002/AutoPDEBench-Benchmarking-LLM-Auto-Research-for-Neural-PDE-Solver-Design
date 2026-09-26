import torch


PDE_NAMES = {
    0: "advection_diffusion_2d",
    1: "burgers_2d",
    2: "gray_scott_2d",
    3: "navier_stokes_vorticity_2d",
}


def _flatten_all(preds, targets):
    pred_parts = [p.reshape(-1) for p in preds]
    target_parts = [t.reshape(-1) for t in targets]
    if not pred_parts:
        empty = torch.zeros(0)
        return empty, empty
    return torch.cat(pred_parts), torch.cat(target_parts)


def _rmse_flat(pred_vec, target_vec):
    return torch.sqrt(torch.mean((pred_vec - target_vec) ** 2))


def _rmse(preds, targets):
    pred_vec, target_vec = _flatten_all(preds, targets)
    return _rmse_flat(pred_vec, target_vec)


def _r2(preds, targets, eps=1e-12):
    pred_vec, target_vec = _flatten_all(preds, targets)
    ybar = target_vec.mean()
    ss_res = torch.sum((pred_vec - target_vec) ** 2)
    ss_tot = torch.sum((target_vec - ybar) ** 2)
    return 1.0 - ss_res / (ss_tot + eps)


def _frmse(preds, targets):
    sq_err_parts = []
    for p, t in zip(preds, targets):
        p_F = torch.fft.fftn(p, dim=[2, 3])
        t_F = torch.fft.fftn(t, dim=[2, 3])
        sq_err_parts.append(((p_F.abs() - t_F.abs()) ** 2).reshape(-1))
    if not sq_err_parts:
        return torch.tensor(float("nan"))
    return torch.sqrt(torch.mean(torch.cat(sq_err_parts)))


def _per_pde_rmse(preds, targets, pde_ids):
    per_pde = {}
    for pde_id, name in PDE_NAMES.items():
        key = f"rmse_{name}"
        pred_parts, target_parts = [], []
        for p, t, ids in zip(preds, targets, pde_ids):
            mask = ids == pde_id
            if mask.any():
                pred_parts.append(p[mask].reshape(-1))
                target_parts.append(t[mask].reshape(-1))
        if pred_parts:
            per_pde[key] = _rmse_flat(
                torch.cat(pred_parts), torch.cat(target_parts)
            ).item()
        else:
            per_pde[key] = float("nan")
    return per_pde


def eval_metrics(preds, targets, pde_ids):
    if torch.is_tensor(preds):
        preds, targets, pde_ids = [preds], [targets], [pde_ids]

    results = {
        "rmse": _rmse(preds, targets).item(),
        "r2": _r2(preds, targets).item(),
        "frmse": _frmse(preds, targets).item(),
    }
    results.update(_per_pde_rmse(preds, targets, pde_ids))
    return results