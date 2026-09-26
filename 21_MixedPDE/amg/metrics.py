import torch


PDE_NAMES = {
    0: "advection_diffusion_2d",
    1: "burgers_2d",
    2: "gray_scott_2d",
    3: "navier_stokes_vorticity_2d",
}


PDE_CHANNELS = {
    0: 1,
    1: 2,
    2: 1,
    3: 2,
}


def _flatten_real(pred, target, pde_ids):
    pred_parts, target_parts = [], []
    for pde_id in PDE_CHANNELS:
        mask = pde_ids == pde_id
        if not mask.any():
            continue
        nc = PDE_CHANNELS[pde_id]
        pred_parts.append(pred[mask, :nc].reshape(-1))
        target_parts.append(target[mask, :nc].reshape(-1))
    if not pred_parts:
        empty = pred.new_zeros(0)
        return empty, empty
    return torch.cat(pred_parts), torch.cat(target_parts)


def _rmse_flat(pred_vec, target_vec):
    return torch.sqrt(torch.mean((pred_vec - target_vec) ** 2))


def _rmse(pred, target, pde_ids):
    pred_vec, target_vec = _flatten_real(pred, target, pde_ids)
    return _rmse_flat(pred_vec, target_vec)


def _r2(pred, target, pde_ids, eps=1e-12):
    pred_vec, target_vec = _flatten_real(pred, target, pde_ids)
    ybar = target_vec.mean()
    ss_res = torch.sum((pred_vec - target_vec) ** 2)
    ss_tot = torch.sum((target_vec - ybar) ** 2)
    r2 = 1.0 - ss_res / (ss_tot + eps)
    return r2


def _frmse(pred, target, pde_ids):
    sq_err_parts = []
    for pde_id in PDE_CHANNELS:
        mask = pde_ids == pde_id
        if not mask.any():
            continue
        nc = PDE_CHANNELS[pde_id]
        p = pred[mask, :nc]
        t = target[mask, :nc]
        p_F = torch.fft.fftn(p, dim=[2, 3])
        t_F = torch.fft.fftn(t, dim=[2, 3])
        sq_err_parts.append(((p_F.abs() - t_F.abs()) ** 2).reshape(-1))
    if not sq_err_parts:
        return pred.new_tensor(float("nan"))
    return torch.sqrt(torch.mean(torch.cat(sq_err_parts)))


def _per_pde_rmse(pred, target, pde_ids):
    per_pde = {}
    for pde_id, name in PDE_NAMES.items():
        mask = pde_ids == pde_id
        key = f"rmse_{name}"
        if mask.any():
            nc = PDE_CHANNELS[pde_id]
            per_pde[key] = _rmse_flat(
                pred[mask, :nc].reshape(-1),
                target[mask, :nc].reshape(-1),
            ).item()
        else:
            per_pde[key] = float("nan")
    return per_pde


def eval_metrics(pred, target, pde_ids):
    rmse = _rmse(pred, target, pde_ids)
    r2 = _r2(pred, target, pde_ids)
    frmse = _frmse(pred, target, pde_ids)

    results = {
        "rmse": rmse.item(),
        "r2": r2.item(),
        "frmse": frmse.item(),
    }
    results.update(_per_pde_rmse(pred, target, pde_ids))
    return results