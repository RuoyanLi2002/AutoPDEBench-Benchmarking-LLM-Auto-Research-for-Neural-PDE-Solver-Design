import math
import torch

from utils import vel_unnormalizer


DP = 0.006
COEFH = 0.91924
H = 0.0625
DOMAIN = (2.0, 2.0)
ORIGIN = (-1.0, -1.0)


def _rmse_position(pred_pos, target_pos):
    sq_dist = ((pred_pos - target_pos) ** 2).sum(dim=-1)
    return torch.sqrt(sq_dist.mean())


def _kinetic_energy(vel):
    return (vel ** 2).sum(dim=-1).sum(dim=-1)


def _wendland_kernel_2d(r, h):
    q = r / h
    alpha = 7.0 / (4.0 * math.pi * h ** 2)
    t = (1.0 - 0.5 * q).clamp(min=0.0)
    return alpha * t ** 4 * (2.0 * q + 1.0)


def _make_grid(nx, ny, domain, origin, device, dtype):
    lx, lz = domain
    x0, z0 = origin
    xs = x0 + (torch.arange(nx, device=device, dtype=dtype) + 0.5) * (lx / nx)
    zs = z0 + (torch.arange(ny, device=device, dtype=dtype) + 0.5) * (lz / ny)
    gz, gx = torch.meshgrid(zs, xs, indexing="ij")
    return torch.stack([gx.reshape(-1), gz.reshape(-1)], dim=-1)


def _sph_to_grid(pos, vel, grid, h, shepard=True,
                 grid_chunk=2048, eps=1e-12):
    out = torch.empty(grid.shape[0], vel.shape[-1],
                      device=pos.device, dtype=pos.dtype)

    for s in range(0, grid.shape[0], grid_chunk):
        g = grid[s:s + grid_chunk]
        d = g[:, None, :] - pos[None, :, :]
        r = torch.linalg.norm(d, dim=-1)
        W = _wendland_kernel_2d(r, h)

        num = W @ vel
        if shepard:
            den = W.sum(dim=-1, keepdim=True).clamp(min=eps)
            out[s:s + grid_chunk] = num / den
        else:
            out[s:s + grid_chunk] = num * (DP ** 2)

    return out


def _rmse(pred, target):
    return torch.sqrt(torch.mean((pred - target) ** 2))


def _r2(pred, target, eps=1e-12):
    ybar = target.mean(0, keepdim=True)
    ss_res = torch.sum((pred - target) ** 2)
    ss_tot = torch.sum((target - ybar) ** 2)
    return 1.0 - ss_res / (ss_tot + eps)


def _frmse(pred, target):
    pred_F = torch.fft.fftn(pred, dim=[2, 3])
    target_F = torch.fft.fftn(target, dim=[2, 3])
    return torch.sqrt(torch.mean((pred_F.abs() - target_F.abs()) ** 2))


def _to_np(x):
    return x.detach().cpu().double().numpy()


def sinkhorn_divergence(pred, gt, reg=0.1, numItermax=500, stopThr=1e-5,
                        max_points=4096, seed=0):
    import numpy as np
    import ot

    p = _to_np(pred) if torch.is_tensor(pred) else np.asarray(pred, np.float64)
    g = _to_np(gt) if torch.is_tensor(gt) else np.asarray(gt, np.float64)

    if max_points is not None:
        rng = np.random.default_rng(seed)
        if p.shape[0] > max_points:
            p = p[rng.choice(p.shape[0], max_points, replace=False)]
        if g.shape[0] > max_points:
            g = g[rng.choice(g.shape[0], max_points, replace=False)]

    n, m = p.shape[0], g.shape[0]
    a, b = np.full(n, 1.0 / n), np.full(m, 1.0 / m)

    kw = dict(reg=reg, numItermax=numItermax, stopThr=stopThr)
    c_ab = ot.sinkhorn2(a, b, ot.dist(p, g, metric="sqeuclidean"), **kw)
    c_aa = ot.sinkhorn2(a, a, ot.dist(p, p, metric="sqeuclidean"), **kw)
    c_bb = ot.sinkhorn2(b, b, ot.dist(g, g, metric="sqeuclidean"), **kw)

    scalar = lambda c: float(np.asarray(c).reshape(-1)[0])
    D = scalar(c_ab) - 0.5 * (scalar(c_aa) + scalar(c_bb))
    return max(D, 0.0)


def eval_particle_metrics(pred_vel_norm, target_vel_norm, last_pos,
                          h=H, domain=DOMAIN, origin=ORIGIN,
                          grid_dx=DP, grid_nx=32, grid_ny=32,
                          shepard=True, compute_sinkhorn=True,
                          sinkhorn_reg=0.1, sinkhorn_iters=500,
                          sinkhorn_thr=1e-5, sinkhorn_max_points=4096):
    device = pred_vel_norm.device
    dtype = pred_vel_norm.dtype

    pred_vel = vel_unnormalizer(pred_vel_norm)
    target_vel = vel_unnormalizer(target_vel_norm)

    pred_pos = last_pos + pred_vel
    target_pos = last_pos + target_vel
    rmse_pos = _rmse_position(pred_pos, target_pos)

    ke_pred = _kinetic_energy(pred_vel)
    ke_target = _kinetic_energy(target_vel)
    ke_rmse = torch.sqrt(torch.mean((ke_pred - ke_target) ** 2))

    lx, lz = domain
    nx = grid_nx if grid_nx is not None else int(round(lx / grid_dx))
    ny = grid_ny if grid_ny is not None else int(round(lz / grid_dx))
    grid = _make_grid(nx, ny, domain, origin, device, dtype)

    B = pred_vel.shape[0]
    pred_fields, target_fields = [], []
    for b in range(B):
        up = _sph_to_grid(last_pos[b], pred_vel[b], grid, h, shepard)
        ut = _sph_to_grid(last_pos[b], target_vel[b], grid, h, shepard)
        pred_fields.append(up.T.reshape(2, ny, nx))
        target_fields.append(ut.T.reshape(2, ny, nx))

    pred_field = torch.stack(pred_fields, dim=0)
    target_field = torch.stack(target_fields, dim=0)

    eul_rmse = _rmse(pred_field, target_field)
    eul_frmse = _frmse(pred_field, target_field)
    eul_r2 = _r2(pred_field, target_field)

    metrics = {
        "rmse_pos": rmse_pos.item(),
        "ke_rmse": ke_rmse.item(),
        "eul_vel_rmse": eul_rmse.item(),
        "eul_vel_frmse": eul_frmse.item(),
        "eul_vel_r2": eul_r2.item(),
    }

    if compute_sinkhorn:
        sk = [sinkhorn_divergence(pred_pos[b], target_pos[b],
                                  reg=sinkhorn_reg, numItermax=sinkhorn_iters,
                                  stopThr=sinkhorn_thr,
                                  max_points=sinkhorn_max_points)
              for b in range(B)]
        metrics["sinkhorn_pos"] = float(sum(sk) / len(sk))

    return metrics