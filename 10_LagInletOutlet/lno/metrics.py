import math
import torch

from utils import vel_unnormalizer


DP = 0.002
BX = 0.2
BLAYERS = 4
INLETZ = 0.12
INLETZSIZE = 0.005

XMIN = -int(BX / DP) * DP - 6 * DP
XMAX = -XMIN
ZMIN = -BLAYERS * DP
ZMAX = INLETZ + INLETZSIZE + 2 * DP
DOMAIN = ((XMIN, XMAX), (ZMIN, ZMAX))

GRID_N = 32


H = (XMAX - XMIN) / GRID_N / 2.0


def _wendland_kernel_2d(r, h):
    q = r / h
    alpha = 7.0 / (4.0 * math.pi * h ** 2)
    w = ((1.0 - 0.5 * q).clamp(min=0.0) ** 4) * (2.0 * q + 1.0)
    return alpha * w


def _make_grid(n, domain, device, dtype):
    (xmin, xmax), (zmin, zmax) = domain
    xs = xmin + (torch.arange(n, device=device, dtype=dtype) + 0.5) * ((xmax - xmin) / n)
    zs = zmin + (torch.arange(n, device=device, dtype=dtype) + 0.5) * ((zmax - zmin) / n)
    gz, gx = torch.meshgrid(zs, xs, indexing="ij")
    return torch.stack([gx.reshape(-1), gz.reshape(-1)], dim=-1)   # (G, 2), (x, z)


def _sph_to_grid(pos, vel, grid, h, grid_chunk=1024, eps=1e-12):
    out = torch.zeros(grid.shape[0], vel.shape[-1], device=pos.device, dtype=pos.dtype)

    for s in range(0, grid.shape[0], grid_chunk):
        g = grid[s:s + grid_chunk]
        d = g[:, None, :] - pos[None, :, :]          # (g, N, 2) - no periodic wrap
        r = torch.linalg.norm(d, dim=-1)             # (g, N)
        W = _wendland_kernel_2d(r, h)                # (g, N)

        num = W @ vel                                # (g, 2)
        den = W.sum(dim=-1, keepdim=True)            # (g, 1)
        out[s:s + grid_chunk] = torch.where(
            den > eps, num / den.clamp(min=eps), torch.zeros_like(num)
        )

    return out


def _rmse(pred, target):
    return torch.sqrt(torch.mean((pred - target) ** 2))


def _r2(pred_field, target_field, eps=1e-12):
    ybar = target_field.mean(dim=0, keepdim=True)
    ss_res = torch.sum((pred_field - target_field) ** 2)
    ss_tot = torch.sum((target_field - ybar) ** 2)
    return 1.0 - ss_res / (ss_tot + eps)


def _frmse(pred_field, target_field):
    pred_F = torch.fft.fftn(pred_field, dim=[2, 3])
    target_F = torch.fft.fftn(target_field, dim=[2, 3])
    return torch.sqrt(torch.mean((pred_F.abs() - target_F.abs()) ** 2))


def eval_particle_metrics(pred_vel_norm, target_vel_norm, positions,
                          h=H, domain=DOMAIN, grid_n=GRID_N, device=None):
    if device is None:
        device = pred_vel_norm[0].device
    dtype = pred_vel_norm[0].dtype

    grid = _make_grid(grid_n, domain, device, dtype)   # (grid_n^2, 2)

    sq_err_sum = torch.zeros((), device=device, dtype=dtype)
    n_particles = 0
    ke_pred_list, ke_target_list = [], []
    pred_fields, target_fields = [], []

    for pv_n, tv_n, pos in zip(pred_vel_norm, target_vel_norm, positions):
        pv = vel_unnormalizer(pv_n.to(device))
        tv = vel_unnormalizer(tv_n.to(device))
        pos = pos.to(device)

        sq_err_sum += ((pv - tv) ** 2).sum()
        n_particles += pv.numel()

        ke_pred_list.append((pv ** 2).sum())
        ke_target_list.append((tv ** 2).sum())


        up = _sph_to_grid(pos, pv, grid, h)
        ut = _sph_to_grid(pos, tv, grid, h)
        pred_fields.append(up.T.reshape(2, grid_n, grid_n))
        target_fields.append(ut.T.reshape(2, grid_n, grid_n))

    vel_rmse = torch.sqrt(sq_err_sum / n_particles)

    ke_pred = torch.stack(ke_pred_list)
    ke_target = torch.stack(ke_target_list)
    ke_rmse = torch.sqrt(torch.mean((ke_pred - ke_target) ** 2))

    pred_field = torch.stack(pred_fields, dim=0)
    target_field = torch.stack(target_fields, dim=0)

    eul_rmse = _rmse(pred_field, target_field)
    eul_frmse = _frmse(pred_field, target_field)
    eul_r2 = _r2(pred_field, target_field)

    return {
        "vel_rmse": vel_rmse.item(),
        "ke_rmse": ke_rmse.item(),
        "eul_vel_rmse": eul_rmse.item(),
        "eul_vel_frmse": eul_frmse.item(),
        "eul_vel_r2": eul_r2.item(),
    }