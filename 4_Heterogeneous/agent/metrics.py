import math
import torch

from utils import vel_unnormalizer

DOMAIN = (1.0, 2.0)
DX = 1.0/32
H = DX

def _rmse_position(pred_pos, target_pos):
    sq_dist = ((pred_pos - target_pos) ** 2).sum(dim=-1)   # (B, N)
    return torch.sqrt(sq_dist.mean())

def _kinetic_energy(vel):
    return (vel ** 2).sum(dim=-1).sum(dim=-1)


def _quintic_kernel_2d(r, h):
    q = r / h
    sigma = 7.0 / (478.0 * math.pi * h ** 2)
    w = ((3.0 - q).clamp(min=0.0) ** 5
         - 6.0 * (2.0 - q).clamp(min=0.0) ** 5
         + 15.0 * (1.0 - q).clamp(min=0.0) ** 5)
    return sigma * w


def _make_grid(nx, ny, domain, device, dtype):
    lx, ly = domain
    xs = (torch.arange(nx, device=device, dtype=dtype) + 0.5) * (lx / nx)
    ys = (torch.arange(ny, device=device, dtype=dtype) + 0.5) * (ly / ny)
    gy, gx = torch.meshgrid(ys, xs, indexing="ij")
    return torch.stack([gx.reshape(-1), gy.reshape(-1)], dim=-1)


def _sph_to_grid(pos, vel, grid, h, domain, shepard=True,
                 grid_chunk=2048, eps=1e-12):
    box = torch.tensor(domain, device=pos.device, dtype=pos.dtype)
    out = torch.empty(grid.shape[0], vel.shape[-1],
                      device=pos.device, dtype=pos.dtype)

    for s in range(0, grid.shape[0], grid_chunk):
        g = grid[s:s + grid_chunk]
        d = g[:, None, :] - pos[None, :, :]
        d = d - box * torch.round(d / box)
        r = torch.linalg.norm(d, dim=-1)
        W = _quintic_kernel_2d(r, h)

        num = W @ vel
        if shepard:
            den = W.sum(dim=-1, keepdim=True).clamp(min=eps)
            out[s:s + grid_chunk] = num / den
        else:
            out[s:s + grid_chunk] = num * (DX ** 2)

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


def eval_particle_metrics(pred_vel_norm, target_vel_norm, last_pos,
                          h=H, domain=DOMAIN, grid_nx=None, grid_ny=None,
                          shepard=True):
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

    lx, ly = domain
    nx = grid_nx if grid_nx is not None else int(round(lx / DX))
    ny = grid_ny if grid_ny is not None else int(round(ly / DX))
    grid = _make_grid(nx, ny, domain, device, dtype)

    B = pred_vel.shape[0]
    pred_fields, target_fields = [], []
    for b in range(B):
        up = _sph_to_grid(last_pos[b], pred_vel[b], grid, h, domain, shepard)
        ut = _sph_to_grid(last_pos[b], target_vel[b], grid, h, domain, shepard)
        pred_fields.append(up.T.reshape(2, ny, nx))
        target_fields.append(ut.T.reshape(2, ny, nx))

    pred_field = torch.stack(pred_fields, dim=0)
    target_field = torch.stack(target_fields, dim=0)

    eul_rmse = _rmse(pred_field, target_field)
    eul_frmse = _frmse(pred_field, target_field)
    eul_r2 = _r2(pred_field, target_field)

    return {
        "rmse_pos": rmse_pos.item(),
        "ke_rmse": ke_rmse.item(),
        "eul_vel_rmse": eul_rmse.item(),
        "eul_vel_frmse": eul_frmse.item(),
        "eul_vel_r2": eul_r2.item(),
    }