import math

import numpy as np
import ot
import torch

from utils import vel_unnormalizer



DOMAIN = (0.9, 0.508)
DP = 0.9 / 32
DX = DP
H = DP

LIQUID = 0
GAS = 1




def _rmse_position(pred_pos, target_pos):
    sq_dist = ((pred_pos - target_pos) ** 2).sum(dim=-1)
    return torch.sqrt(sq_dist.mean())


def _kinetic_energy(vel):
    return (vel ** 2).sum(dim=-1).sum(dim=-1)


def sinkhorn_distance(pred, gt, reg=0.1, numItermax=500, stopThr=1e-5):
    n = pred.shape[0]
    a = b = np.ones(n) / n
    M_ab = ot.dist(pred, gt, metric='sqeuclidean')
    M_aa = ot.dist(pred, pred, metric='sqeuclidean')
    M_bb = ot.dist(gt,   gt,   metric='sqeuclidean')

    C_ab = ot.sinkhorn2(a, b, M_ab, reg=reg, numItermax=numItermax, stopThr=stopThr)
    C_aa = ot.sinkhorn2(a, a, M_aa, reg=reg, numItermax=numItermax, stopThr=stopThr)
    C_bb = ot.sinkhorn2(b, b, M_bb, reg=reg, numItermax=numItermax, stopThr=stopThr)

    D = C_ab - 0.5 * (C_aa + C_bb)
    return max(float(D), 0.0)




def _quintic_kernel_2d(r, h):
    q = r / h
    sigma = 7.0 / (478.0 * math.pi * h ** 2)
    w = ((3.0 - q).clamp(min=0.0) ** 5
         - 6.0 * (2.0 - q).clamp(min=0.0) ** 5
         + 15.0 * (1.0 - q).clamp(min=0.0) ** 5)
    return sigma * w


def _make_grid(nx, ny, domain, origin, device, dtype):
    lx, ly = domain
    x0, y0 = origin
    xs = x0 + (torch.arange(nx, device=device, dtype=dtype) + 0.5) * (lx / nx)
    ys = y0 + (torch.arange(ny, device=device, dtype=dtype) + 0.5) * (ly / ny)
    gy, gx = torch.meshgrid(ys, xs, indexing="ij")
    return torch.stack([gx.reshape(-1), gy.reshape(-1)], dim=-1)


def _sph_to_grid(pos, vel, grid, h, shepard=True, grid_chunk=2048, eps=1e-12):
    out = torch.empty(grid.shape[0], vel.shape[-1],
                      device=pos.device, dtype=pos.dtype)

    if pos.shape[0] == 0:
        out.zero_()
        return out

    for s in range(0, grid.shape[0], grid_chunk):
        g = grid[s:s + grid_chunk]
        d = g[:, None, :] - pos[None, :, :]
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



def eval_particle_metrics(pred_vel_norm, target_vel_norm, last_pos, particle_type,
                          h=H, domain=DOMAIN, origin=(-0.45, 0.0),
                          grid_nx=None, grid_ny=None, shepard=True,
                          sinkhorn_reg=0.1):
    device = pred_vel_norm.device
    dtype = pred_vel_norm.dtype

    pred_vel = vel_unnormalizer(pred_vel_norm, particle_type)
    target_vel = vel_unnormalizer(target_vel_norm, particle_type)

    pred_pos = last_pos + pred_vel
    target_pos = last_pos + target_vel


    
    pt = particle_type
    if pt.dim() == 2:
        pt = pt[0]
    pt = pt.to(device).long()
    liquid_mask = pt == LIQUID
    gas_mask = pt == GAS

    
    def _pos_rmse(mask):
        if mask.sum() == 0:
            return torch.tensor(float("nan"), device=device, dtype=dtype)
        p = pred_pos[:, mask, :].reshape(-1, 2)
        t = target_pos[:, mask, :].reshape(-1, 2)
        return _rmse_position(p, t)

    liquid_rmse_pos = _pos_rmse(liquid_mask)
    gas_rmse_pos = _pos_rmse(gas_mask)


    def _ke_rmse(mask):
        if mask.sum() == 0:
            return torch.tensor(float("nan"), device=device, dtype=dtype)
        ke_p = _kinetic_energy(pred_vel[:, mask, :])
        ke_t = _kinetic_energy(target_vel[:, mask, :])
        return torch.sqrt(torch.mean((ke_p - ke_t) ** 2))

    liquid_ke_rmse = _ke_rmse(liquid_mask)
    gas_ke_rmse = _ke_rmse(gas_mask)



    def _sinkhorn(mask):
        if mask.sum() == 0:
            return float("nan")
        vals = []
        for b in range(pred_pos.shape[0]):
            print(b)
            p = pred_pos[b, mask, :].detach().cpu().numpy().astype(np.float64)
            t = target_pos[b, mask, :].detach().cpu().numpy().astype(np.float64)
            vals.append(sinkhorn_distance(p, t, reg=sinkhorn_reg))
            print(vals[-1])
        return float(np.mean(vals))

    liquid_sinkhorn = _sinkhorn(liquid_mask)
    gas_sinkhorn = _sinkhorn(gas_mask)



    lx, ly = domain
    nx = grid_nx if grid_nx is not None else int(round(lx / DX))
    ny = grid_ny if grid_ny is not None else int(round(ly / DX))
    grid = _make_grid(nx, ny, domain, origin, device, dtype)

    B = pred_vel.shape[0]
    
    pred_fields, target_fields = [], []
    for b in range(B):
        print(b)
        chans_p, chans_t = [], []
        for mask in (liquid_mask, gas_mask):
            pos_b = last_pos[b, mask, :]
            up = _sph_to_grid(pos_b, pred_vel[b, mask, :], grid, h, shepard)
            ut = _sph_to_grid(pos_b, target_vel[b, mask, :], grid, h, shepard)
            chans_p.append(up.T.reshape(2, ny, nx))
            chans_t.append(ut.T.reshape(2, ny, nx))

        
        pred_fields.append(torch.cat(chans_p, dim=0))
        target_fields.append(torch.cat(chans_t, dim=0))

    pred_field = torch.stack(pred_fields, dim=0)
    target_field = torch.stack(target_fields, dim=0)

    eul_rmse = _rmse(pred_field, target_field)
    eul_frmse = _frmse(pred_field, target_field)
    eul_r2 = _r2(pred_field, target_field)

    return {
        "liquid_rmse_pos": liquid_rmse_pos.item(),
        "liquid_ke_rmse": liquid_ke_rmse.item(),
        "liquid_sinkhorn": liquid_sinkhorn,
        "gas_rmse_pos": gas_rmse_pos.item(),
        "gas_ke_rmse": gas_ke_rmse.item(),
        "gas_sinkhorn": gas_sinkhorn,
        "eul_vel_rmse": eul_rmse.item(),
        "eul_vel_frmse": eul_frmse.item(),
        "eul_vel_r2": eul_r2.item(),
    }