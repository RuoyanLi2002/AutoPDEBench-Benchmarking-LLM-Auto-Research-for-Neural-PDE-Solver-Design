import math

import numpy as np
import torch
import ot

from utils import vel_unnormalizer



DOMAIN = (0.1, 0.25)
DX = 0.1 / 32
H = DX

SOLID_TYPE = 0
FLUID_TYPE = 1


# --------------------------------------------------------------------------- #
# Sinkhorn divergence (point clouds), per-sample
# --------------------------------------------------------------------------- #
def sinkhorn_distance(pred, gt, reg=0.1, numItermax=500, stopThr=1e-5):
    n = pred.shape[0]
    a = b = np.ones(n) / n
    M_ab = ot.dist(pred, gt, metric='sqeuclidean')
    M_aa = ot.dist(pred, pred, metric='sqeuclidean')
    M_bb = ot.dist(gt, gt, metric='sqeuclidean')

    C_ab = ot.sinkhorn2(a, b, M_ab, reg=reg, numItermax=numItermax, stopThr=stopThr)
    C_aa = ot.sinkhorn2(a, a, M_aa, reg=reg, numItermax=numItermax, stopThr=stopThr)
    C_bb = ot.sinkhorn2(b, b, M_bb, reg=reg, numItermax=numItermax, stopThr=stopThr)

    D = C_ab - 0.5 * (C_aa + C_bb)
    return max(D, 0)


# --------------------------------------------------------------------------- #
# SPH particle -> Eulerian grid
# --------------------------------------------------------------------------- #
def _quintic_kernel_2d(r, h):
    """
    W(q) = sigma * [ (3-q)_+^5 - 6(2-q)_+^5 + 15(1-q)_+^5 ],  q = r/h,
    sigma_2D = 7 / (478 * pi * h^2), compact support r < 3h.
    """
    q = r / h
    sigma = 7.0 / (478.0 * math.pi * h ** 2)
    w = ((3.0 - q).clamp(min=0.0) ** 5
         - 6.0 * (2.0 - q).clamp(min=0.0) ** 5
         + 15.0 * (1.0 - q).clamp(min=0.0) ** 5)
    return sigma * w


def _make_grid(nx, ny, domain, device, dtype):
    """Cell-centered Eulerian nodes on an (Lx, Ly) box -> (G, 2)."""
    lx, ly = domain
    xs = (torch.arange(nx, device=device, dtype=dtype) + 0.5) * (lx / nx)
    ys = (torch.arange(ny, device=device, dtype=dtype) + 0.5) * (ly / ny)
    gy, gx = torch.meshgrid(ys, xs, indexing="ij")
    return torch.stack([gx.reshape(-1), gy.reshape(-1)], dim=-1)


def _sph_to_grid(pos, vel, grid, h, domain, shepard=True, periodic=False,
                 grid_chunk=2048, eps=1e-12):
    """
    SPH interpolation of particle velocities onto Eulerian nodes.

    pos  : (N, 2) particle positions
    vel  : (N, 2) particle velocities
    grid : (G, 2) Eulerian node positions
    Returns: (G, 2) velocity at the nodes.

    periodic=False (default for the walled tank): plain distances, no wrapping.
    """
    box = torch.tensor(domain, device=pos.device, dtype=pos.dtype)
    out = torch.empty(grid.shape[0], vel.shape[-1],
                      device=pos.device, dtype=pos.dtype)

    for s in range(0, grid.shape[0], grid_chunk):
        g = grid[s:s + grid_chunk]
        d = g[:, None, :] - pos[None, :, :]
        if periodic:
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


# --------------------------------------------------------------------------- #
# streaming accumulators
# --------------------------------------------------------------------------- #
class _RMSEAccum:
    """Accumulate sum of squared error and element count for a global RMSE."""
    def __init__(self):
        self.sse = 0.0
        self.count = 0

    def add(self, pred, target):
        diff = (pred - target) ** 2
        self.sse += diff.sum().item()
        self.count += diff.numel()

    def result(self):
        if self.count == 0:
            return float("nan")
        return math.sqrt(self.sse / self.count)


class _R2Accum:
    """Streaming R^2 about the target mean.

    Accumulate sum(target), sum(target^2), sum((p-t)^2), n.
    ss_tot = sum(t^2) - (sum t)^2 / n.
    """
    def __init__(self):
        self.ss_res = 0.0
        self.sum_t = 0.0
        self.sum_t2 = 0.0
        self.n = 0

    def add(self, pred, target):
        p = pred.reshape(-1)
        t = target.reshape(-1)
        self.ss_res += ((p - t) ** 2).sum().item()
        self.sum_t += t.sum().item()
        self.sum_t2 += (t ** 2).sum().item()
        self.n += t.numel()

    def result(self, eps=1e-12):
        if self.n == 0:
            return float("nan")
        ss_tot = self.sum_t2 - (self.sum_t ** 2) / self.n
        return 1.0 - self.ss_res / (ss_tot + eps)


def _frmse_single(pred_field, target_field):
    """fRMSE for one (2, ny, nx) field pair -> (sum sq err, count)."""
    pf = torch.fft.fftn(pred_field, dim=[1, 2])
    tf = torch.fft.fftn(target_field, dim=[1, 2])
    diff = (pf.abs() - tf.abs()) ** 2
    return diff.sum().item(), diff.numel()


# --------------------------------------------------------------------------- #
# main entry: streaming, one sample at a time
# --------------------------------------------------------------------------- #
class ParticleMetricAccumulator:
    def __init__(self, h=H, domain=DOMAIN, grid_nx=None, grid_ny=None,
                 shepard=True, periodic=False, sinkhorn_kwargs=None):
        self.h = h
        self.domain = domain
        self.shepard = shepard
        self.periodic = periodic
        self.sinkhorn_kwargs = sinkhorn_kwargs or {}

        lx, ly = domain
        self.nx = grid_nx if grid_nx is not None else int(round(lx / DX))
        self.ny = grid_ny if grid_ny is not None else int(round(ly / DX))
        self._grid = None


        self.pos_rmse = {"fluid": _RMSEAccum(), "solid": _RMSEAccum()}
        self.ke_pred = {"fluid": [], "solid": []}
        self.ke_target = {"fluid": [], "solid": []}
        self.sinkhorn = {"fluid": [], "solid": []}
        self.eul_rmse = _RMSEAccum()
        self.eul_r2 = _R2Accum()
        self.eul_frmse_sse = 0.0
        self.eul_frmse_count = 0

    def _grid_for(self, device, dtype):
        if self._grid is None:
            self._grid = _make_grid(self.nx, self.ny, self.domain, device, dtype)
        return self._grid

    def add_sample(self, pred_vel_norm, target_vel_norm, last_pos, particle_type):
        particle_type = particle_type.long()
        pred_vel = vel_unnormalizer(pred_vel_norm, particle_type)
        target_vel = vel_unnormalizer(target_vel_norm, particle_type)

        pred_pos = last_pos + pred_vel
        target_pos = last_pos + target_vel

        fluid_mask = particle_type == FLUID_TYPE
        solid_mask = particle_type == SOLID_TYPE

        for name, mask in (("fluid", fluid_mask), ("solid", solid_mask)):
            if mask.sum() == 0:
                self.ke_pred[name].append(0.0)
                self.ke_target[name].append(0.0)
                continue

            pp, tp = pred_pos[mask], target_pos[mask]
            pv, tv = pred_vel[mask], target_vel[mask]

            # (1,4) position RMSE
            self.pos_rmse[name].add(pp, tp)

            # (2,5) KE per sample = sum_i |v_i|^2
            self.ke_pred[name].append((pv ** 2).sum().item())
            self.ke_target[name].append((tv ** 2).sum().item())

            # (3,6) Sinkhorn on positions of this type
            if mask.sum() >= 2:
                p_np = pp.detach().cpu().numpy().astype(np.float64)
                g_np = tp.detach().cpu().numpy().astype(np.float64)
                self.sinkhorn[name].append(
                    sinkhorn_distance(p_np, g_np, **self.sinkhorn_kwargs))

        # (7,8,9) fluid-velocity Eulerian field for this sample
        if fluid_mask.sum() > 0:
            grid = self._grid_for(pred_vel.device, pred_vel.dtype)
            up = _sph_to_grid(last_pos[fluid_mask], pred_vel[fluid_mask], grid,
                              self.h, self.domain, self.shepard, self.periodic)
            ut = _sph_to_grid(last_pos[fluid_mask], target_vel[fluid_mask], grid,
                              self.h, self.domain, self.shepard, self.periodic)
            pf = up.T.reshape(2, self.ny, self.nx)
            tf = ut.T.reshape(2, self.ny, self.nx)

            self.eul_rmse.add(pf, tf)
            self.eul_r2.add(pf, tf)
            sse, cnt = _frmse_single(pf, tf)
            self.eul_frmse_sse += sse
            self.eul_frmse_count += cnt

    def add_batch(self, pred_vel_norm, target_vel_norm, last_pos, particle_type):
        """Split a (B, N, ...) block into per-sample calls. Only valid when N is
        constant within the block; for variable N call add_sample per sample."""
        B = pred_vel_norm.shape[0]
        for b in range(B):
            self.add_sample(pred_vel_norm[b], target_vel_norm[b],
                            last_pos[b], particle_type[b])

    def compute(self):
        def _ke_rmse(name):
            if not self.ke_pred[name]:
                return float("nan")
            p = torch.tensor(self.ke_pred[name])
            t = torch.tensor(self.ke_target[name])
            return torch.sqrt(torch.mean((p - t) ** 2)).item()

        def _mean_sink(name):
            v = self.sinkhorn[name]
            return float(np.mean(v)) if v else float("nan")

        frmse = (math.sqrt(self.eul_frmse_sse / self.eul_frmse_count)
                 if self.eul_frmse_count else float("nan"))

        return {
            "fluid_rmse_pos": self.pos_rmse["fluid"].result(),
            "fluid_ke_rmse": _ke_rmse("fluid"),
            "fluid_sinkhorn": _mean_sink("fluid"),
            "solid_rmse_pos": self.pos_rmse["solid"].result(),
            "solid_ke_rmse": _ke_rmse("solid"),
            "solid_sinkhorn": _mean_sink("solid"),
            "fluid_eul_vel_rmse": self.eul_rmse.result(),
            "fluid_eul_vel_frmse": frmse,
            "fluid_eul_vel_r2": self.eul_r2.result(),
        }