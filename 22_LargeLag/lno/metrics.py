import torch

from utils import vel_unnormalizer


def fluid_batch_stats(pred_vel_norm, target_vel_norm, fluid_mask):
    """Per-batch sufficient statistics over FLUID particles only.

    pred_vel_norm   : (B, N, 2) model output, normalized velocity
    target_vel_norm : (B, N, 2) dataset y (normalized)
    fluid_mask      : (B, N) bool, True only for real, supervised fluid
                      particles (excludes dense-batch padding, solids, and
                      particles absent at the target frame -- i.e. the same
                      mask used for the training loss)

    Returns:
        sq_err    : (P * 2,) per-component squared velocity errors of the
                    masked fluid particles (unnormalized / physical units)
        ke_pred   : (B,) kinetic energy per sample, sum_i m_i |v_i|^2 with
                    m_i = 1, fluid particles only
        ke_target : (B,) same for the target velocities
    """
    pred = vel_unnormalizer(pred_vel_norm)
    target = vel_unnormalizer(target_vel_norm)
    mask = fluid_mask.bool()

    sq_err = ((pred - target) ** 2)[mask].reshape(-1)          # (P*2,)

    ke_pred = ((pred ** 2).sum(dim=-1) * mask).sum(dim=-1)     # (B,)
    ke_target = ((target ** 2).sum(dim=-1) * mask).sum(dim=-1) # (B,)

    return sq_err, ke_pred, ke_target


def eval_particle_metrics(sq_err, ke_pred, ke_target):
    """Finalize metrics from accumulated per-batch statistics.

    sq_err    : (M,) all per-component squared velocity errors (fluid only)
    ke_pred   : (B_total,) per-sample fluid kinetic energy, predictions
    ke_target : (B_total,) per-sample fluid kinetic energy, targets
    """
    vel_rmse = torch.sqrt(sq_err.mean())
    ke_rmse = torch.sqrt(torch.mean((ke_pred - ke_target) ** 2))

    return {
        "fluid_vel_rmse": vel_rmse.item(),
        "fluid_ke_rmse": ke_rmse.item(),
    }