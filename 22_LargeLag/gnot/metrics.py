import torch

from utils import vel_unnormalizer


def fluid_batch_stats(pred_vel_norm, target_vel_norm, fluid_mask):
    pred = vel_unnormalizer(pred_vel_norm)
    target = vel_unnormalizer(target_vel_norm)
    mask = fluid_mask.bool()

    sq_err = ((pred - target) ** 2)[mask].reshape(-1)          # (P*2,)

    ke_pred = ((pred ** 2).sum(dim=-1) * mask).sum(dim=-1)     # (B,)
    ke_target = ((target ** 2).sum(dim=-1) * mask).sum(dim=-1) # (B,)

    return sq_err, ke_pred, ke_target


def eval_particle_metrics(sq_err, ke_pred, ke_target):
    vel_rmse = torch.sqrt(sq_err.mean())
    ke_rmse = torch.sqrt(torch.mean((ke_pred - ke_target) ** 2))

    return {
        "fluid_vel_rmse": vel_rmse.item(),
        "fluid_ke_rmse": ke_rmse.item(),
    }