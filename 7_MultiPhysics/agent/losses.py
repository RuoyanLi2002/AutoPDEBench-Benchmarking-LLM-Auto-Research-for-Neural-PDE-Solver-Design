# losses.py
#
# Training objective for the per-field neural surrogate.
#
# Contract:
#     def compute_loss(out, y, x):
#         out : model prediction, [B, C_out, D, H, W]  (normalized field space)
#         y   : ground-truth target, same shape as `out`
#         x   : input conditioning tensor fed to the model, [B, C_in, D, H, W]
#         returns (scalar_loss_tensor, components_dict)

import torch.nn.functional as F


def compute_loss(out, y, x):
    loss = F.mse_loss(out, y)
    components = {
        "mse": float(loss.detach().cpu()),
    }
    return loss, components