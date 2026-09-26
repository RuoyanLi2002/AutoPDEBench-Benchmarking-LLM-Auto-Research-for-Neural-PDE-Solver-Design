import os
import torch

from utils import load_valid, load_test
from metrics import eval_particle_metrics

_LOADERS = {"valid": load_valid, "test": load_test}


def eval(cfg, model, split="test"):
    if split not in _LOADERS:
        raise ValueError(f"split must be one of {list(_LOADERS)}, got '{split}'")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    ckpt_path = os.path.join(cfg.training.exp_name, "model.pth")
    if os.path.exists(ckpt_path):
        state_dict = torch.load(ckpt_path, map_location=device)
        model.load_state_dict(state_dict)
        print(f"Loaded model weights from {ckpt_path}")
    else:
        print(f"WARNING: no checkpoint found at {ckpt_path}; "
              f"evaluating current (untrained) weights.")

    model.to(device)
    model.eval()

    dataloader = _LOADERS[split](cfg)

    # Batches are padded to the per-batch max particle count, which differs
    # between batches, so results are unpadded into per-sample lists here.
    pred_vels, target_vels, positions = [], [], []
    with torch.no_grad():
        for batch in dataloader:
            x = batch["x"].to(device, non_blocking=True).float()
            in_mask = batch["in_mask"].to(device, non_blocking=True).float()
            x_out = batch["x_out"].to(device, non_blocking=True).float()

            out = model(x, x_out, in_mask=in_mask)          # (B, N_out, 2)

            out = out.cpu()
            out_mask = batch["out_mask"]
            for b in range(out.shape[0]):
                n = int(out_mask[b].sum().item())
                pred_vels.append(out[b, :n])
                target_vels.append(batch["y"][b, :n].float())
                positions.append(batch["x_out"][b, :n].float())

    if not pred_vels:
        print(f"WARNING: split '{split}' produced no batches; nothing to evaluate.")
        return {}

    results = eval_particle_metrics(pred_vels, target_vels, positions,
                                    device=device)

    lines = [
        f"[{split}] RMSE (velocity)   : {results['vel_rmse']:.7e}",
        f"[{split}] RMSE (KE)         : {results['ke_rmse']:.7e}",
        f"[{split}] Eul. vel RMSE     : {results['eul_vel_rmse']:.7e}",
        f"[{split}] Eul. vel fRMSE    : {results['eul_vel_frmse']:.7e}",
        f"[{split}] Eul. vel R^2      : {results['eul_vel_r2']:.7f}",
    ]
    for line in lines:
        print(line)

    eval_txt_path = os.path.join(cfg.training.exp_name, f"eval_{split}_results.txt")
    with open(eval_txt_path, "w") as f:
        f.write("\n".join(lines) + "\n")
    print(f"Eval results saved to {eval_txt_path}")

    return results