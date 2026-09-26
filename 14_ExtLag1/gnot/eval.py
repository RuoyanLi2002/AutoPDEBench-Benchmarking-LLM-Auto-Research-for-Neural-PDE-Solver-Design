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

    pred_vels, target_vels, last_positions = [], [], []
    with torch.no_grad():
        for batch in dataloader:
            x = batch.x.to(device).float()
            pos = batch.pos.to(device).float()

            out = model(x, pos)

            pred_vels.append(out.cpu())
            target_vels.append(batch.y.cpu().float())
            last_positions.append(batch.pos.cpu().float())

    if not pred_vels:
        print(f"WARNING: split '{split}' produced no batches; nothing to evaluate.")
        return {}

    pred_vel = torch.cat(pred_vels, dim=0)
    target_vel = torch.cat(target_vels, dim=0)
    last_pos = torch.cat(last_positions, dim=0)

    results = eval_particle_metrics(pred_vel, target_vel, last_pos)

    lines = [
        f"[{split}] RMSE (pos)        : {results['rmse_pos']:.7e}",
        f"[{split}] RMSE (KE)         : {results['ke_rmse']:.7e}",
        f"[{split}] Eul. vel RMSE     : {results['eul_vel_rmse']:.7e}",
        f"[{split}] Eul. vel fRMSE    : {results['eul_vel_frmse']:.7e}",
        f"[{split}] Eul. vel R^2      : {results['eul_vel_r2']:.7f}",
    ]
    if "sinkhorn_pos" in results:
        lines.append(
            f"[{split}] Sinkhorn    : {results['sinkhorn_pos']:.7e}"
        )
    for line in lines:
        print(line)

    eval_txt_path = os.path.join(cfg.training.exp_name, f"eval_{split}_results.txt")
    with open(eval_txt_path, "w") as f:
        f.write("\n".join(lines) + "\n")
    print(f"Eval results saved to {eval_txt_path}")

    return results