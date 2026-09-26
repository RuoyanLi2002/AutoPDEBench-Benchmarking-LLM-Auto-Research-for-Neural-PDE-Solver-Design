import os
import torch

from utils import load_valid, load_test
from train import _densify
from metrics import fluid_batch_stats, eval_particle_metrics

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

    
    
    sq_errs, ke_preds, ke_targets = [], [], []
    with torch.no_grad():
        for batch in dataloader:
            
            
            x, y, pos, fluid_mask = _densify(batch, device)

            out = model(x, pos)

            sq_err, ke_p, ke_t = fluid_batch_stats(out, y, fluid_mask)
            sq_errs.append(sq_err.cpu())
            ke_preds.append(ke_p.cpu())
            ke_targets.append(ke_t.cpu())

    if not sq_errs:
        print(f"WARNING: split '{split}' produced no batches; nothing to evaluate.")
        return {}

    results = eval_particle_metrics(
        torch.cat(sq_errs, dim=0),
        torch.cat(ke_preds, dim=0),
        torch.cat(ke_targets, dim=0),
    )

    lines = [
        f"[{split}] Fluid vel RMSE : {results['fluid_vel_rmse']:.7e}",
        f"[{split}] Fluid KE RMSE  : {results['fluid_ke_rmse']:.7e}",
    ]
    for line in lines:
        print(line)

    eval_txt_path = os.path.join(cfg.training.exp_name, f"eval_{split}_results.txt")
    with open(eval_txt_path, "w") as f:
        f.write("\n".join(lines) + "\n")
    print(f"Eval results saved to {eval_txt_path}")

    return results