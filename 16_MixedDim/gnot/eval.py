import os
import torch

from utils import load_valid, load_test
from metrics import eval_metrics

_LOADERS = {"valid": load_valid, "test": load_test}


def eval(cfg, model, split="test"):
    if split not in _LOADERS:
        raise ValueError(f"split must be one of {list(_LOADERS)}, got '{split}'")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    ckpt_path = os.path.join(cfg.exp_name, "model.pth")
    if os.path.exists(ckpt_path):
        state_dict = torch.load(ckpt_path, map_location=device)
        model.load_state_dict(state_dict)
        print(f"Loaded model weights from {ckpt_path}")
    else:
        print(f"WARNING: no checkpoint found at {ckpt_path}; "
              f"evaluating current (untrained) weights.")

    model.to(device)
    model.eval()

    
    dataloaders = _LOADERS[split](cfg)

    per_dim = {}   # full metric dict per dimensionality
    for key, dataloader in dataloaders.items():
        preds, targets = [], []
        with torch.no_grad():
            for x, y in dataloader:          # (N, num_points, 1)
                x = x.to(device)
                y = y.to(device)

                out = model(x)               # (N, num_points, 1)
                preds.append(out.cpu())
                targets.append(y.cpu())

        if not preds:
            print(f"WARNING: split '{split}' [{key}] produced no batches; skipping.")
            continue

        pred = torch.cat(preds, dim=0)
        target = torch.cat(targets, dim=0)
        per_dim[key] = eval_metrics(pred, target)

    if not per_dim:
        print(f"WARNING: split '{split}' produced no batches; nothing to evaluate.")
        return {}


    n = len(per_dim)
    combined = {
        "rmse": sum(m["rmse"] for m in per_dim.values()) / n,
        "r2": sum(m["r2"] for m in per_dim.values()) / n,
        "frmse": sum(m["frmse"] for m in per_dim.values()) / n,
    }

    print(f"[{split}] RMSE  : {combined['rmse']:.7f}")
    print(f"[{split}] R^2   : {combined['r2']:.7f}")
    print(f"[{split}] fRMSE : {combined['frmse']:.7e}")
    for key, m in per_dim.items():
        print(f"[{split}][{key}] RMSE  : {m['rmse']:.7f}")

    exp_name = getattr(cfg.training, "exp_name", cfg.exp_name)
    eval_txt_path = os.path.join(exp_name, f"eval_{split}_results.txt")
    with open(eval_txt_path, "w") as f:
        f.write(f"[{split}] RMSE  : {combined['rmse']:.7f}\n")
        f.write(f"[{split}] R^2   : {combined['r2']:.7f}\n")
        f.write(f"[{split}] fRMSE : {combined['frmse']:.7e}\n")
        for key, m in per_dim.items():
            f.write(f"[{split}][{key}] RMSE  : {m['rmse']:.7f}\n")
    print(f"Eval results saved to {eval_txt_path}")

    combined["per_dim_rmse"] = {k: m["rmse"] for k, m in per_dim.items()}
    return combined