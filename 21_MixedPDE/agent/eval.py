import os
from collections import defaultdict

import torch

from utils import load_valid, load_test
from metrics import eval_metrics, PDE_NAMES


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

    
    grouped = defaultdict(lambda: {"pred": [], "target": [], "pde": []})
    with torch.no_grad():
        for x, y, pde in dataloader:
            x = x.to(device)
            y = y.to(device)
            pde = pde.to(device)

            out = model(x, pde)          # [B, C, H, W]

            key = tuple(y.shape[1:])
            grouped[key]["pred"].append(out.cpu())
            grouped[key]["target"].append(y.cpu())
            grouped[key]["pde"].append(pde.cpu())

    if not grouped:
        print(f"WARNING: split '{split}' produced no batches; nothing to evaluate.")
        return {}

    preds, targets, pde_ids = [], [], []
    for key in sorted(grouped):
        preds.append(torch.cat(grouped[key]["pred"], dim=0))      # [N_g, C, H, W]
        targets.append(torch.cat(grouped[key]["target"], dim=0))  # [N_g, C, H, W]
        pde_ids.append(torch.cat(grouped[key]["pde"], dim=0))     # [N_g]

    results = eval_metrics(preds, targets, pde_ids)

    print(f"[{split}] RMSE  : {results['rmse']:.7f}")
    print(f"[{split}] R^2   : {results['r2']:.7f}")
    print(f"[{split}] fRMSE : {results['frmse']:.7e}")
    for name in PDE_NAMES.values():
        print(f"[{split}] RMSE ({name}) : {results[f'rmse_{name}']:.7f}")

    eval_txt_path = os.path.join(cfg.training.exp_name, f"eval_{split}_results.txt")
    with open(eval_txt_path, "w") as f:
        f.write(f"[{split}] RMSE  : {results['rmse']:.7f}\n")
        f.write(f"[{split}] R^2   : {results['r2']:.7f}\n")
        f.write(f"[{split}] fRMSE : {results['frmse']:.7e}\n")
        for name in PDE_NAMES.values():
            f.write(f"[{split}] RMSE ({name}) : {results[f'rmse_{name}']:.7f}\n")
    print(f"Eval results saved to {eval_txt_path}")

    return results