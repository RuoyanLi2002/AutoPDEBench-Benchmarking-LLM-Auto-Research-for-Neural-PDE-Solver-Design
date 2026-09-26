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

    dataloader = _LOADERS[split](cfg)

    # Each frame is an independent sample; collect them as [N, C, nx, ny, nz].
    # 3D volumes -- no time axis, no sequence stacking.
    preds, targets = [], []
    with torch.no_grad():
        for x, y in dataloader:
            x = x.to(device)
            y = y.to(device)

            out = model(x)          # [B, C, nx, ny, nz]
            preds.append(out.cpu())
            targets.append(y.cpu())

    if not preds:
        print(f"WARNING: split '{split}' produced no batches; nothing to evaluate.")
        return {}

    pred = torch.cat(preds, dim=0)          # [N, C, nx, ny, nz]
    target = torch.cat(targets, dim=0)      # [N, C, nx, ny, nz]

    results = eval_metrics(pred, target)

    print(f"[{split}] RMSE  : {results['rmse']:.7e}")
    print(f"[{split}] R^2   : {results['r2']:.7f}")
    print(f"[{split}] fRMSE : {results['frmse']:.7e}")
    print(f"[{split}] KE    : {results['ke']:.7e}")

    eval_txt_path = os.path.join(cfg.training.exp_name, f"eval_{split}_results.txt")
    with open(eval_txt_path, "w") as f:
        f.write(f"[{split}] RMSE  : {results['rmse']:.7e}\n")
        f.write(f"[{split}] R^2   : {results['r2']:.7f}\n")
        f.write(f"[{split}] fRMSE : {results['frmse']:.7e}\n")
        f.write(f"[{split}] KE    : {results['ke']:.7e}\n")
    print(f"Eval results saved to {eval_txt_path}")

    return results