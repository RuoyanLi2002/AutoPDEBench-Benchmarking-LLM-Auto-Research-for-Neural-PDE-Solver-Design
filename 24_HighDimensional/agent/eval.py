import os

import torch

from utils import load_valid, load_test
from metrics import StreamingMetrics


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



    acc = StreamingMetrics()
    n_batches = 0
    with torch.no_grad():
        for x, y in dataloader:
            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)

            out = model(x)      # [B, 2, nvpar, nmu, ns, nkx, nky]
            acc.update(out, y)
            n_batches += 1

    if n_batches == 0:
        print(f"WARNING: split '{split}' produced no batches; nothing to evaluate.")
        return {}

    results = acc.compute()

    print(f"[{split}] RMSE     : {results['rmse']:.7f}")
    print(f"[{split}] R^2      : {results['r2']:.7f}")

    os.makedirs(cfg.training.exp_name, exist_ok=True)
    eval_txt_path = os.path.join(cfg.training.exp_name, f"eval_{split}_results.txt")
    with open(eval_txt_path, "w") as f:
        f.write(f"[{split}] RMSE     : {results['rmse']:.7f}\n")
        f.write(f"[{split}] R^2      : {results['r2']:.7f}\n")
    print(f"Eval results saved to {eval_txt_path}")

    return results