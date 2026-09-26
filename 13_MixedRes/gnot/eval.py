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

    se_total = 0.0
    n_total = 0
    fse_total = 0.0
    n_freq_total = 0
    ke_weighted = 0.0
    n_ke_total = 0

    ss_res_total = 0.0
    target_sum = 0.0
    target_sq_sum = 0.0

    n_samples = 0
    with torch.no_grad():
        for x, pos, y in dataloader:
            x = x.to(device)
            pos = pos.to(device)
            y = y.to(device)

            out = model(x, pos)          # [B, C, H, W]

            
            for b in range(out.shape[0]):
                pred = out[b:b + 1]      # [1, C, H, W]
                target = y[b:b + 1]      # [1, C, H, W]

                
                m = eval_metrics(pred, target)

                n = target.numel()
                n_freq = torch.fft.fftn(target, dim=[2, 3]).numel()

                
                se_total += (m["rmse"] ** 2) * n
                n_total += n

                fse_total += (m["frmse"] ** 2) * n_freq
                n_freq_total += n_freq

                ke_weighted += m["ke"] * n
                n_ke_total += n


                ss_res_total += torch.sum((pred - target) ** 2).item()
                target_sum += torch.sum(target).item()
                target_sq_sum += torch.sum(target ** 2).item()

                n_samples += 1

    if n_samples == 0:
        print(f"WARNING: split '{split}' produced no batches; nothing to evaluate.")
        return {}

    eps = 1e-12

    rmse = (se_total / n_total) ** 0.5
    frmse = (fse_total / n_freq_total) ** 0.5
    ke = ke_weighted / n_ke_total

    ybar = target_sum / n_total
    ss_tot = target_sq_sum - n_total * (ybar ** 2)
    r2 = 1.0 - ss_res_total / (ss_tot + eps)

    results = {
        "rmse": rmse,
        "r2": r2,
        "frmse": frmse,
        "ke": ke,
    }

    print(f"[{split}] RMSE  : {results['rmse']:.7f}")
    print(f"[{split}] R^2   : {results['r2']:.7f}")
    print(f"[{split}] fRMSE : {results['frmse']:.7e}")
    print(f"[{split}] KE    : {results['ke']:.7e}")

    eval_txt_path = os.path.join(cfg.training.exp_name, f"eval_{split}_results.txt")
    with open(eval_txt_path, "w") as f:
        f.write(f"[{split}] RMSE  : {results['rmse']:.7f}\n")
        f.write(f"[{split}] R^2   : {results['r2']:.7f}\n")
        f.write(f"[{split}] fRMSE : {results['frmse']:.7e}\n")
        f.write(f"[{split}] KE    : {results['ke']:.7e}\n")
    print(f"Eval results saved to {eval_txt_path}")

    return results