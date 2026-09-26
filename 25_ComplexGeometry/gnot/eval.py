import os

import numpy as np
import torch

from utils import load_valid, load_test, get_coef_norm
from shapenetcar import get_samples
from metrics import eval_metrics, coefficient_metrics

try:
    from drag_coefficient import cal_coefficient
    _HAS_DRAG = True
except ImportError:
    _HAS_DRAG = False

_LOADERS = {"valid": load_valid, "test": load_test}


def _get_vallst(cfg):
    """Reconstruct the held-out fold's sample list, filtered exactly the way
    get_datalist() filters (so indices line up with dataloader order)."""
    samples = get_samples(cfg.data.dataset_root)
    vallst = samples[cfg.data.fold_id]

    kept = []
    preprocessed = getattr(cfg.data, "preprocessed", False)
    savedir = getattr(cfg.data, "save_dir", None)
    for s in vallst:
        if preprocessed and savedir is not None:
            if os.path.exists(os.path.join(savedir, s)):
                kept.append(s)
        else:
            press = os.path.join(cfg.data.dataset_root, s, "quadpress_smpl.vtk")
            velo = os.path.join(cfg.data.dataset_root, s, "hexvelo_smpl.vtk")
            if os.path.exists(press) and os.path.exists(velo):
                kept.append(s)
    return kept


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
    vallst = _get_vallst(cfg)


    _, _, mean_out, std_out = get_coef_norm(cfg)
    mean_out = torch.as_tensor(np.asarray(mean_out), dtype=torch.float32)
    std_out = torch.as_tensor(np.asarray(std_out), dtype=torch.float32)

    total_vol, total_surf = 0.0, 0.0
    ls_pred_coef, ls_gt_coef = [], []
    drag_ok = _HAS_DRAG
    count = 0

    with torch.no_grad():
        for batch in dataloader:
            cfd_data, geom_data = batch
            x = cfd_data.x[None, :, :].to(device)

            pos = cfd_data.pos
            pos = pos[None, :, :]
            pos = pos.to(device)

            out = model(x, pos)                       # [1, N, C] (or [N, C])
            out = out.squeeze(0).detach().cpu()  # [N, C]
            target = cfd_data.y.cpu()            # [N, C]
            surf = cfd_data.surf.cpu()           # [N] bool


            sample_metrics = eval_metrics(out, target, surf)
            total_vol += sample_metrics["rmse_volume"]
            total_surf += sample_metrics["rmse_surface"]


            if drag_ok and count < len(vallst):
                out_phys = out * (std_out + 1e-8) + mean_out
                target_phys = target * (std_out + 1e-8) + mean_out

                pred_press = out_phys[surf, -1]
                gt_press = target_phys[surf, -1]
                pred_surf_velo = out_phys[surf, :-1]
                gt_surf_velo = target_phys[surf, :-1]

                sample_name = vallst[count].split("/")[1]
                try:
                    pred_coef = cal_coefficient(
                        sample_name,
                        pred_press[:, None].numpy(),
                        pred_surf_velo.numpy(),
                    )
                    gt_coef = cal_coefficient(
                        sample_name,
                        gt_press[:, None].numpy(),
                        gt_surf_velo.numpy(),
                    )
                    ls_pred_coef.append(pred_coef)
                    ls_gt_coef.append(gt_coef)
                except Exception as e:
                    print(f"WARNING: drag coefficient failed for "
                          f"'{vallst[count]}' ({e}); skipping coefficient "
                          f"metrics for this run.")
                    drag_ok = False

            count += 1

    if count == 0:
        print(f"WARNING: split '{split}' produced no batches; nothing to evaluate.")
        return {}

    results = {
        "rmse_volume": total_vol / count,
        "rmse_surface": total_surf / count,
    }

    if drag_ok and ls_gt_coef:
        results.update(coefficient_metrics(ls_pred_coef, ls_gt_coef))

    lines = [
        f"[{split}] RMSE (volume)      : {results['rmse_volume']:.7f}",
        f"[{split}] RMSE (surface)     : {results['rmse_surface']:.7f}",
    ]
    if "rmse_coefficient" in results:
        lines.append(f"[{split}] RMSE (coefficient) : {results['rmse_coefficient']:.7f}")
    if "r2" in results:
        lines.append(f"[{split}] R^2  (coefficient) : {results['r2']:.7f}")

    for line in lines:
        print(line)

    eval_txt_path = os.path.join(cfg.training.exp_name, f"eval_{split}_results.txt")
    with open(eval_txt_path, "w") as f:
        f.write("\n".join(lines) + "\n")
    print(f"Eval results saved to {eval_txt_path}")

    return results