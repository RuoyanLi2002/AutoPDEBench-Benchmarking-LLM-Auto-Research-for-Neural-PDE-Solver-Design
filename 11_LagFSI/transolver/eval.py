import os
import torch

from utils import load_valid, load_test
from metrics import ParticleMetricAccumulator

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

    
    
    domain = getattr(cfg.data, "eul_domain", (0.1, 0.25))
    grid_nx = getattr(cfg.data, "eul_grid_nx", 50)
    grid_ny = getattr(cfg.data, "eul_grid_ny", 125)
    h = getattr(cfg.data, "eul_h", 2e-3)

    acc = ParticleMetricAccumulator(
        h=h, domain=domain, grid_nx=grid_nx, grid_ny=grid_ny,
        shepard=True, periodic=False,
    )

    n_samples = 0
    with torch.no_grad():
        for batch in dataloader:
            for data in batch.to_data_list():
                x = data.x.to(device).float()          # (1, N, F)

                out = model(x)                          # (1, N, 2)

                pred = out.squeeze(0).cpu().float()             # (N, 2)
                target = data.y.squeeze(0).cpu().float()        # (N, 2)
                last_pos = data.pos.squeeze(0).cpu().float()    # (N, 2)
                ptype = data.particle_information.squeeze(0).squeeze(-1).long().cpu()  # (N,)

                acc.add_sample(pred, target, last_pos, ptype)
                n_samples += 1

    if n_samples == 0:
        print(f"WARNING: split '{split}' produced no samples; nothing to evaluate.")
        return {}

    results = acc.compute()

    lines = [
        f"[{split}] Fluid RMSE (pos)       : {results['fluid_rmse_pos']:.7e}",
        f"[{split}] Fluid RMSE (KE)        : {results['fluid_ke_rmse']:.7e}",
        f"[{split}] Fluid Sinkhorn div     : {results['fluid_sinkhorn']:.7e}",
        f"[{split}] Solid RMSE (pos)       : {results['solid_rmse_pos']:.7e}",
        f"[{split}] Solid RMSE (KE)        : {results['solid_ke_rmse']:.7e}",
        f"[{split}] Solid Sinkhorn div     : {results['solid_sinkhorn']:.7e}",
        f"[{split}] Fluid Eul. vel RMSE    : {results['fluid_eul_vel_rmse']:.7e}",
        f"[{split}] Fluid Eul. vel fRMSE   : {results['fluid_eul_vel_frmse']:.7e}",
        f"[{split}] Fluid Eul. vel R^2     : {results['fluid_eul_vel_r2']:.7f}",
    ]
    for line in lines:
        print(line)

    eval_txt_path = os.path.join(cfg.training.exp_name, f"eval_{split}_results.txt")
    with open(eval_txt_path, "w") as f:
        f.write("\n".join(lines) + "\n")
    print(f"Eval results saved to {eval_txt_path}")

    return results