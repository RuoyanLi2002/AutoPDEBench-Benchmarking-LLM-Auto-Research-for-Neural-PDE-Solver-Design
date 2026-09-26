import os
import torch
from torch_geometric.utils import unbatch

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
            batch = batch.to(device)
            x = batch.x.to(device).float()
            bch = batch.batch.to(device).float()
            particle_information = batch.particle_information.to(device).float()

            out = model(x, bch, particle_information)

            # Split node-dim tensors back per sample via the batch assignment
            # vector; handles variable particle counts cleanly.
            node_batch = batch.batch.cpu()
            preds = unbatch(out.detach().cpu().float(), node_batch)
            targets = unbatch(batch.y.cpu().float(), node_batch)
            positions = unbatch(batch.pos.cpu().float(), node_batch)
            ptypes = unbatch(
                batch.particle_information.squeeze(-1).long().cpu(), node_batch)

            for pred, target, last_pos, ptype in zip(
                    preds, targets, positions, ptypes):
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