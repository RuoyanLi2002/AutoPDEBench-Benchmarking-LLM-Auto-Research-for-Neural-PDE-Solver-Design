import os
import pickle
import torch

from utils import load_trajectories
from metrics import rollout_metrics

_SPLITS = ("valid", "test")


def eval(cfg, model, split="test"):
    if split not in _SPLITS:
        raise ValueError(f"split must be one of {list(_SPLITS)}, got '{split}'")

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

    # (num_traj, T, GridSize), e.g. (10, 5001, 256)
    traj = load_trajectories(cfg, split)[:, :5001, :]
    num_traj, T, grid_size = traj.shape
    num_steps = T - 1 
    print(f"[{split}] rolling out {num_traj} trajectories "
          f"for {num_steps} steps (grid size {grid_size})")


    state = traj[:, 0].to(device)                       # (num_traj, GridSize)
    preds = torch.empty(num_traj, num_steps, grid_size)  # kept on CPU

    with torch.no_grad():
        for t in range(num_steps):
            out = model(state.unsqueeze(-1))            # (num_traj, GridSize, 1)
            state = out.squeeze(-1)
            preds[:, t] = state.cpu()

    target = traj[:, 1:]                                # (num_traj, num_steps, GridSize)

    
    results = rollout_metrics(preds, target)
    rmse, r2, frmse = results["rmse"], results["r2"], results["frmse"]

    
    pkl_path = os.path.join(cfg.training.exp_name, f"eval_{split}_rollout.pkl")
    with open(pkl_path, "wb") as f:
        pickle.dump(
            {
                "rmse": rmse.numpy(),
                "r2": r2.numpy(),
                "frmse": frmse.numpy(),
                "num_traj": num_traj,
                "num_steps": num_steps,
                "grid_size": grid_size,
            },
            f,
        )

    txt_path = os.path.join(cfg.training.exp_name, f"eval_{split}_rollout.txt")
    with open(txt_path, "w") as f:
        f.write(f"# Rollout evaluation on '{split}' split: "
                f"{num_traj} trajectories, {num_steps} steps\n")
        f.write(f"# step\tRMSE\tfRMSE\tR^2\n")
        for t in range(num_steps):
            f.write(f"{t + 1}\t{rmse[t]:.7e}\t{frmse[t]:.7e}\t{r2[t]:.7f}\n")

    
    print(f"\n===== [{split}] rollout results over {num_steps} steps =====")
    print(f"RMSE  : mean = {rmse.mean():.7e}, final (t={num_steps}) = {rmse[-1]:.7e}")
    print(f"fRMSE : mean = {frmse.mean():.7e}, final (t={num_steps}) = {frmse[-1]:.7e}")
    print(f"R^2   : mean = {r2.mean():.7f}, final (t={num_steps}) = {r2[-1]:.7f}")
    print(f"Per-step curves saved to {txt_path} and {pkl_path}")

    return results