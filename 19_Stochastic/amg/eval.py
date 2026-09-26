import os
import torch

from utils import load_valid, load_test
from metrics import eval_metrics

_LOADERS = {"valid": load_valid, "test": load_test}


def _infer_spatial_shape(pos):
    return tuple(int(torch.unique(pos[:, d]).numel()) for d in range(pos.shape[1]))


def _nodes_to_grid(node_feats, num_graphs, spatial_shape):
    C = node_feats.shape[-1]
    per_graph = node_feats.reshape(num_graphs, -1, C)   # [B, N, C]
    per_graph = per_graph.permute(0, 2, 1).contiguous() # [B, C, N]
    return per_graph.reshape(num_graphs, C, *spatial_shape)


def eval(cfg, model, split="test"):
    if split not in _LOADERS:
        raise ValueError(f"split must be one of {list(_LOADERS)}, got '{split}'")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    exp_dir = cfg.training.exp_name
    ckpt_path = os.path.join(exp_dir, "model_epoch10.pth")
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

    spatial_shape = None
    preds, targets = [], []
    with torch.no_grad():
        for batch in dataloader:
            batch = batch.to(device)

            if spatial_shape is None:
                n_per_graph = batch.num_nodes // batch.num_graphs
                spatial_shape = _infer_spatial_shape(batch.pos[:n_per_graph])

            out = model(batch)

            
            if out.dim() == 3:
                out = out.reshape(-1, out.shape[-1])

            preds.append(
                _nodes_to_grid(out, batch.num_graphs, spatial_shape).cpu()
            )
            targets.append(
                _nodes_to_grid(batch.y, batch.num_graphs, spatial_shape).cpu()
            )

    if not preds:
        print(f"WARNING: split '{split}' produced no batches; nothing to evaluate.")
        return {}

    pred = torch.cat(preds, dim=0)      # [N, C, H, W]
    target = torch.cat(targets, dim=0)  # [N, C, H, W]
    results = eval_metrics(pred, target)

    print(f"[{split}] RMSE  : {results['rmse']:.7f}")
    print(f"[{split}] R^2   : {results['r2']:.7f}")
    print(f"[{split}] fRMSE : {results['frmse']:.7e}")

    eval_txt_path = os.path.join(exp_dir, f"eval_{split}_results.txt")
    with open(eval_txt_path, "w") as f:
        f.write(f"[{split}] RMSE  : {results['rmse']:.7f}\n")
        f.write(f"[{split}] R^2   : {results['r2']:.7f}\n")
        f.write(f"[{split}] fRMSE : {results['frmse']:.7e}\n")
    print(f"Eval results saved to {eval_txt_path}")

    return results