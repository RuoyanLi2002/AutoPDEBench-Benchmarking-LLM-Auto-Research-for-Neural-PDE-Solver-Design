import os
import torch

from utils import load_valid, load_test
from metrics import eval_metrics, DOMAIN_NAMES

_LOADERS = {"valid": load_valid, "test": load_test}


def _domain_of(batch):
    node_type = float(batch.x[0, -1])
    if node_type == 0.0:
        return 0
    elif node_type == 1.0:
        return 1
    return 2


def eval(cfg, model, split="test"):
    if split not in _LOADERS:
        raise ValueError(f"split must be one of {list(_LOADERS)}, got '{split}'")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    exp_name = cfg.training.exp_name
    ckpt_path = os.path.join(exp_name, "model.pth")
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
    n_domains = len(dataloader.dataset.domains)

    preds = [[] for _ in range(n_domains)]
    targets = [[] for _ in range(n_domains)]

    with torch.no_grad():
        for batch in dataloader:
            domain = _domain_of(batch)
            batch = batch.to(device)

            out = model(batch)                    # [B * N_nodes, C]

            preds[domain].append(out.detach().cpu())
            targets[domain].append(batch.y.detach().cpu())

    if not any(preds):
        print(f"WARNING: split '{split}' produced no batches; nothing to evaluate.")
        return {}

    
    
    preds = [torch.cat(p, dim=0) if p else torch.empty(0) for p in preds]
    targets = [torch.cat(t, dim=0) if t else torch.empty(0) for t in targets]

    results = eval_metrics(preds, targets)

    lines = [
        f"[{split}] RMSE          : {results['rmse']:.7f}",
        f"[{split}] R^2 (avg)     : {results['r2']:.7f}",
    ]
    for name in DOMAIN_NAMES:
        lines.append(
            f"[{split}] RMSE {name:<9}: {results[f'rmse_{name}']:.7f}"
            f"   R^2 {name:<9}: {results[f'r2_{name}']:.7f}"
        )

    for line in lines:
        print(line)

    os.makedirs(exp_name, exist_ok=True)
    eval_txt_path = os.path.join(exp_name, f"eval_{split}_results.txt")
    with open(eval_txt_path, "w") as f:
        f.write("\n".join(lines) + "\n")
    print(f"Eval results saved to {eval_txt_path}")

    return results