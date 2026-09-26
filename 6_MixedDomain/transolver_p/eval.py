import os
import torch

from utils import load_valid, load_test
from metrics import eval_metrics, DOMAIN_NAMES

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

    
    
    group_sizes = dataloader.dataset.group_sizes
    n_domains = len(group_sizes)

    preds = [[] for _ in range(n_domains)]
    targets = [[] for _ in range(n_domains)]

    domain, seen = 0, 0
    with torch.no_grad():
        for x, y in dataloader:
            x = x.to(device)

            out = model(x)                    # [B, N_nodes, C]
            preds[domain].append(out.cpu())
            targets[domain].append(y)

            seen += y.shape[0]
            if seen >= group_sizes[domain]:
                domain += 1
                seen = 0

    if not any(preds):
        print(f"WARNING: split '{split}' produced no batches; nothing to evaluate.")
        return {}

    
    
    preds = [torch.cat(p, dim=0) for p in preds]      # each [N_t, N_nodes, C]
    targets = [torch.cat(t, dim=0) for t in targets]

    results = eval_metrics(preds, targets)

    lines = [
        f"[{split}] RMSE          : {results['rmse']:.7f}",
        f"[{split}] R^2 (avg)     : {results['r2']:.7f}",
    ]
    for name in DOMAIN_NAMES:
        lines.append(f"[{split}] RMSE {name:<9}: {results[f'rmse_{name}']:.7f}")

    for line in lines:
        print(line)

    eval_txt_path = os.path.join(cfg.training.exp_name, f"eval_{split}_results.txt")
    with open(eval_txt_path, "w") as f:
        f.write("\n".join(lines) + "\n")
    print(f"Eval results saved to {eval_txt_path}")

    return results