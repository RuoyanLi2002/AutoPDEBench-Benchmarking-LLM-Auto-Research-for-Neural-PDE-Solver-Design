import os
import torch

from utils import load_valid, load_couple, load_config, normalize, renormalize, assemble_cond
from metrics import eval_metrics

FIELDS = ["neutron", "solid", "fluid"]


def _load_weights(model, exp_name, device):
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
    return model


def _print_and_save(cfg, split, lines):
    for line in lines:
        print(line)
    os.makedirs(cfg.training.exp_name, exist_ok=True)
    eval_txt_path = os.path.join(cfg.training.exp_name, f"eval_{split}_results.txt")
    with open(eval_txt_path, "w") as f:
        f.write("\n".join(lines) + "\n")
    print(f"Eval results saved to {eval_txt_path}")


def _metric_lines(tag, results):
    return [
        f"[{tag}] RMSE  : {results['rmse']:.7f}",
        f"[{tag}] fRMSE : {results['frmse']:.7e}",
    ]


def eval_valid(cfg, model):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    _load_weights(model, cfg.training.exp_name, device)

    preds, targets = [], []
    with torch.no_grad():
        for x, y in load_valid(cfg):
            preds.append(model(x.to(device)).cpu())
            targets.append(y)

    pred = torch.cat(preds, dim=0)
    target = torch.cat(targets, dim=0)

    results = eval_metrics(pred, target)
    _print_and_save(cfg, "valid", _metric_lines(f"valid/{cfg.data.field}", results))
    return results


def k(t):
    return 17.5 * (1 - 0.223) / (1 + 0.161) + 1.54e-2 * (1 + 0.0061) / (1 + 0.161) * t + 9.38e-6 * t * t


def update_neutron(model, neu, fuel, fluid, bc):
    fuel_n = normalize(fuel, "solid")
    fluid_n = normalize(fluid[:, :1], "fluid")
    bc_n = normalize(bc, "neutron")
    x = assemble_cond("neutron", [bc_n, torch.cat((fuel_n, fluid_n), dim=-1)])
    return renormalize(model(x), "neutron")


def update_solid(model, neu, fuel, fluid, bc, neu_norm="neutron"):
    neu_n = normalize(neu[..., :8], neu_norm)
    fluid_n = normalize(fluid[:, 0:1, :, :, 0:1], "fluid")
    x = assemble_cond("solid", [neu_n, fluid_n])
    return renormalize(model(x), "solid")


def update_fluid(model, neu, fuel, fluid, bc):
    flux = (fuel[..., -2:-1] - fuel[..., -1:]) * k(fuel[..., -1:])
    x = assemble_cond("fluid", [normalize(flux, "flux")])
    return renormalize(model(x), "fluid")


def _rel_update(new, old):
    b = new.shape[0]
    delta = (new - old).reshape(b, -1).norm(dim=1)
    norm = new.reshape(b, -1).norm(dim=1)
    return (delta / norm).mean().item()


def eval_test(cfg, model_builder):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    coeff = cfg.couple.damping
    max_iter = cfg.couple.max_iter
    tol = cfg.couple.tol
    neu_norm = getattr(cfg.couple, "solid_neu_norm", "neutron")

    models = {}
    for field in FIELDS:
        field_cfg = load_config(getattr(cfg.couple, f"{field}_config"))
        models[field] = _load_weights(model_builder(field_cfg), field_cfg.training.exp_name, device)

    bc, neu, fuel, fluid = load_couple(cfg)
    bc, neu, fuel, fluid = bc.to(device), neu.to(device), fuel.to(device), fluid.to(device)
    print(f"Coupled test set: {bc.shape[0]} samples, "
          f"solid neutron-cond normalization: '{neu_norm}'")

    neu_p = renormalize(torch.ones_like(neu) * 0.5, "neutron")
    fuel_p = renormalize(torch.ones_like(fuel) * 0.5, "solid")
    fluid_p = renormalize(torch.ones_like(fluid) * 0.5, "fluid")

    with torch.no_grad():
        for i in range(max_iter):
            neu_old, fuel_old, fluid_old = neu_p, fuel_p, fluid_p
            neu_p = update_neutron(models["neutron"], neu_p, fuel_p, fluid_p, bc) * coeff + neu_old * (1 - coeff)
            fuel_p = update_solid(models["solid"], neu_p, fuel_p, fluid_p, bc, neu_norm) * coeff + fuel_old * (1 - coeff)
            fluid_p = update_fluid(models["fluid"], neu_p, fuel_p, fluid_p, bc) * coeff + fluid_old * (1 - coeff)

            d1 = _rel_update(neu_p, neu_old)
            d2 = _rel_update(fuel_p, fuel_old)
            d3 = _rel_update(fluid_p, fluid_old)
            delta = (d1 + d2 + d3) / 3
            print(f"iter {i}: delta={delta:.3e} (neu={d1:.3e}, solid={d2:.3e}, fluid={d3:.3e})")
            if delta < tol:
                print(f"converged at iteration {i}")
                break
            if i == max_iter - 1:
                print("reached max iterations without convergence")

    pairs = {
        "neutron": (normalize(neu_p, "neutron"), normalize(neu, "neutron")),
        "solid": (normalize(fuel_p, "solid"), normalize(fuel, "solid")),
        "fluid": (normalize(fluid_p, "fluid"), normalize(fluid, "fluid")),
    }

    lines = []
    all_results = {}
    for field, (pred, target) in pairs.items():
        results = eval_metrics(pred.cpu(), target.cpu())
        all_results[field] = results
        lines.extend(_metric_lines(f"test/{field}", results))
    lines.append(f"[test] fixed-point iterations: {i + 1}, final delta: {delta:.3e}")

    _print_and_save(cfg, "test", lines)
    all_results["iterations"] = i + 1
    all_results["final_delta"] = delta
    return all_results


def eval(cfg, model, split="test", model_builder=None):
    if split == "valid":
        return eval_valid(cfg, model)
    elif split == "test":
        if model_builder is None:
            raise ValueError("split 'test' requires model_builder to construct the three field surrogates")
        return eval_test(cfg, model_builder)
    else:
        raise ValueError(f"split must be 'valid' or 'test', got '{split}'")
