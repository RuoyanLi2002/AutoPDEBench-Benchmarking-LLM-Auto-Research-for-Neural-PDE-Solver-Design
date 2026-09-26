import os
import yaml
import argparse
from types import SimpleNamespace
import numpy as np
import torch

from utils import load_train
from model import Model
from train import train
from eval import eval

class Config(SimpleNamespace):
    """Namespace supporting nested attribute access (e.g. cfg.data.dataset_root)."""
    pass


def _dict_to_namespace(d):
    if isinstance(d, dict):
        return Config(**{k: _dict_to_namespace(v) for k, v in d.items()})
    elif isinstance(d, list):
        return [_dict_to_namespace(v) for v in d]
    else:
        return d


def _deep_merge(base, override):
    """Recursively merge `override` into `base`. Values in `override` win on conflict."""
    result = dict(base)
    for k, v in override.items():
        if k in result and isinstance(result[k], dict) and isinstance(v, dict):
            result[k] = _deep_merge(result[k], v)
        else:
            result[k] = v
    return result


def _resolve_base(config):
    """Recursively resolve `base_config` references inside a loaded config dict."""
    if "base_config" in config:
        base_path = config.pop("base_config")
        with open(base_path, "r") as f:
            base = yaml.safe_load(f) or {}
        base = _resolve_base(base)
        config = _deep_merge(base, config)
    return config


def load_config(config_path):
    """Load a YAML config file, resolving any `base_config` chain into a single namespace."""
    with open(config_path, "r") as f:
        config = yaml.safe_load(f) or {}
    config = _resolve_base(config)
    return _dict_to_namespace(config)


def main():
    parser = argparse.ArgumentParser(description="Entry point for training / evaluation")
    parser.add_argument("--config", type=str, required=True, help="Path to YAML config file")
    parser.add_argument("--to_train", action="store_true", help="Run training (otherwise run eval)")
    parser.add_argument(
        "--eval_split",
        type=str,
        default="test",
        choices=["valid", "test", "both"],
        help="Which split(s) to evaluate on (only used when --to_train is not set)",
    )
    args = parser.parse_args()

    cfg = load_config(args.config)

    np.random.seed(cfg.training.seed)
    torch.manual_seed(cfg.training.seed)
    torch.cuda.manual_seed(cfg.training.seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    if not os.path.exists(cfg.training.exp_name):
        os.makedirs(cfg.training.exp_name)
        print(f"Folder '{cfg.training.exp_name}' created.")
    else:
        print(f"Folder '{cfg.training.exp_name}' already exists.")

    model = Model(cfg)

    total_params = sum(p.numel() for p in model.parameters())
    print(f"Model built: {total_params:,} parameters ({total_params/1e6:.2f}M)")

    print(f"to_train: {args.to_train}")
    if args.to_train:
        train_dataloader = load_train(cfg)
        train(cfg, model, train_dataloader)
    else:
        splits = ["valid", "test"] if args.eval_split == "both" else [args.eval_split]
        with torch.no_grad():
            for split in splits:
                eval(cfg, model, split=split)


if __name__ == "__main__":
    main()