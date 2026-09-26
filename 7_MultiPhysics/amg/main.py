import os
import argparse
import numpy as np
import torch

from utils import load_config, load_train
from model import Model
from train import train
from eval import eval


def main():
    parser = argparse.ArgumentParser(description="Entry point for training / evaluation")
    parser.add_argument("--config", type=str, required=True, help="Path to YAML config file")
    parser.add_argument("--to_train", action="store_true", help="Run training (otherwise run eval)")
    parser.add_argument(
        "--eval_split",
        type=str,
        default="test",
        choices=["valid", "test", "both"],
        help="valid = decoupled held-out set, test = coupled fixed-point evaluation",
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
    # print(f"Model built ({cfg.data.field}): {total_params:,} parameters ({total_params/1e6:.2f}M)")

    print(f"to_train: {args.to_train}")
    if args.to_train:
        train_dataloader = load_train(cfg)
        train(cfg, model, train_dataloader)
    else:
        splits = ["valid", "test"] if args.eval_split == "both" else [args.eval_split]
        with torch.no_grad():
            for split in splits:
                eval(cfg, model, split=split, model_builder=Model)


if __name__ == "__main__":
    main()