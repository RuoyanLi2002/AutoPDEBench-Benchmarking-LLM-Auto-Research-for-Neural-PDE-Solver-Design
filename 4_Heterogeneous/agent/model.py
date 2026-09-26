# model.py
import torch
import torch.nn as nn


class Model(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.mlp = nn.Linear(1,1)

    def forward(self, x, attr):
        return x