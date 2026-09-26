# model.py
import torch
import torch.nn as nn
import torch.nn.functional as F


class Model(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.mlp = nn.Linear(1,1)

    def forward(self, x, mask):
        return x