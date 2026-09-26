# model.py
import torch
import torch.nn as nn



class Model(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.mlp = nn.Linear(1,1)


    def forward(self, x, x_out=None, in_mask=None):
        return x