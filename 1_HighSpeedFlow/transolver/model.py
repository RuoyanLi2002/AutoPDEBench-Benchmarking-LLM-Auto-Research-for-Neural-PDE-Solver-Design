import numpy as np
import torch
import torch.nn as nn

from network import Physics_Attention_Irregular_Mesh

ACTIVATIONS = {
    "relu": nn.ReLU,
    "gelu": nn.GELU,
    "tanh": nn.Tanh,
    "sigmoid": nn.Sigmoid,
    "leaky_relu": nn.LeakyReLU,
    "elu": nn.ELU,
    "silu": nn.SiLU,
    "swish": nn.SiLU,
}


def build_mlp(in_size, hidden_size, out_size, lay_norm=True, activation="relu"):
    act = ACTIVATIONS[activation]
    module = nn.Sequential(nn.Linear(in_size, hidden_size),
                           act(),
                           nn.Linear(hidden_size, hidden_size),
                           act(),
                           nn.Linear(hidden_size, hidden_size),
                           act(),
                           nn.Linear(hidden_size, out_size))
    if lay_norm:
        return nn.Sequential(module, nn.LayerNorm(normalized_shape=out_size))
    return module


class Transolver_block(nn.Module):
    """Transformer encoder block."""
    def __init__(self, num_heads, hidden_dim, dropout, mlp_ratio=4.0, last_layer=False, out_dim=1, slice_num=32):
        super().__init__()
        self.last_layer = last_layer
        self.ln_1 = nn.LayerNorm(hidden_dim)
        self.Attn = Physics_Attention_Irregular_Mesh(hidden_dim, heads=num_heads, dim_head=hidden_dim // num_heads,
                                                     dropout=dropout, slice_num=slice_num)
        self.ln_2 = nn.LayerNorm(hidden_dim)
        self.mlp = build_mlp(hidden_dim, int(hidden_dim * mlp_ratio), hidden_dim, lay_norm=False)

        if self.last_layer:
            self.ln_3 = nn.LayerNorm(hidden_dim)
            self.mlp2 = nn.Linear(hidden_dim, out_dim)

    def forward(self, fx):
        fx = self.Attn(self.ln_1(fx)) + fx
        fx = self.mlp(self.ln_2(fx)) + fx
        if self.last_layer:
            return self.mlp2(self.ln_3(fx))
        else:
            return fx


class Model(nn.Module):
    def __init__(self, cfg):
        super(Model, self).__init__()
        self._input_dim = cfg.model.input_dim
        self._latent_dim = cfg.model.latent_dim
        self._num_head = cfg.model.num_head
        self._dropout = cfg.model.dropout
        self._mlp_ratio = cfg.model.mlp_ratio
        self._slice_num = cfg.model.slice_num
        self._n_layers = cfg.model.n_layers
        self._output_dim = cfg.model.output_dim
        self._space_dim = cfg.model.space_dim

        self.preprocess = build_mlp(self._input_dim + self._space_dim, self._latent_dim, self._latent_dim, lay_norm=True)

        self.blocks = nn.ModuleList([Transolver_block(num_heads=self._num_head, hidden_dim=self._latent_dim,
                                                      dropout=self._dropout,
                                                      mlp_ratio=self._mlp_ratio,
                                                      out_dim=self._latent_dim,
                                                      slice_num=self._slice_num,
                                                      last_layer=(_ == self._n_layers - 1))
                                     for _ in range(self._n_layers)])
        self.decoder = build_mlp(self._latent_dim, self._latent_dim, self._output_dim, lay_norm=False)

    def forward(self, x, tx, ty):
        B, C, H, W = x.shape
        fx = x.permute(0, 2, 3, 1).reshape(B, H * W, C)

        if self._space_dim > 0:
            gridx = torch.linspace(0, 1, W, device=x.device)
            gridy = torch.linspace(0, 1, H, device=x.device)
            gy, gx = torch.meshgrid(gridy, gridx, indexing='ij')
            grid = torch.stack([gx, gy], dim=-1).reshape(1, H * W, 2).repeat(B, 1, 1)
            fx = torch.cat([fx, grid], dim=-1)

        fx = self.preprocess(fx)
        for block in self.blocks:
            fx = block(fx)

        fx = self.decoder(fx)
        fx = fx.reshape(B, H, W, C).permute(0, 3, 1, 2)

        return fx