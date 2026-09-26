import math
import numpy as np
import torch
import torch.nn as nn
from einops import rearrange
from torch.nn import functional as F
from torch.nn import GELU, ReLU, Tanh, Sigmoid

class MultipleTensors():
    def __init__(self, x):
        self.x = x
    def to(self, device):
        self.x = [x_.to(device) for x_ in self.x]
        return self
    def __len__(self):
        return len(self.x)
    def __getitem__(self, item):
        return self.x[item]


ACTIVATION = {'gelu': nn.GELU(), 'tanh': nn.Tanh(), 'sigmoid': nn.Sigmoid(),
              'relu': nn.ReLU(), 'leaky_relu': nn.LeakyReLU(0.1),
              'softplus': nn.Softplus(), 'ELU': nn.ELU()}

class MLP(nn.Module):
    def __init__(self, n_input, n_hidden, n_output, n_layers=1, act='gelu'):
        super(MLP, self).__init__()
        if act in ACTIVATION.keys():
            self.act = ACTIVATION[act]
        else:
            raise NotImplementedError
        self.n_input = n_input
        self.n_hidden = n_hidden
        self.n_output = n_output
        self.n_layers = n_layers
        self.linear_pre = nn.Linear(n_input, n_hidden)
        self.linear_post = nn.Linear(n_hidden, n_output)
        self.linears = nn.ModuleList([nn.Linear(n_hidden, n_hidden) for _ in range(n_layers)])

    def forward(self, x):
        x = self.act(self.linear_pre(x))
        for i in range(self.n_layers):
            x = self.act(self.linears[i](x)) + x
        x = self.linear_post(x)
        return x


class MoEGPTConfig():
    """ base GPT config, params common to all GPT versions """
    def __init__(self, attn_type='linear', embd_pdrop=0.0, resid_pdrop=0.0, attn_pdrop=0.0,
                 n_embd=128, n_head=1, n_layer=3, block_size=128, n_inner=4, act='gelu',
                 n_experts=2, space_dim=1, branch_sizes=None, n_inputs=1):
        self.attn_type = attn_type
        self.embd_pdrop = embd_pdrop
        self.resid_pdrop = resid_pdrop
        self.attn_pdrop = attn_pdrop
        self.n_embd = n_embd
        self.n_head = n_head
        self.n_layer = n_layer
        self.block_size = block_size
        self.n_inner = n_inner * self.n_embd
        self.act = act
        self.n_experts = n_experts
        self.space_dim = space_dim
        self.branch_sizes = branch_sizes
        self.n_inputs = n_inputs


class LinearAttention(nn.Module):
    def __init__(self, config):
        super(LinearAttention, self).__init__()
        assert config.n_embd % config.n_head == 0
        self.key = nn.Linear(config.n_embd, config.n_embd)
        self.query = nn.Linear(config.n_embd, config.n_embd)
        self.value = nn.Linear(config.n_embd, config.n_embd)
        self.attn_drop = nn.Dropout(config.attn_pdrop)
        self.proj = nn.Linear(config.n_embd, config.n_embd)
        self.n_head = config.n_head
        self.attn_type = 'l1'

    def forward(self, x, y=None, layer_past=None):
        y = x if y is None else y
        B, T1, C = x.size()
        _, T2, _ = y.size()
        q = self.query(x).view(B, T1, self.n_head, C // self.n_head).transpose(1, 2)
        k = self.key(y).view(B, T2, self.n_head, C // self.n_head).transpose(1, 2)
        v = self.value(y).view(B, T2, self.n_head, C // self.n_head).transpose(1, 2)
        if self.attn_type == 'l1':
            q = q.softmax(dim=-1)
            k = k.softmax(dim=-1)
            k_cumsum = k.sum(dim=-2, keepdim=True)
            D_inv = 1. / (q * k_cumsum).sum(dim=-1, keepdim=True)
        elif self.attn_type == "galerkin":
            q = q.softmax(dim=-1)
            k = k.softmax(dim=-1)
            D_inv = 1. / T2
        elif self.attn_type == "l2":
            q = q / q.norm(dim=-1, keepdim=True, p=1)
            k = k / k.norm(dim=-1, keepdim=True, p=1)
            k_cumsum = k.sum(dim=-2, keepdim=True)
            D_inv = 1. / (q * k_cumsum).abs().sum(dim=-1, keepdim=True)
        else:
            raise NotImplementedError
        context = k.transpose(-2, -1) @ v
        y = self.attn_drop((q @ context) * D_inv + q)
        y = rearrange(y, 'b h n d -> b n (h d)')
        y = self.proj(y)
        return y


class LinearCrossAttention(nn.Module):
    def __init__(self, config):
        super(LinearCrossAttention, self).__init__()
        assert config.n_embd % config.n_head == 0
        self.query = nn.Linear(config.n_embd, config.n_embd)
        self.keys = nn.ModuleList([nn.Linear(config.n_embd, config.n_embd) for _ in range(config.n_inputs)])
        self.values = nn.ModuleList([nn.Linear(config.n_embd, config.n_embd) for _ in range(config.n_inputs)])
        self.attn_drop = nn.Dropout(config.attn_pdrop)
        self.proj = nn.Linear(config.n_embd, config.n_embd)
        self.n_head = config.n_head
        self.n_inputs = config.n_inputs
        self.attn_type = 'l1'

    def forward(self, x, y=None, layer_past=None):
        y = x if y is None else y
        B, T1, C = x.size()
        q = self.query(x).view(B, T1, self.n_head, C // self.n_head).transpose(1, 2)
        q = q.softmax(dim=-1)
        out = q
        for i in range(self.n_inputs):
            _, T2, _ = y[i].size()
            k = self.keys[i](y[i]).view(B, T2, self.n_head, C // self.n_head).transpose(1, 2)
            v = self.values[i](y[i]).view(B, T2, self.n_head, C // self.n_head).transpose(1, 2)
            k = k.softmax(dim=-1)
            k_cumsum = k.sum(dim=-2, keepdim=True)
            D_inv = 1. / (q * k_cumsum).sum(dim=-1, keepdim=True)
            out = out + 1 * (q @ (k.transpose(-2, -1) @ v)) * D_inv
        out = rearrange(out, 'b h n d -> b n (h d)')
        out = self.proj(out)
        return out


def horizontal_fourier_embedding(X, n=3):
    freqs = 2**torch.linspace(-n, n, 2*n+1).to(X.device)
    freqs = freqs[None, None, None, ...]
    X_ = X.unsqueeze(-1).repeat([1, 1, 1, 2*n+1])
    X_cos = torch.cos(freqs * X_)
    X_sin = torch.sin(freqs * X_)
    X = torch.cat([X.unsqueeze(-1), X_cos, X_sin], dim=-1).view(X.shape[0], X.shape[1], -1)
    return X


class MIOECrossAttentionBlock(nn.Module):
    def __init__(self, config):
        super(MIOECrossAttentionBlock, self).__init__()
        self.ln1 = nn.LayerNorm(config.n_embd)
        self.ln2_branch = nn.ModuleList([nn.LayerNorm(config.n_embd) for _ in range(config.n_inputs)])
        self.ln3 = nn.LayerNorm(config.n_embd)
        self.ln4 = nn.LayerNorm(config.n_embd)
        self.ln5 = nn.LayerNorm(config.n_embd)
        if config.attn_type == 'linear':
            self.selfattn = LinearAttention(config)
            self.crossattn = LinearCrossAttention(config)
        else:
            raise NotImplementedError
        if config.act == 'gelu':
            self.act = GELU
        elif config.act == "tanh":
            self.act = Tanh
        elif config.act == 'relu':
            self.act = ReLU
        elif config.act == 'sigmoid':
            self.act = Sigmoid
        self.resid_drop1 = nn.Dropout(config.resid_pdrop)
        self.resid_drop2 = nn.Dropout(config.resid_pdrop)
        self.n_experts = config.n_experts
        self.n_inputs = config.n_inputs
        self.moe_mlp1 = nn.ModuleList([nn.Sequential(
            nn.Linear(config.n_embd, config.n_inner),
            self.act(),
            nn.Linear(config.n_inner, config.n_embd),
        ) for _ in range(self.n_experts)])
        self.moe_mlp2 = nn.ModuleList([nn.Sequential(
            nn.Linear(config.n_embd, config.n_inner),
            self.act(),
            nn.Linear(config.n_inner, config.n_embd),
        ) for _ in range(self.n_experts)])
        self.gatenet = nn.Sequential(
            nn.Linear(config.space_dim, config.n_inner),
            self.act(),
            nn.Linear(config.n_inner, config.n_inner),
            self.act(),
            nn.Linear(config.n_inner, self.n_experts)
        )

    def ln_branchs(self, y):
        return MultipleTensors([self.ln2_branch[i](y[i]) for i in range(self.n_inputs)])

    def forward(self, x, y, pos):
        gate_score = F.softmax(self.gatenet(pos), dim=-1).unsqueeze(2)  # B, T1, 1, m
        x = x + self.resid_drop1(self.crossattn(self.ln1(x), self.ln_branchs(y)))
        x_moe1 = torch.stack([self.moe_mlp1[i](x) for i in range(self.n_experts)], dim=-1)
        x_moe1 = (gate_score * x_moe1).sum(dim=-1, keepdim=False)
        x = x + self.ln3(x_moe1)
        x = x + self.resid_drop2(self.selfattn(self.ln4(x)))
        x_moe2 = torch.stack([self.moe_mlp2[i](x) for i in range(self.n_experts)], dim=-1)
        x_moe2 = (gate_score * x_moe2).sum(dim=-1, keepdim=False)
        x = x + self.ln5(x_moe2)
        return x


class Model(nn.Module):
    def __init__(self, cfg):
        super(Model, self).__init__()
        branch_sizes = None
        trunk_size = cfg.model.trunk_size
        horiz_fourier_dim = cfg.model.horiz_fourier_dim
        output_size = cfg.model.output_size
        space_dim = cfg.model.space_dim
        attn_type = cfg.model.attn_type
        ffn_dropout = cfg.model.ffn_dropout
        attn_dropout = cfg.model.attn_dropout
        n_hidden = cfg.model.n_hidden
        n_head = cfg.model.n_head
        n_layers = cfg.model.n_layers
        act = cfg.model.act
        n_experts = cfg.model.n_experts
        n_inner = cfg.model.n_inner
        mlp_layers = cfg.model.mlp_layers


        if branch_sizes is None:
            branch_sizes = [trunk_size]
        self.horiz_fourier_dim = horiz_fourier_dim
        self.trunk_size = trunk_size * (4*horiz_fourier_dim + 3) if horiz_fourier_dim > 0 else trunk_size
        self.branch_sizes = [bsize * (4*horiz_fourier_dim + 3) for bsize in branch_sizes] if horiz_fourier_dim > 0 else branch_sizes
        self.n_inputs = len(self.branch_sizes)
        self.output_size = output_size
        self.space_dim = space_dim
        self.gpt_config = MoEGPTConfig(attn_type=attn_type, embd_pdrop=ffn_dropout, resid_pdrop=ffn_dropout,
                                       attn_pdrop=attn_dropout, n_embd=n_hidden, n_head=n_head, n_layer=n_layers,
                                       block_size=128, act=act, n_experts=n_experts, space_dim=space_dim,
                                       branch_sizes=branch_sizes, n_inputs=len(branch_sizes), n_inner=n_inner)
        self.trunk_mlp = MLP(self.trunk_size, n_hidden, n_hidden, n_layers=mlp_layers, act=act)
        self.branch_mlps = nn.ModuleList([MLP(bsize, n_hidden, n_hidden, n_layers=mlp_layers, act=act) for bsize in self.branch_sizes])
        self.blocks = nn.Sequential(*[MIOECrossAttentionBlock(self.gpt_config) for _ in range(self.gpt_config.n_layer)])
        self.out_mlp = MLP(n_hidden, n_hidden, output_size, n_layers=mlp_layers)

    def forward(self, x):
        B, N, C = x.shape
        pos = self.get_coord(x)  # (B, N, d)
        inputs = MultipleTensors([x])
        x = self.trunk_mlp(x)
        z = MultipleTensors([self.branch_mlps[i](inputs[i]) for i in range(self.n_inputs)])
        for block in self.blocks:
            x = block(x, z, pos)
        x = self.out_mlp(x)
        return x

    def get_coord(self, x, side=64):
        B, N, C = x.shape
        d = round(math.log(N) / math.log(side))   # 1, 2, or 3
        assert side ** d == N, f"N={N} is not {side}^d"

        axis = torch.linspace(0, 1, steps=side, device=x.device)
        grids = torch.meshgrid(*([axis] * d), indexing='ij')
        coords = torch.stack(grids, dim=-1).reshape(N, d)   # (N, d)

        padded = torch.zeros(N, 3, device=x.device)
        padded[:, :d] = coords                              # cols d..2 stay 0
        return padded.unsqueeze(0).expand(B, -1, -1)        # (B, N, 3)