import math
import torch

def Attention_Vanilla(q, k, v):
    score = torch.softmax(torch.einsum("bhic,bhjc->bhij", q, k) / math.sqrt(k.shape[-1]), dim=-1)
    r = torch.einsum("bhij,bhjc->bhic", score, v)
    return r


def Attention_Linear_GNOT(q, k, v):
    q = q.softmax(dim=-1)
    k = k.softmax(dim=-1)
    k_sum = k.sum(dim=-2, keepdim=True)
    inv = 1. / (q * k_sum).sum(dim=-1, keepdim=True)
    r = q + (q @ (k.transpose(-2, -1) @ v)) * inv
    return r

ACTIVATION = {"Sigmoid": torch.nn.Sigmoid(),
              "Tanh": torch.nn.Tanh(),
              "ReLU": torch.nn.ReLU(),
              "LeakyReLU": torch.nn.LeakyReLU(0.1),
              "ELU": torch.nn.ELU(),
              "GELU": torch.nn.GELU()
              }

ATTENTION = {"Attention_Vanilla": Attention_Vanilla,
             "Attention_Linear_GNOT": Attention_Linear_GNOT
            }


class MLP(torch.nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, n_layer, act):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.n_layer = n_layer
        self.act = act
        self.input = torch.nn.Linear(self.input_dim, self.hidden_dim)
        self.hidden = torch.nn.ModuleList([torch.nn.Linear(self.hidden_dim, self.hidden_dim) for _ in range(self.n_layer)])
        self.output = torch.nn.Linear(self.hidden_dim, self.output_dim)
        
    def forward(self, x):
        r = self.act(self.input(x))
        for i in range(0, self.n_layer):
            r = r + self.act(self.hidden[i](r))
        r = self.output(r)
        return r
  

class Model(torch.nn.Module):
    class SelfAttention(torch.nn.Module):
        def __init__(self, n_mode, n_dim, n_head, attn):
            super().__init__()
            self.n_mode = n_mode
            self.n_dim = n_dim
            self.n_head = n_head
            self.Wq = torch.nn.Linear(self.n_dim, self.n_dim, bias=False)
            self.Wk = torch.nn.Linear(self.n_dim, self.n_dim, bias=False)
            self.Wv = torch.nn.Linear(self.n_dim, self.n_dim, bias=False)
            self.attn = attn
            self.proj = torch.nn.Linear(self.n_dim, self.n_dim, bias=False)
        
        def forward(self, x):
            B, N, D = x.size()
            q = self.Wq(x).view(B, N, self.n_head, D // self.n_head).permute(0, 2, 1, 3)
            k = self.Wk(x).view(B, N, self.n_head, D // self.n_head).permute(0, 2, 1, 3)
            v = self.Wv(x).view(B, N, self.n_head, D // self.n_head).permute(0, 2, 1, 3)
            r = self.attn(q, k, v).permute(0, 2, 1, 3).contiguous().view(B, N, D)
            r = self.proj(r)
            return r
    
    class AttentionBlock(torch.nn.Module):
        def __init__(self, n_mode, n_dim, n_head, attn, act):
            super().__init__()
            self.n_mode = n_mode
            self.n_dim = n_dim
            self.n_head = n_head
            self.attn = attn
            self.act = act
            
            self.self_attn = Model.SelfAttention(self.n_mode, self.n_dim, self.n_head, self.attn)
            self.ln1 = torch.nn.LayerNorm(self.n_dim)
            self.ln2 = torch.nn.LayerNorm(self.n_dim)
            self.drop = torch.nn.Dropout(0.0)
            
            self.mlp = torch.nn.Sequential(
                torch.nn.Linear(self.n_dim, self.n_dim*2),
                self.act,
                torch.nn.Linear(self.n_dim*2, self.n_dim),
            )

        def forward(self, y):   
            y = y + self.drop(self.self_attn(self.ln1(y)))
            y = y + self.mlp(self.ln2(y))
            return y


    def __init__(self, cfg):
        super().__init__()
        self.n_block = cfg.model.n_block
        self.n_mode = cfg.model.n_mode
        self.n_dim = cfg.model.n_dim
        self.n_head = cfg.model.n_head
        self.n_layer = cfg.model.n_layer

        self.attn = ATTENTION[cfg.model.attn]
        self.act = ACTIVATION[cfg.model.act]
        
        self.y1_dim = cfg.model.y1_dim
        self.y2_dim = cfg.model.y2_dim
        self.x_dim = cfg.model.x_dim    # dimension of output query coordinates
        
        self.in_mlp = MLP(self.y1_dim, self.n_dim, self.n_dim, self.n_layer, self.act)
        self.out_mlp = MLP(self.n_dim, self.n_dim, self.y2_dim, self.n_layer, self.act)
        self.Wm = torch.nn.Linear(self.n_dim, self.n_mode, bias=False)
        
        # query branch: embeds output coordinates and maps them to mode weights,
        # decoupling the output point set from the input point set
        self.query_mlp = MLP(self.x_dim, self.n_dim, self.n_dim, self.n_layer, self.act)
        self.Wm_out = torch.nn.Linear(self.n_dim, self.n_mode, bias=False)
        
        self.attn_blocks = torch.nn.Sequential(*[Model.AttentionBlock(self.n_mode, self.n_dim, self.n_head, self.attn, self.act) for _ in range(0, self.n_block)])

    def _init_weights(self, module):
        if isinstance(module, (torch.nn.Linear, torch.nn.Embedding)):
            module.weight.data.normal_(mean=0.0, std=0.0002)
            if isinstance(module, torch.nn.Linear) and module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, torch.nn.LayerNorm):
            module.weight.data.fill_(1.0)
            module.bias.data.zero_()

    def forward(self, y, x_out=None, in_mask=None):
        """
        y:       (B, N, y1_dim)  input points (values, possibly with coords appended)
        x_out:   (B, M, x_dim)   optional output query coordinates; if None, the
                                 output is evaluated at the input points (M = N),
                                 reproducing the original behavior exactly.
        in_mask: (B, N)          optional 1/0 mask marking valid input points (needed
                                 when batching samples with different N). Padded
                                 points are excluded from the latent encoding.
        returns: (B, M, y2_dim) if x_out is given, else (B, N, y2_dim)
        """
        # encode: aggregate N input points into n_mode latent tokens
        y = self.in_mlp(y)
        weight_in = torch.softmax(self.Wm(y), dim=-1)            # (B, N, n_mode)
        if in_mask is not None:
            weight_in = weight_in * in_mask.unsqueeze(-1)
        denom = torch.sum(weight_in, dim=-2).unsqueeze(-1)       # (B, n_mode, 1)
        z = torch.einsum("bij,bic->bjc", weight_in, y) / (denom + 1e-6)
        
        # process in latent space
        for block in self.attn_blocks:
            z = block(z)

        # decode: project latent modes onto the output point set
        if x_out is None:
            weight_out = weight_in                               # (B, N, n_mode)
        else:
            q = self.query_mlp(x_out)                            # (B, M, n_dim)
            weight_out = torch.softmax(self.Wm_out(q), dim=-1)   # (B, M, n_mode)
        
        r = torch.einsum("bij,bjc->bic", weight_out, z)
        r = self.out_mlp(r)

        return r