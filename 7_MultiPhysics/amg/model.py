import einops
import torch
import torch.nn as nn
from torch_geometric.nn import GATv2Conv
from torch_geometric.utils import unbatch

from torch_geometric.nn.pool import fps
from torch_geometric.nn.unpool import knn_interpolate
from torch_geometric.nn import knn_graph


# =============================================================================
# Graph-model components (behavior unchanged from the provided graph model)
# =============================================================================
def compute_feature_map(y, pos, ratio=0.25, batch=None):
    index_down = fps(pos, ratio=ratio, batch=batch)
    pos_down = pos[index_down]
    y_down = y[index_down]
    y_up = knn_interpolate(x=y_down, pos_x=pos_down, pos_y=pos, batch_x=batch[index_down] if batch is not None else None, batch_y=batch)

    fm = torch.abs(y - y_up)
    fm = torch.sum(fm, dim=1)
    return fm, index_down


def local_sample(x, pos, sample_nodes=512, k=8, ratio=0.25, batch=None, cosine=False, use_pos=False):
    fm, _ = compute_feature_map(x, pos, ratio, batch)

    if batch is not None:
        batch_size = batch.max().item() + 1
        sampled_indices = None
        sampled_batch = None

        for b in range(batch_size):
            mask = (batch == b)
            fm_batch = fm[mask]
            _, indices_topk_local = torch.topk(fm_batch, k=min(sample_nodes, fm_batch.size(0)), largest=True)

            if sampled_indices is None:
                sampled_indices = torch.nonzero(mask).to(x.device).flatten()[indices_topk_local].to(indices_topk_local.device)
                sampled_batch = torch.tensor([b] * indices_topk_local.size(0)).to(x.device)
            else:
                sampled_indices = torch.cat([sampled_indices, torch.nonzero(mask).flatten()[indices_topk_local]], dim=0)
                sampled_batch = torch.cat([sampled_batch, torch.tensor([b] * indices_topk_local.size(0)).to(x.device)], dim=0)
        if use_pos:
            sampled_edge_index = knn_graph(pos[sampled_indices], k=k, loop=False, batch=sampled_batch, cosine=cosine)
        else:
            sampled_edge_index = knn_graph(x[sampled_indices], k=k, loop=False, batch=sampled_batch, cosine=cosine)
        edge_index = torch.cat([
            sampled_indices[sampled_edge_index[0]].view(1, -1),
            sampled_indices[sampled_edge_index[1]].view(1, -1)], dim=0)
    else:
        _, indices_topk_local = torch.topk(fm, k=min(sample_nodes, fm.size(0)), largest=True)
        sampled_indices = torch.nonzero(fm).flatten()[indices_topk_local]
        if use_pos:
            sampled_edge_index = knn_graph(pos[sampled_indices], k=k, loop=False, cosine=cosine)
        else:
            sampled_edge_index = knn_graph(x[sampled_indices], k=k, loop=False, cosine=cosine)
        edge_index = torch.cat([
            sampled_indices[sampled_edge_index[0]].view(1, -1),
            sampled_indices[sampled_edge_index[1]].view(1, -1)], dim=0)
        sampled_batch = None

    return edge_index, sampled_edge_index, sampled_indices, sampled_batch


def global_sample(x, pos, ratio=0.25, k=8, batch=None, cosine=False, use_pos=False):
    sampled_indices = fps(pos, ratio=ratio, batch=batch)
    sampled_batch = batch[sampled_indices] if batch is not None else None

    if use_pos:
        sampled_edge_index = knn_graph(pos[sampled_indices], k=k, loop=False, batch=sampled_batch, cosine=cosine)
    else:
        sampled_edge_index = knn_graph(x[sampled_indices], k=k, loop=False, batch=sampled_batch, cosine=cosine)
    edge_index = torch.cat([
        sampled_indices[sampled_edge_index[0]].view(1, -1),
        sampled_indices[sampled_edge_index[1]].view(1, -1)], dim=0)

    return edge_index, sampled_edge_index, sampled_indices, sampled_batch


class MLP(nn.Module):
    def __init__(self, input_size, hidden_size, output_size,
                 num_layers=1, act='gelu', batch_norm=False, **kwargs):
        super(MLP, self).__init__()
        self.num_layers = num_layers

        layers = [nn.Linear(input_size, hidden_size)]
        activation = {'gelu': nn.GELU(), 'relu': nn.ReLU(), 'tanh': nn.Tanh(),
                      'sigmoid': nn.Sigmoid(), 'leaky_relu': nn.LeakyReLU(),
                      'elu': nn.ELU(), 'softplus': nn.Softplus()}.get(act, None)
        if activation is None:
            raise NotImplementedError("Activation function not supported.")

        for _ in range(num_layers):
            layers.append(activation)
            if batch_norm:
                layers.append(nn.BatchNorm1d(hidden_size))
            layers.append(nn.Linear(hidden_size, hidden_size))

        layers.append(activation)
        layers.append(nn.Linear(hidden_size, output_size))
        self.processor = nn.Sequential(*layers)

    def forward(self, x):
        return self.processor(x)


class AttentionGraphBlock(nn.Module):
    def __init__(self, feature_width, input_features=1, output_features=1, num_heads=8,
                 batch_norm=True, act='relu', **kwargs):
        super(AttentionGraphBlock, self).__init__()
        self.graph_conv = GATv2Conv(input_features, feature_width, heads=num_heads, concat=False, negative_slope=0.2, dropout=0.0)
        self.ln_1 = nn.LayerNorm(feature_width)
        self.ffn = MLP(feature_width, feature_width, output_features, num_layers=1, batch_norm=batch_norm, act=act, **kwargs)
        self.ln_2 = nn.LayerNorm(input_features)

    def forward(self, x, edge_index):
        shortcut = x
        x = self.graph_conv(x, edge_index)
        x = self.ln_1(x + shortcut)
        shortcut = x
        x = self.ffn(x)
        x = self.ln_2(x + shortcut)
        return x


class PhysicsGraphBlock(nn.Module):
    def __init__(self, feature_width, num_heads=8, num_phys=32, dropout=0., **kwargs):
        super(PhysicsGraphBlock, self).__init__()
        hidden_width = feature_width * num_heads
        self.feature_width = feature_width
        self.num_heads = num_heads
        self.num_phys = num_phys
        self.softmax = nn.Softmax(dim=-1)
        self.scale = feature_width ** -0.5
        self.dropout = nn.Dropout(dropout)
        self.temperature = nn.Parameter(torch.ones([num_heads, 1, 1]) * 0.5)

        self.l_in = nn.Linear(feature_width, hidden_width)
        self.l_token = nn.Linear(feature_width, hidden_width)
        self.l_phy = nn.Linear(feature_width, num_phys)
        for l in [self.l_phy]:
            torch.nn.init.orthogonal_(l.weight)

        self.q = nn.Linear(feature_width, feature_width, bias=False)
        self.k = nn.Linear(feature_width, feature_width, bias=False)
        self.v = nn.Linear(feature_width, feature_width, bias=False)

        self.l_out = nn.Linear(hidden_width, feature_width)

        self.ln_1 = nn.LayerNorm(feature_width)
        self.ffn = MLP(feature_width, feature_width, feature_width, num_layers=1, act='relu')
        self.ln_2 = nn.LayerNorm(feature_width)

    def single_in(self, x):
        phy_x = einops.rearrange(self.l_in(x), 'n (h c) -> h n c', h=self.num_heads)
        phy_weights = self.softmax(self.l_phy(phy_x) / self.temperature) # H N M

        phy_norm = phy_weights.sum(1).unsqueeze(-1) # H M 1
        phy_norm = (phy_norm + 1e-5).repeat(1, 1, self.feature_width) # H M C

        phy_token = einops.rearrange(self.l_token(x), 'n (h c) -> h n c', h=self.num_heads)
        phy_token = torch.einsum("hnc,hnm->hmc", phy_token, phy_weights)
        phy_token = phy_token / phy_norm

        return phy_token.unsqueeze(0), phy_weights

    def single_out(self, phy_token, phy_weights):
        out = torch.einsum("hmc,hnm->hnc", phy_token, phy_weights)
        out = einops.rearrange(out, 'h n c -> n (h c)')
        out = self.l_out(out)

        return out

    def forward(self, x, batch):
        shortcut = x
        x_batch = unbatch(x, batch)
        B = len(x_batch)
        phy_token_batch = []
        phy_weights_batch = []

        for x in x_batch:
            phy_token, phy_weights = self.single_in(x)
            phy_token_batch.append(phy_token)
            phy_weights_batch.append(phy_weights)

        phy_token = torch.cat(phy_token_batch, dim=0)
        phy_q = self.q(phy_token)
        phy_k = self.k(phy_token)
        phy_v = self.v(phy_token)
        dots = torch.matmul(phy_q, phy_k.transpose(-1, -2)) * self.scale
        attn = self.softmax(dots)
        attn = self.dropout(attn)
        phy_y = torch.matmul(attn, phy_v)

        out_batch = []
        for i in range(B):
            out = self.single_out(phy_y[i], phy_weights_batch[i])
            out_batch.append(out)

        x = torch.cat(out_batch, dim=0)
        x = self.ln_1(x + shortcut)
        shortcut = x
        out = self.ln_2(self.ffn(x) + shortcut)

        return out


class MultiscaleGraphBlock(nn.Module):
    def __init__(self, feature_width, input_features=1, output_features=1,
                 batch_norm=True, act='relu', num_phys=32, num_heads=8,
                 local_nodes=512, local_ratio=0.25, local_k=4, local_cos=False, local_pos=True,
                 global_ratio=0.25, global_k=8, global_cos=True, global_pos=False,
                 **kwargs):
        super(MultiscaleGraphBlock, self).__init__()
        self.local_nodes = local_nodes
        self.local_ratio = local_ratio
        self.local_k = local_k
        self.local_cos = local_cos
        self.local_pos = local_pos

        self.global_ratio = global_ratio
        self.global_k = global_k
        self.global_cos = global_cos
        self.global_pos = global_pos

        self.num_phys = num_phys
        self.phy_aggr = PhysicsGraphBlock(feature_width, num_heads=num_heads, num_phys=num_phys, dropout=0.0)
        self.global_aggr = AttentionGraphBlock(feature_width, input_features=feature_width, output_features=feature_width, num_heads=num_heads, batch_norm=batch_norm, act=act, **kwargs)
        self.local_aggr = AttentionGraphBlock(feature_width, input_features=feature_width, output_features=feature_width, num_heads=num_heads, batch_norm=batch_norm, act=act, **kwargs)

        self.ln_1 = nn.LayerNorm(input_features)
        self.ln_2 = nn.LayerNorm(input_features)

        self.mlp = MLP(input_features, feature_width, output_features, num_layers=0, batch_norm=batch_norm, act=act, **kwargs)

    def forward(self, x, pos, batch=None):
        x_in = x
        x = self.phy_aggr(x, batch)

        local_edge_index, _, _, _ = local_sample(x, pos, sample_nodes=self.local_nodes, k=self.local_k,
                                                 ratio=self.local_ratio, batch=batch, cosine=self.local_cos,
                                                 use_pos=self.local_pos)
        x = self.local_aggr(x, local_edge_index)

        global_edge_index, _, _, _ = global_sample(x, pos, ratio=self.global_ratio, k=self.global_k,
                                                   batch=batch, cosine=self.global_cos,
                                                   use_pos=self.global_pos)

        x = self.global_aggr(x, global_edge_index)

        x = self.mlp(self.ln_2(x + x_in))

        return x


class GraphModel(nn.Module):
    """The provided multiscale physics-graph model.

    Consumes an object exposing `.x [N, C_in]`, `.pos [N, d]`, and `.batch [N]`;
    returns per-node predictions `[N, C_out]`. Edges are built internally
    (feature-map guided top-k + kNN for the local branch, FPS + kNN for the
    global branch).
    """
    def __init__(self, cfg, **kwargs):
        super(GraphModel, self).__init__()
        feature_width = cfg.model.feature_width
        num_layers = cfg.model.num_layers
        pos_dim = cfg.model.pos_dim
        input_features = cfg.model.input_features
        output_features = cfg.model.output_features
        batch_norm = cfg.model.batch_norm
        act = cfg.model.act
        local_nodes = cfg.model.local_nodes
        local_ratio = cfg.model.local_ratio
        local_k = cfg.model.local_k
        local_cos = cfg.model.local_cos
        local_pos = cfg.model.local_pos
        global_ratio = cfg.model.global_ratio
        global_k = cfg.model.global_k
        global_cos = cfg.model.global_cos
        global_pos = cfg.model.global_pos
        num_phys = cfg.model.num_phys
        num_heads = cfg.model.num_heads

        self.num_layers = num_layers

        self.in_mlp = MLP(input_features + pos_dim, feature_width * 2, feature_width, num_layers=0, batch_norm=batch_norm, act=act, **kwargs)
        self.blocks = nn.ModuleList([
            MultiscaleGraphBlock(feature_width, input_features=feature_width, output_features=feature_width,
                                 batch_norm=batch_norm, act=act, local_nodes=local_nodes,
                                 local_ratio=local_ratio, local_cos=local_cos, local_pos=local_pos,
                                 local_k=local_k, global_ratio=global_ratio, global_k=global_k,
                                 global_cos=global_cos, global_pos=global_pos,
                                 num_phys=num_phys, num_heads=num_heads) for _ in range(num_layers)])
        self.ln = nn.LayerNorm(feature_width)
        self.out_mlp = MLP(feature_width, feature_width, output_features, num_layers=0, batch_norm=batch_norm, act=act, **kwargs)

    def forward(self, data):
        x = torch.cat([data.x, data.pos], dim=-1)
        x = self.in_mlp(x)
        x_in = x

        for i in range(self.num_layers):
            x = self.blocks[i](x, data.pos, batch=data.batch)

        x_out = self.out_mlp(self.ln(x + x_in))

        return x_out


# =============================================================================
# Adapter: dense grid tensor  <->  graph, so the graph model is a drop-in for
# the transformer pipeline. The pipeline passes x = [B, C_in, *spatial] and
# expects out = [B, C_out, *spatial]; the graph model works on flat nodes.
# =============================================================================
class _GraphBatch:
    """Minimal stand-in for a PyG Batch, carrying only what GraphModel reads."""
    __slots__ = ("x", "pos", "batch")

    def __init__(self, x, pos, batch):
        self.x = x
        self.pos = pos
        self.batch = batch


class Model(nn.Module):
    """Drop-in replacement for the original transformer surrogate.

    Interface preserved exactly:
      * constructed with `Model(cfg)`
      * `forward(x)` with x = [B, C_in, *spatial]
      * returns [B, C_out, *spatial]

    Each sample's spatial grid is flattened (row-major) to graph nodes; the
    multiscale physics-graph model runs over the whole batched node set; the
    per-node predictions are folded back onto the grid. `cfg.model.pos_dim`
    must equal the number of spatial dims (3 for this pipeline's [.,.,4,16,nx]).
    """
    def __init__(self, cfg):
        super(Model, self).__init__()
        self.cfg = cfg
        self._input_dim = cfg.model.input_features
        self._output_dim = cfg.model.output_features
        self._pos_dim = cfg.model.pos_dim
        self.gnn = GraphModel(cfg)
        self._pos_cache = {}   # (spatial, device) -> [N, pos_dim] coordinates

    def _grid_pos(self, spatial, device):
        """Row-major [prod(spatial), pos_dim] unit-domain coordinates.

        The ordering matches the C-order flatten used to turn [C, *spatial]
        into nodes, so coordinates stay aligned with node features.
        """
        key = (spatial, device)
        if key in self._pos_cache:
            return self._pos_cache[key]

        assert len(spatial) == self._pos_dim, (
            f"pos_dim ({self._pos_dim}) must match the number of spatial dims "
            f"({len(spatial)}) of the input grid {spatial}. Set cfg.model.pos_dim "
            f"= {len(spatial)}."
        )
        axes = [torch.linspace(0.0, 1.0, s, device=device) for s in spatial]
        mesh = torch.meshgrid(*axes, indexing="ij")
        pos = torch.stack([m.reshape(-1) for m in mesh], dim=1).contiguous()
        self._pos_cache[key] = pos
        return pos

    def forward(self, x):
        B, C = x.shape[0], x.shape[1]
        spatial = tuple(x.shape[2:])
        device = x.device

        n_nodes = 1
        for s in spatial:
            n_nodes *= s

        # [B, C, *spatial] -> [B, N, C] -> [B*N, C]  (row-major over spatial)
        fx = x.reshape(B, C, n_nodes).permute(0, 2, 1).reshape(B * n_nodes, C)

        # per-node positions, tiled across the batch
        pos_single = self._grid_pos(spatial, device)                 # [N, pos_dim]
        pos = pos_single.unsqueeze(0).expand(B, -1, -1).reshape(B * n_nodes, self._pos_dim)

        # batch assignment vector [0,...,0,1,...,1,...]
        batch = torch.arange(B, device=device).repeat_interleave(n_nodes)

        data = _GraphBatch(x=fx, pos=pos, batch=batch)
        out = self.gnn(data)                                         # [B*N, C_out]

        # [B*N, C_out] -> [B, N, C_out] -> [B, C_out, *spatial]
        out = out.reshape(B, n_nodes, self._output_dim).permute(0, 2, 1).reshape(B, self._output_dim, *spatial)
        return out


if __name__ == "__main__":
    from types import SimpleNamespace

    def make_cfg(in_dim, out_dim):
        return SimpleNamespace(model=SimpleNamespace(
            input_features=in_dim, output_features=out_dim, pos_dim=3,
            feature_width=32, num_layers=2, batch_norm=False, act="gelu",
            local_nodes=64, local_ratio=0.5, local_k=4, local_cos=False, local_pos=True,
            global_ratio=0.5, global_k=4, global_cos=True, global_pos=False,
            num_phys=8, num_heads=4))

    specs = {"neutron": (2, 1, 20), "solid": (2, 1, 8), "fluid": (1, 4, 12)}
    for field, (in_dim, out_dim, nx) in specs.items():
        model = Model(make_cfg(in_dim, out_dim))
        n_params = sum(p.numel() for p in model.parameters())
        x = torch.randn(2, in_dim, 4, 16, nx)
        y = model(x)
        print(f"{field}: params={n_params:,}, in={tuple(x.shape)}, out={tuple(y.shape)}")
        assert y.shape == (2, out_dim, 4, 16, nx)

        loss = y.square().mean()
        loss.backward()
        grads = [p.grad for p in model.parameters() if p.requires_grad and p.grad is not None]
        assert all(torch.isfinite(g).all() for g in grads)
        total_grad = sum(g.abs().sum() for g in grads)
        assert total_grad > 0
        print(f"{field}: backward ok, sum|grad|={total_grad:.3e}")
