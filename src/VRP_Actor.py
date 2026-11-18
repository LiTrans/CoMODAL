import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import MessagePassing
from torch_geometric.utils import softmax

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
INIT = True

from vrpUpdate import V_UAV_MAX, V_ADR_MAX, _leg_metrics as _vrp_leg_metrics, _battery_reserve as _vrp_battery_reserve


# Decoder trace sink: set to a list to capture per-step decisions.
_DECODER_TRACE_SINK = None

def _trace_push(record):
    if _DECODER_TRACE_SINK is not None:
        _DECODER_TRACE_SINK.append(record)


class AdaptiveTimeEncoder(nn.Module):
    """Multi-scale Fourier time embedding with learned log-frequencies."""
    def __init__(self, n_freq: int = 8, d_out: int = 16):
        super().__init__()
        # Initial periods span ~1 min to ~120 min.
        self.log_freqs = nn.Parameter(torch.linspace(-4.8, 0.0, n_freq))
        self.proj = nn.Linear(2 * n_freq + 1, d_out)
        if INIT:
            nn.init.orthogonal_(self.proj.weight)
            nn.init.zeros_(self.proj.bias)

    def forward(self, t_normed: torch.Tensor) -> torch.Tensor:
        t_min  = t_normed * 180.0
        freqs  = self.log_freqs.exp()
        angles = t_min * freqs.view(*([1] * (t_min.dim() - 1)), -1)
        feat   = torch.cat([t_normed, angles.sin(), angles.cos()], dim=-1)
        return self.proj(feat)


class HetGatConv(MessagePassing):

    def __init__(self, in_channels, out_channels, edge_channels,
                 n_heads=8, negative_slope=0.2, dropout=0.0,
                 encoder_attn_mask: bool = True):
        super().__init__(aggr='add')
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.n_heads = n_heads
        self.head_dim = out_channels // n_heads
        self.scale = math.sqrt(self.head_dim)
        self.negative_slope = negative_slope
        self.dropout = dropout
        # When False, attention softmax is dense (no mode-reachability filter).
        self.encoder_attn_mask = bool(encoder_attn_mask)

        self.Wq = nn.Linear(in_channels, out_channels, bias=False)
        self.Wk = nn.Linear(in_channels, out_channels, bias=False)
        self.Wv = nn.Linear(in_channels, out_channels, bias=False)
        self.We = nn.Linear(edge_channels, n_heads, bias=True)
        self.role_bias_all    = nn.Parameter(torch.zeros(n_heads))
        self.role_bias_pickup = nn.Parameter(torch.zeros(n_heads))
        self.role_bias_pair   = nn.Parameter(torch.zeros(n_heads))
        self.norm = nn.LayerNorm(out_channels)

        if INIT:
            for name, p in self.named_parameters():
                if 'weight' in name and p.dim() >= 2:
                    nn.init.orthogonal_(p, gain=1)
                elif 'bias' in name:
                    nn.init.constant_(p, 0)

    def forward(self, x, edge_index, edge_attr, num_depots, num_nodes,
                mask, batch_size, size=None):
        xq = self.Wq(x)
        xk = self.Wk(x)
        xv = self.Wv(x)
        n_pickup = num_nodes // 2

        xq_v = xq.view(batch_size, -1, self.out_channels)
        xk_v = xk.view(batch_size, -1, self.out_channels)
        xv_v = xv.view(batch_size, -1, self.out_channels)

        xq_p = xq_v[:, num_depots:num_depots + n_pickup].reshape(-1, self.out_channels)
        xk_p = xk_v[:, num_depots:num_depots + n_pickup].reshape(-1, self.out_channels)
        xv_p = xv_v[:, num_depots:num_depots + n_pickup].reshape(-1, self.out_channels)

        edge_attr_v = edge_attr.view(batch_size, num_depots + num_nodes, num_depots + num_nodes, -1)
        edge_attr_p = edge_attr_v[:, num_depots:num_depots + n_pickup,
                                     num_depots:num_depots + n_pickup, :].reshape(-1, edge_attr.size(-1))
        edge_attr_flat = edge_attr_v.reshape(-1, edge_attr.size(-1))

        mask_v    = mask.view(batch_size, num_depots + num_nodes, num_depots + num_nodes)
        mask_p    = mask_v[:, num_depots:num_depots + n_pickup,
                              num_depots:num_depots + n_pickup].reshape(-1, 1)
        mask_flat = mask_v.reshape(-1, 1)

        ei_sub = torch.stack([
            torch.arange(batch_size * n_pickup, device=x.device).repeat_interleave(n_pickup),
            torch.arange(batch_size * n_pickup, device=x.device).view(-1, n_pickup)
                .repeat_interleave(n_pickup, dim=0).flatten()
        ])

        batch_off = (torch.arange(batch_size, device=x.device)
                     .repeat_interleave(n_pickup) * (2 * n_pickup))
        local_p   = torch.arange(n_pickup, device=x.device).repeat(batch_size)
        abs_p     = batch_off + local_p
        abs_d     = batch_off + local_p + n_pickup
        ei_pair   = torch.stack([torch.cat([abs_p, abs_d]),
                                  torch.cat([abs_d, abs_p])])

        xq_pd = xq_v[:, num_depots:num_depots + 2 * n_pickup].reshape(-1, self.out_channels)
        xk_pd = xk_v[:, num_depots:num_depots + 2 * n_pickup].reshape(-1, self.out_channels)
        xv_pd = xv_v[:, num_depots:num_depots + 2 * n_pickup].reshape(-1, self.out_channels)

        diag  = torch.arange(n_pickup, device=x.device)
        ea_p2d = (edge_attr_v[:, num_depots:num_depots + n_pickup,
                                 num_depots + n_pickup:num_depots + 2 * n_pickup, :]
                  [:, diag, diag, :].reshape(-1, edge_attr.size(-1)))
        ea_d2p = (edge_attr_v[:, num_depots + n_pickup:num_depots + 2 * n_pickup,
                                 num_depots:num_depots + n_pickup, :]
                  [:, diag, diag, :].reshape(-1, edge_attr.size(-1)))
        edge_attr_pair = torch.cat([ea_p2d, ea_d2p], dim=0)
        mask_pair      = torch.ones(2 * batch_size * n_pickup, 1, device=x.device)

        X_all  = self.propagate(edge_index, q=xq,    k=xk,    v=xv,    edge_attr=edge_attr_flat, mask=mask_flat, key='all',    size=size)
        X_pick = self.propagate(ei_sub,     q=xq_p,  k=xk_p,  v=xv_p,  edge_attr=edge_attr_p,   mask=mask_p,   key='pickup',  size=None)
        X_pair = self.propagate(ei_pair,    q=xq_pd, k=xk_pd, v=xv_pd, edge_attr=edge_attr_pair, mask=mask_pair, key='pair',   size=None)

        X_all_v  = X_all.view(batch_size, -1, self.out_channels)
        X_pick_v = X_pick.view(batch_size, n_pickup, self.out_channels)
        X_pair_v = X_pair.view(batch_size, 2 * n_pickup, self.out_channels)

        out = torch.cat([
            X_all_v[:, :num_depots, :],
            X_all_v[:, num_depots:num_depots + n_pickup, :]           + X_pick_v + X_pair_v[:, :n_pickup, :],
            X_all_v[:, num_depots + n_pickup:num_depots + num_nodes, :]            + X_pair_v[:, n_pickup:, :],
        ], dim=1).view(-1, self.out_channels)

        return self.norm(out + xv)

    def message(self, edge_index_i, q_i, k_j, v_j, size_i, edge_attr, mask, key):
        E = q_i.size(0)
        H, d = self.n_heads, self.head_dim
        q = q_i.view(E, H, d)
        k = k_j.view(E, H, d)
        v = v_j.view(E, H, d)
        alpha = (q * k).sum(-1) / self.scale
        alpha = alpha + self.We(edge_attr)
        if key == 'all':
            alpha = alpha + self.role_bias_all
        elif key == 'pickup':
            alpha = alpha + self.role_bias_pickup
        else:
            alpha = alpha + self.role_bias_pair
        if self.encoder_attn_mask:
            alpha = alpha.masked_fill(mask.expand(-1, H) == 0, -1e9)
        alpha = softmax(alpha, edge_index_i, num_nodes=size_i)
        alpha = F.dropout(alpha, p=self.dropout, training=self.training)
        return (v * alpha.unsqueeze(-1)).reshape(E, H * d)

    def update(self, aggr_out):
        return torch.nan_to_num(aggr_out, nan=0.0)

class SimpleGatConv(MessagePassing):

    def __init__(self, in_channels, out_channels, edge_channels,
                 n_heads=8, negative_slope=0.2, dropout=0.0,
                 encoder_attn_mask: bool = True):
        super().__init__(aggr='add')
        self.out_channels = out_channels
        self.n_heads = n_heads
        self.head_dim = out_channels // n_heads
        self.scale = math.sqrt(self.head_dim)
        self.negative_slope = negative_slope
        self.dropout = dropout
        self.encoder_attn_mask = bool(encoder_attn_mask)

        self.Wq = nn.Linear(in_channels, out_channels, bias=False)
        self.Wk = nn.Linear(in_channels, out_channels, bias=False)
        self.Wv = nn.Linear(in_channels, out_channels, bias=False)
        self.We = nn.Linear(edge_channels, n_heads, bias=True)
        self.norm = nn.LayerNorm(out_channels)

        if INIT:
            for name, p in self.named_parameters():
                if 'weight' in name and p.dim() >= 2:
                    nn.init.orthogonal_(p, gain=1)
                elif 'bias' in name:
                    nn.init.constant_(p, 0)

    def forward(self, x, edge_index, edge_attr, mask, batch_size, size=None):
        edge_attr_flat = edge_attr.view(-1, edge_attr.size(-1))
        mask_flat = mask.view(-1, 1)

        xq = self.Wq(x)
        xk = self.Wk(x)
        xv = self.Wv(x)

        out = self.propagate(edge_index, q=xq, k=xk, v=xv,
                             edge_attr=edge_attr_flat, mask=mask_flat, size=size)
        return self.norm(out + xv)

    def message(self, edge_index_i, q_i, k_j, v_j, size_i, edge_attr, mask):
        E = q_i.size(0)
        H, d = self.n_heads, self.head_dim
        q = q_i.view(E, H, d)
        k = k_j.view(E, H, d)
        v = v_j.view(E, H, d)
        alpha = (q * k).sum(-1) / self.scale
        alpha = alpha + self.We(edge_attr)
        if self.encoder_attn_mask:
            alpha = alpha.masked_fill(mask.expand(-1, H) == 0, -1e9)
        alpha = softmax(alpha, edge_index_i, num_nodes=size_i)
        alpha = F.dropout(alpha, p=self.dropout, training=self.training)
        return (v * alpha.unsqueeze(-1)).reshape(E, H * d)

    def update(self, aggr_out):
        return torch.nan_to_num(aggr_out, nan=0.0)

class SimpleGatEncoder(nn.Module):

    def __init__(self, input_node_dim=11, hidden_node_dim=128,
                 input_edge_dim=1, hidden_edge_dim=16, conv_layers=3,
                 encoder_attn_mask: bool = True):
        super().__init__()
        self.hidden_node_dim = hidden_node_dim
        self.encoder_attn_mask = bool(encoder_attn_mask)

        self.W_depot    = nn.Linear(input_node_dim, hidden_node_dim, bias=False)
        self.W_pickup   = nn.Linear(input_node_dim * 2, hidden_node_dim, bias=False)
        self.W_delivery = nn.Linear(input_node_dim * 2, hidden_node_dim, bias=False)

        self.W_edge_uav  = nn.Linear(input_edge_dim, hidden_edge_dim, bias=False)
        self.W_edge_adr  = nn.Linear(input_edge_dim, hidden_edge_dim, bias=False)
        self.ln_node     = nn.LayerNorm(hidden_node_dim)
        self.ln_edge_uav = nn.LayerNorm(hidden_edge_dim)
        self.ln_edge_adr = nn.LayerNorm(hidden_edge_dim)

        self.mode_token_uav = nn.Parameter(torch.zeros(hidden_node_dim))
        self.mode_token_adr = nn.Parameter(torch.zeros(hidden_node_dim))
        nn.init.normal_(self.mode_token_uav, std=0.02)
        nn.init.normal_(self.mode_token_adr, std=0.02)

        self.convs_uav = nn.ModuleList([
            SimpleGatConv(hidden_node_dim, hidden_node_dim, hidden_edge_dim,
                          encoder_attn_mask=self.encoder_attn_mask)
            for _ in range(conv_layers)
        ])
        self.convs_adr = nn.ModuleList([
            SimpleGatConv(hidden_node_dim, hidden_node_dim, hidden_edge_dim,
                          encoder_attn_mask=self.encoder_attn_mask)
            for _ in range(conv_layers)
        ])

        self.ff_uav = nn.Sequential(
            nn.Linear(hidden_node_dim, hidden_node_dim), nn.ReLU(),
            nn.Linear(hidden_node_dim, hidden_node_dim),
        )
        self.ff_adr = nn.Sequential(
            nn.Linear(hidden_node_dim, hidden_node_dim), nn.ReLU(),
            nn.Linear(hidden_node_dim, hidden_node_dim),
        )
        self.ln_final_uav = nn.LayerNorm(hidden_node_dim)
        self.ln_final_adr = nn.LayerNorm(hidden_node_dim)

        self.time_enc_uav = AdaptiveTimeEncoder(d_out=hidden_node_dim)
        self.time_enc_adr = AdaptiveTimeEncoder(d_out=hidden_node_dim)
        # Learned token for unrevealed nodes.
        self.unknown_token = nn.Parameter(torch.zeros(hidden_node_dim))
        nn.init.normal_(self.unknown_token, std=0.02)

        if INIT:
            for name, p in self.named_parameters():
                if 'weight' in name and p.dim() >= 2:
                    nn.init.orthogonal_(p, gain=1)
                elif 'bias' in name:
                    nn.init.constant_(p, 0)

    def forward(self, data, initial_visited=None):
        if initial_visited is None:
            initial_visited = data.get('_iv', None)

        batch_size = data['x'].shape[0]
        feat  = data['x']
        n_total = feat.shape[1]
        nd = data.get('n_depots', None)
        if nd is not None:
            num_depots = int(nd[0]) if hasattr(nd, '__len__') else int(nd)
        else:
            num_depots = int((data['demand'] == 0).sum().item() // batch_size)
        num_nodes = n_total - num_depots
        n_pickup  = num_nodes // 2

        depot_f         = feat[:, :num_depots, :].reshape(batch_size * num_depots, -1)
        pickup_f        = feat[:, num_depots:num_depots + n_pickup, :]
        delivery_f      = feat[:, num_depots + n_pickup:, :]
        pickup_paired   = torch.cat([pickup_f,   delivery_f], dim=-1).reshape(batch_size * n_pickup, -1)
        delivery_paired = torch.cat([delivery_f, pickup_f],   dim=-1).reshape(batch_size * n_pickup, -1)

        emb_depot    = self.W_depot(depot_f)
        emb_pickup   = self.W_pickup(pickup_paired)
        emb_delivery = self.W_delivery(delivery_paired)

        x = torch.cat([
            emb_depot.view(batch_size, num_depots, -1),
            emb_pickup.view(batch_size, n_pickup, -1),
            emb_delivery.view(batch_size, n_pickup, -1),
        ], dim=1).view(-1, self.hidden_node_dim)
        x = self.ln_node(x)

        ea_uav = data['edge_attr_uav'].view(-1, data['edge_attr_uav'].shape[-1])
        ea_adr = data['edge_attr_adr'].view(-1, data['edge_attr_adr'].shape[-1])
        ea_uav = self.ln_edge_uav(self.W_edge_uav(ea_uav))
        ea_adr = self.ln_edge_adr(self.W_edge_adr(ea_adr))

        ei = data['edge_index']
        if ei.dim() == 3:
            # [B, 2, N²] local → [2, B*N²] global: permute is required so row 0
            # ends up as all sources and row 1 all destinations after batching.
            off = (torch.arange(batch_size, device=ei.device) * n_total).view(batch_size, 1, 1)
            edge_index = (ei + off).permute(1, 0, 2).reshape(2, -1)
        else:
            edge_index = ei
        mask_uav   = data['mask_adjacency_uav']
        mask_adr   = data['mask_adjacency_adr']

        x_uav = x + self.mode_token_uav
        x_adr = x + self.mode_token_adr

        # dim 3 / dim 4 are t_pickup / t_delivery in x; each stream learns its own encoding.
        t_pk = feat[:, num_depots:num_depots + n_pickup, 3:4]
        t_dl = feat[:, num_depots + n_pickup:, 4:5]

        x_uav_v = x_uav.reshape(batch_size, n_total, self.hidden_node_dim)
        x_adr_v = x_adr.reshape(batch_size, n_total, self.hidden_node_dim)

        te_uav_pk = self.time_enc_uav(t_pk.reshape(-1, 1)).reshape(batch_size, n_pickup, -1)
        te_uav_dl = self.time_enc_uav(t_dl.reshape(-1, 1)).reshape(batch_size, n_pickup, -1)
        te_adr_pk = self.time_enc_adr(t_pk.reshape(-1, 1)).reshape(batch_size, n_pickup, -1)
        te_adr_dl = self.time_enc_adr(t_dl.reshape(-1, 1)).reshape(batch_size, n_pickup, -1)

        x_uav = torch.cat([
            x_uav_v[:, :num_depots, :],
            x_uav_v[:, num_depots:num_depots + n_pickup, :] + te_uav_pk,
            x_uav_v[:, num_depots + n_pickup:, :]           + te_uav_dl,
        ], dim=1).reshape(-1, self.hidden_node_dim)

        x_adr = torch.cat([
            x_adr_v[:, :num_depots, :],
            x_adr_v[:, num_depots:num_depots + n_pickup, :] + te_adr_pk,
            x_adr_v[:, num_depots + n_pickup:, :]           + te_adr_dl,
        ], dim=1).reshape(-1, self.hidden_node_dim)

        # Replace unrevealed-node features with a learned token (vs zeroed-out).
        if initial_visited is not None:
            iv = initial_visited.reshape(-1, 1).float()
            unk = self.unknown_token.unsqueeze(0)
            x_uav = x_uav * (1.0 - iv) + unk * iv
            x_adr = x_adr * (1.0 - iv) + unk * iv

        for conv_u, conv_a in zip(self.convs_uav, self.convs_adr):
            x_uav = conv_u(x_uav, edge_index, ea_uav, mask_uav, batch_size)
            x_adr = conv_a(x_adr, edge_index, ea_adr, mask_adr, batch_size)

        x_uav = self.ln_final_uav(x_uav + self.ff_uav(x_uav))
        x_adr = self.ln_final_adr(x_adr + self.ff_adr(x_adr))

        return (x_uav.reshape(batch_size, n_total, self.hidden_node_dim),
                x_adr.reshape(batch_size, n_total, self.hidden_node_dim))

class Encoder(nn.Module):
    def __init__(self, input_node_dim=11, hidden_node_dim=128,
                 input_edge_dim=4, hidden_edge_dim=16, conv_layers=3,
                 encoder_attn_mask: bool = True):
        super().__init__()
        self.hidden_node_dim = hidden_node_dim
        self.hidden_edge_dim = hidden_edge_dim
        self.encoder_attn_mask = bool(encoder_attn_mask)

        self.W_depot = nn.Linear(input_node_dim, hidden_node_dim, bias=False)
        self.W_pickup = nn.Linear(input_node_dim * 2, hidden_node_dim, bias=False)
        self.W_delivery = nn.Linear(input_node_dim * 2, hidden_node_dim, bias=False)

        self.W_edge_uav = nn.Linear(input_edge_dim, hidden_edge_dim, bias=False)
        self.W_edge_adr = nn.Linear(input_edge_dim, hidden_edge_dim, bias=False)

        self.ln_node = nn.LayerNorm(hidden_node_dim)
        self.ln_edge_uav = nn.LayerNorm(hidden_edge_dim)
        self.ln_edge_adr = nn.LayerNorm(hidden_edge_dim)

        self.mode_token_uav = nn.Parameter(torch.zeros(hidden_node_dim))
        self.mode_token_adr = nn.Parameter(torch.zeros(hidden_node_dim))
        nn.init.normal_(self.mode_token_uav, std=0.02)
        nn.init.normal_(self.mode_token_adr, std=0.02)

        self.convs_uav = nn.ModuleList([
            HetGatConv(hidden_node_dim, hidden_node_dim, hidden_edge_dim,
                       encoder_attn_mask=self.encoder_attn_mask)
            for _ in range(conv_layers)
        ])
        self.convs_adr = nn.ModuleList([
            HetGatConv(hidden_node_dim, hidden_node_dim, hidden_edge_dim,
                       encoder_attn_mask=self.encoder_attn_mask)
            for _ in range(conv_layers)
        ])

        self.ff_uav = nn.Sequential(
            nn.Linear(hidden_node_dim, hidden_node_dim),
            nn.ReLU(),
            nn.Linear(hidden_node_dim, hidden_node_dim),
        )
        self.ff_adr = nn.Sequential(
            nn.Linear(hidden_node_dim, hidden_node_dim),
            nn.ReLU(),
            nn.Linear(hidden_node_dim, hidden_node_dim),
        )
        self.ln_final_uav = nn.LayerNorm(hidden_node_dim)
        self.ln_final_adr = nn.LayerNorm(hidden_node_dim)

        self.time_enc_uav = AdaptiveTimeEncoder(d_out=hidden_node_dim)
        self.time_enc_adr = AdaptiveTimeEncoder(d_out=hidden_node_dim)
        self.unknown_token = nn.Parameter(torch.zeros(hidden_node_dim))
        nn.init.normal_(self.unknown_token, std=0.02)

        if INIT:
            for name, p in self.named_parameters():
                if 'weight' in name and p.dim() >= 2:
                    nn.init.orthogonal_(p, gain=1)
                elif 'bias' in name:
                    nn.init.constant_(p, 0)

    def forward(self, data, initial_visited=None):
        if initial_visited is None:
            initial_visited = data.get('_iv', None)

        batch_size = data['x'].shape[0]
        feat = data['x']
        n_total = feat.shape[1]
        nd = data.get('n_depots', None)
        if nd is not None:
            num_depots = int(nd[0]) if hasattr(nd, '__len__') else int(nd)
        else:
            num_depots = int((data['demand'] == 0).sum().item() // batch_size)
        num_nodes = n_total - num_depots
        n_pickup = num_nodes // 2

        depot_f = feat[:, :num_depots, :].reshape(batch_size * num_depots, -1)
        pickup_f = feat[:, num_depots:num_depots + n_pickup, :]
        delivery_f = feat[:, num_depots + n_pickup:, :]
        pickup_paired   = torch.cat([pickup_f,   delivery_f], dim=-1).reshape(batch_size * n_pickup, -1)
        delivery_paired = torch.cat([delivery_f, pickup_f],   dim=-1).reshape(batch_size * n_pickup, -1)

        emb_depot    = self.W_depot(depot_f)
        emb_pickup   = self.W_pickup(pickup_paired)
        emb_delivery = self.W_delivery(delivery_paired)

        x = torch.cat([
            emb_depot.view(batch_size, num_depots, -1),
            emb_pickup.view(batch_size, n_pickup, -1),
            emb_delivery.view(batch_size, n_pickup, -1),
        ], dim=1).view(-1, self.hidden_node_dim)
        x = self.ln_node(x)

        ea_uav = data['edge_attr_uav'].view(-1, data['edge_attr_uav'].shape[-1])
        ea_adr = data['edge_attr_adr'].view(-1, data['edge_attr_adr'].shape[-1])
        ea_uav = self.ln_edge_uav(self.W_edge_uav(ea_uav))
        ea_adr = self.ln_edge_adr(self.W_edge_adr(ea_adr))

        ei = data['edge_index']
        if ei.dim() == 3:
            off = (torch.arange(batch_size, device=ei.device) * n_total).view(batch_size, 1, 1)
            edge_index = (ei + off).permute(1, 0, 2).reshape(2, -1)
        else:
            edge_index = ei
        mask_uav = data['mask_adjacency_uav']
        mask_adr = data['mask_adjacency_adr']

        x_uav = x + self.mode_token_uav
        x_adr = x + self.mode_token_adr

        t_pk = feat[:, num_depots:num_depots + n_pickup, 3:4]
        t_dl = feat[:, num_depots + n_pickup:, 4:5]

        x_uav_v = x_uav.reshape(batch_size, n_total, self.hidden_node_dim)
        x_adr_v = x_adr.reshape(batch_size, n_total, self.hidden_node_dim)

        te_uav_pk = self.time_enc_uav(t_pk.reshape(-1, 1)).reshape(batch_size, n_pickup, -1)
        te_uav_dl = self.time_enc_uav(t_dl.reshape(-1, 1)).reshape(batch_size, n_pickup, -1)
        te_adr_pk = self.time_enc_adr(t_pk.reshape(-1, 1)).reshape(batch_size, n_pickup, -1)
        te_adr_dl = self.time_enc_adr(t_dl.reshape(-1, 1)).reshape(batch_size, n_pickup, -1)

        x_uav = torch.cat([
            x_uav_v[:, :num_depots, :],
            x_uav_v[:, num_depots:num_depots + n_pickup, :] + te_uav_pk,
            x_uav_v[:, num_depots + n_pickup:, :]           + te_uav_dl,
        ], dim=1).reshape(-1, self.hidden_node_dim)

        x_adr = torch.cat([
            x_adr_v[:, :num_depots, :],
            x_adr_v[:, num_depots:num_depots + n_pickup, :] + te_adr_pk,
            x_adr_v[:, num_depots + n_pickup:, :]           + te_adr_dl,
        ], dim=1).reshape(-1, self.hidden_node_dim)

        if initial_visited is not None:
            iv = initial_visited.reshape(-1, 1).float()
            unk = self.unknown_token.unsqueeze(0)
            x_uav = x_uav * (1.0 - iv) + unk * iv
            x_adr = x_adr * (1.0 - iv) + unk * iv

        for conv_u, conv_a in zip(self.convs_uav, self.convs_adr):
            x_uav = conv_u(x_uav, edge_index, ea_uav, num_depots, num_nodes,
                           mask_uav, batch_size)
            x_adr = conv_a(x_adr, edge_index, ea_adr, num_depots, num_nodes,
                           mask_adr, batch_size)

        x_uav = self.ln_final_uav(x_uav + self.ff_uav(x_uav))
        x_adr = self.ln_final_adr(x_adr + self.ff_adr(x_adr))

        x_uav = x_uav.reshape(batch_size, n_total, self.hidden_node_dim)
        x_adr = x_adr.reshape(batch_size, n_total, self.hidden_node_dim)

        return x_uav, x_adr

class MultiHeadAttention(nn.Module):
    def __init__(self, n_heads, cat, input_dim, hidden_dim):
        super().__init__()
        self.n_heads = n_heads
        self.hidden_dim = hidden_dim
        self.head_dim = hidden_dim // n_heads
        self.norm = 1.0 / math.sqrt(self.head_dim)

        self.w = nn.Linear(input_dim * cat, hidden_dim, bias=False)
        self.k = nn.Linear(input_dim, hidden_dim, bias=False)
        self.v = nn.Linear(input_dim, hidden_dim, bias=False)
        self.fc = nn.Linear(hidden_dim, hidden_dim, bias=False)

        if INIT:
            for name, p in self.named_parameters():
                if 'weight' in name and p.dim() >= 2:
                    nn.init.orthogonal_(p, gain=1)

    def precompute(self, context):
        B, N, _ = context.shape
        K = self.k(context).view(B, N, self.n_heads, self.head_dim).transpose(1, 2)
        V = self.v(context).view(B, N, self.n_heads, self.head_dim).transpose(1, 2)
        return K, V

    def forward(self, Q_raw, K, V, mask):
        B, A, _ = Q_raw.shape
        Q = self.w(Q_raw).view(B, A, self.n_heads, self.head_dim).transpose(1, 2)
        compat = self.norm * torch.matmul(Q, K.transpose(2, 3))
        mask = mask.unsqueeze(1).expand_as(compat)
        compat = compat.masked_fill(mask.bool(), float(-1e9))
        scores = F.softmax(compat, dim=-1)
        out = torch.matmul(scores, V)
        out = out.transpose(1, 2).contiguous().view(B, A, self.hidden_dim)
        return self.fc(out)

def _parallel_select(u: torch.Tensor, num_depots: int,
                     greedy: bool, current_nodes=None) -> tuple:
    B, A, N = u.shape
    dev = u.device
    valid = u > -1e8

    if not greedy:
        gumbel = -torch.log(
            -torch.log(torch.rand_like(u).clamp(min=1e-10)) + 1e-10
        )
        u_g = u + gumbel
    else:
        u_g = u

    winner = u_g.argmax(dim=1)

    ag_ids = torch.arange(A, device=dev).view(1, A, 1)
    agent_wins = (winner.unsqueeze(1) == ag_ids) & valid
    agent_wins[:, :, :num_depots] = valid[:, :, :num_depots]

    u_won = u_g.masked_fill(~agent_wins, float('-inf'))
    no_win = (u_won == float('-inf')).all(dim=-1)

    u_fallback = u_g.masked_fill(~valid, float('-inf'))
    u_fallback[:, :, num_depots:] = float('-inf')
    u_won = torch.where(no_win.unsqueeze(-1), u_fallback, u_won)

    all_masked = (u_won == float('-inf')).all(dim=-1)
    # No feasible move: stay at current node without executing service.
    if current_nodes is None:
        current_nodes = torch.zeros(B, A, device=dev, dtype=torch.long)
    stay_scores = u_won.gather(2, current_nodes.unsqueeze(2))
    u_won = u_won.scatter(2, current_nodes.unsqueeze(2),
                         torch.where(all_masked.unsqueeze(2), 0.0, stay_scores))

    selected = u_won.argmax(dim=-1)

    lp = F.log_softmax(
        u.masked_fill(~valid, float('-inf')).clamp(min=-1e9), dim=-1
    )
    lp = lp.masked_fill(all_masked.unsqueeze(-1), 0.0)
    log_p = lp.gather(2, selected.unsqueeze(2)).squeeze(2)

    # Categorical entropy over valid actions; mask invalid entries to avoid NaN grads.
    probs = lp.exp()
    plogp = probs * lp
    plogp = torch.where(torch.isfinite(plogp), plogp, torch.zeros_like(plogp))
    entropy = -plogp.sum(dim=-1)
    entropy = entropy.masked_fill(all_masked, 0.0)

    return selected, log_p, entropy

class CoopDecoder(nn.Module):
    def __init__(self, input_dim, hidden_dim, n_heads=8, decoder_biases: bool = True,
                 margin_bias: bool = False, dyn_features: str = 'basic',
                 battery_mask: bool = False, battery_mask_reserve_frac: float = 0.10,
                 battery_mask_reserve_frac_uav: float = None,
                 battery_mask_reserve_frac_adr: float = None,
                 battery_mask_lookahead: bool = False,
                 battery_mandatory_recharge: bool = False,
                 battery_mandatory_recharge_buffer: float = 0.10,
                 paper_strict_depot_mask: bool = True,
                 paper_strict_depot_margin: float = 0.10,
                 use_chain_lookahead_in_mask: bool = True):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.n_heads = n_heads
        self.battery_mandatory_recharge = bool(battery_mandatory_recharge)
        self.battery_mandatory_recharge_buffer = float(battery_mandatory_recharge_buffer)
        # Depot-as-fallback mask (paper-strict, Section 4.4).
        self.paper_strict_depot_mask = bool(paper_strict_depot_mask)
        self.paper_strict_depot_margin = float(paper_strict_depot_margin)
        # Chain-lookahead toggle: redundant with vrpUpdate._safe_continuation_mask;
        # kept ON as defence-in-depth for untrained ckpts.
        self.use_chain_lookahead_in_mask = bool(use_chain_lookahead_in_mask)
        self._diag = {
            'dyn_depot_fires': 0,
            'dyn_depot_total_slots': 0,
            'depot_pick_sum_tour_len': 0,
            'depot_pick_count': 0,
            'legal_cust_at_depot_pick_sum': 0,
        }
        # Battery hard-mask: customer nodes whose post-visit battery (incl.
        # return-to-depot) falls below `battery_mask_reserve_frac` are masked.
        # Per-fleet overrides fall back to the single-scalar reserve_frac.
        self.battery_mask = bool(battery_mask)
        self.battery_mask_reserve_frac = float(battery_mask_reserve_frac)
        self.battery_mask_reserve_frac_uav = (
            float(battery_mask_reserve_frac_uav)
            if battery_mask_reserve_frac_uav is not None
            else float(battery_mask_reserve_frac)
        )
        self.battery_mask_reserve_frac_adr = (
            float(battery_mask_reserve_frac_adr)
            if battery_mask_reserve_frac_adr is not None
            else float(battery_mask_reserve_frac)
        )
        # When True, mask margin uses pickup→delivery→depot chain (not just target→depot).
        self.battery_mask_lookahead = bool(battery_mask_lookahead)
        # When False, dist / urgency / live-margin biases are zeroed (ablation).
        self.decoder_biases = bool(decoder_biases)
        # When True, adds a per-candidate post-visit battery-margin logit bias.
        self.margin_bias = bool(margin_bias)
        # 'basic' = 3 scalars (Section 4.2); 'extended' = 10 scalars.
        if dyn_features not in ('basic', 'extended'):
            raise ValueError("dyn_features must be 'basic' or 'extended'")
        self.dyn_features = dyn_features
        self.dyn_dim = 10 if dyn_features == 'extended' else 3

        self.mha_uav = MultiHeadAttention(n_heads, 1, input_dim, hidden_dim)
        self.mha_adr = MultiHeadAttention(n_heads, 1, input_dim, hidden_dim)
        self.k_proj_uav = nn.Linear(input_dim, hidden_dim, bias=False)
        self.k_proj_adr = nn.Linear(input_dim, hidden_dim, bias=False)

        self.agent_id_emb_uav = nn.Parameter(torch.zeros(16, hidden_dim))
        self.agent_id_emb_adr = nn.Parameter(torch.zeros(16, hidden_dim))
        nn.init.normal_(self.agent_id_emb_uav, std=0.02)
        nn.init.normal_(self.agent_id_emb_adr, std=0.02)

        self.Wd_uav     = nn.Linear(hidden_dim * 2, hidden_dim, bias=False)
        self.Wd_adr     = nn.Linear(hidden_dim * 2, hidden_dim, bias=False)
        self.dyn_proj_uav = nn.Linear(self.dyn_dim, hidden_dim, bias=True)
        self.dyn_proj_adr = nn.Linear(self.dyn_dim, hidden_dim, bias=True)
        self.recharge_state_weight = nn.Parameter(torch.zeros(2, hidden_dim))
        self.stop_weight = nn.Parameter(torch.zeros(2, hidden_dim))
        self.stop_bias = nn.Parameter(torch.full((2,), -20.0))
        self.ln_uav = nn.LayerNorm(hidden_dim)
        self.ln_adr = nn.LayerNorm(hidden_dim)

        self.dist_compat_uav = nn.Linear(1, 1, bias=True)
        self.dist_compat_adr = nn.Linear(1, 1, bias=True)
        nn.init.constant_(self.dist_compat_uav.weight, -1.0)
        nn.init.constant_(self.dist_compat_uav.bias,    0.0)
        nn.init.constant_(self.dist_compat_adr.weight, -1.0)
        nn.init.constant_(self.dist_compat_adr.bias,    0.0)
        self.h_empty_uav = nn.Parameter(torch.zeros(1, 1, hidden_dim))
        self.h_empty_adr = nn.Parameter(torch.zeros(1, 1, hidden_dim))

        self.fc1_uav = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.fc1_adr = nn.Linear(hidden_dim, hidden_dim, bias=False)

        self.cross_attn_uav = nn.MultiheadAttention(hidden_dim, n_heads,
                                                     batch_first=True, bias=False, dropout=0.0)
        self.cross_attn_adr = nn.MultiheadAttention(hidden_dim, n_heads,
                                                     batch_first=True, bias=False, dropout=0.0)
        self.ln_cross_uav = nn.LayerNorm(hidden_dim)
        self.ln_cross_adr = nn.LayerNorm(hidden_dim)

        if INIT:
            for name, p in self.named_parameters():
                if 'weight' in name and p.dim() >= 2:
                    nn.init.orthogonal_(p, gain=1)
        nn.init.constant_(self.dist_compat_uav.weight, -1.0)
        nn.init.constant_(self.dist_compat_adr.weight, -1.0)

        self.urgency_proj_uav = nn.Linear(1, 1, bias=True)
        self.urgency_proj_adr = nn.Linear(1, 1, bias=True)
        # Small positive init so urgency starts as a weak-but-present signal.
        nn.init.constant_(self.urgency_proj_uav.weight, 0.1)
        nn.init.constant_(self.urgency_proj_uav.bias,   0.0)
        nn.init.constant_(self.urgency_proj_adr.weight, 0.1)
        nn.init.constant_(self.urgency_proj_adr.bias,   0.0)

        # Learned urgency time-scales (init = 120 min, clamped to [30, 600]).
        self.log_urg_pool_scale = nn.Parameter(torch.tensor(math.log(120.0)))
        self.log_urg_scale_uav  = nn.Parameter(torch.tensor(math.log(120.0)))
        self.log_urg_scale_adr  = nn.Parameter(torch.tensor(math.log(120.0)))

        # Live feasibility margin: per-step (deadline - T_t_agent - travel_time) bias.
        self.live_margin_proj_uav = nn.Linear(1, 1, bias=True)
        self.live_margin_proj_adr = nn.Linear(1, 1, bias=True)
        nn.init.constant_(self.live_margin_proj_uav.weight, 0.3)
        nn.init.constant_(self.live_margin_proj_uav.bias,   0.0)
        nn.init.constant_(self.live_margin_proj_adr.weight, 0.3)
        nn.init.constant_(self.live_margin_proj_adr.bias,   0.0)

        # Only construct these when flag is on so the RNG stream for other weights
        # is untouched (bit-for-bit init reproducibility).
        if self.margin_bias:
            self.post_visit_margin_proj_uav = nn.Linear(1, 1, bias=True)
            self.post_visit_margin_proj_adr = nn.Linear(1, 1, bias=True)
            nn.init.constant_(self.post_visit_margin_proj_uav.weight, 0.2)
            nn.init.constant_(self.post_visit_margin_proj_uav.bias,   0.0)
            nn.init.constant_(self.post_visit_margin_proj_adr.weight, 0.2)
            nn.init.constant_(self.post_visit_margin_proj_adr.bias,   0.0)

    def _compatibility(self, q_vec, K, mask, T=1.0):
        norm = 1.0 / math.sqrt(K.shape[-1])
        compat = norm * torch.bmm(q_vec, K.transpose(1, 2))
        compat = 10.0 * torch.tanh(compat / T)
        compat = compat.masked_fill(mask.bool(), float('-inf'))
        return compat

    def _load_from_state_dict(self, state_dict, prefix, *args, **kwargs):
        # Back-compat: older route checkpoints omit these three keys.
        names = ('recharge_state_weight', 'stop_weight', 'stop_bias')
        if not any(prefix + name in state_dict for name in names):
            for name in names:
                default = -20.0 if name == 'stop_bias' else 0.0
                state_dict[prefix + name] = torch.full_like(getattr(self, name), default)
        super()._load_from_state_dict(state_dict, prefix, *args, **kwargs)

    def _stop_logits(self, ctx_uav, ctx_adr):
        return torch.cat((ctx_uav @ self.stop_weight[0] + self.stop_bias[0],
                          ctx_adr @ self.stop_weight[1] + self.stop_bias[1]), dim=1)

    @staticmethod
    def _agent_ids(table, count, batch_size):
        if count <= table.size(0):
            ids = table[:count]
        else:
            repeats = math.ceil(count / table.size(0))
            ids = table.repeat(repeats, 1)[:count]
        return ids.unsqueeze(0).expand(batch_size, -1, -1)

    def forward(self, emb_uav, emb_adr,
                capacity, demand, battery, time_window,
                num_depots, num_nodes, num_uav, num_adr,
                edge_attr_uav, edge_attr_adr, T=1.0, greedy=False,
                acc_uav=None, acc_adr=None, parallel_select=False,
                n_depots_uav=None, initial_visited=None,
                start_node=None, start_time=None,
                start_battery=None, start_capacity=None,
                available_mask=None, delivery_owner=None, wind=None, coordinates=None,
                sharing_context=None, service_policy='legacy_optional', allow_partial=False,
                max_decode_steps=None, agent_depot_allowed=None, physical_config=None,
                remaining_deliveries=None, mask_adj_uav=None, mask_adj_adr=None,
                return_entropy=False):

        batch_size, _, H = emb_uav.shape
        n_agents = num_uav + num_adr
        from reward import TensorSharingState, IncompleteTensorRouteError, native_physical_configuration
        from vrpUpdate import _pending_deliveries, pending_after_moves
        config = native_physical_configuration({'physical_config': physical_config})

        mask1 = emb_uav.new_zeros((batch_size, n_agents, emb_uav.size(1)))
        if initial_visited is not None:
            mask1 = initial_visited.unsqueeze(1).expand(batch_size, n_agents, -1).clone()
        mask = emb_uav.new_zeros((batch_size, n_agents, emb_uav.size(1)))

        self._diag = {
            'dyn_depot_fires': 0,
            'dyn_depot_total_slots': 0,
            'depot_pick_sum_tour_len': 0,
            'depot_pick_count': 0,
            'legal_cust_at_depot_pick_sum': 0,
            'last_episode_steps': 0,
            'last_max_steps': 0,
            'hit_max_steps_batches': 0,
        }
        _tour_step = emb_uav.new_zeros((batch_size, n_agents), dtype=torch.long)

        # start_battery (when set) initialises running battery below full
        # so training can start mid-shift with partial charge.
        full_capacity = capacity.view(batch_size, n_agents).unsqueeze(2).float()
        _full_bat = battery.view(batch_size, n_agents).unsqueeze(2).float()
        E = _full_bat.squeeze(2)
        if start_battery is None and config is not None:
            start_battery = _full_bat.new_tensor(
                [config.uav_battery.initial_charge_kj] * num_uav
                + [config.adr_battery.initial_charge_kj] * num_adr).expand(batch_size, -1)

        battery  = (start_battery.view(batch_size, n_agents).unsqueeze(2).float()
                    if start_battery is not None else _full_bat)
        capacity = (start_capacity.view(batch_size, n_agents).unsqueeze(2).float()
                    if start_capacity is not None
                    else full_capacity)
        T_t = (start_time.view(batch_size, n_agents, 1).to(emb_uav.device).float()
               if start_time is not None
               else torch.zeros(batch_size, n_agents, 1, device=emb_uav.device))
        if available_mask is not None:
            available_mask = available_mask.view(batch_size, n_agents).to(emb_uav.device).float()
        demands = demand.view(batch_size, emb_uav.size(1))
        time_window = time_window.view(batch_size, emb_uav.size(1))
        edge_attr_uav = edge_attr_uav.view(batch_size, num_depots + num_nodes, num_depots + num_nodes)
        edge_attr_adr = edge_attr_adr.view(batch_size, num_depots + num_nodes, num_depots + num_nodes)

        K_uav, V_uav = self.mha_uav.precompute(emb_uav)
        K_adr, V_adr = self.mha_adr.precompute(emb_adr)
        K_all_uav = self.k_proj_uav(emb_uav)
        K_all_adr = self.k_proj_adr(emb_adr)

        n_pickup = num_nodes // 2
        if service_policy not in ('mandatory', 'legacy_optional'):
            raise ValueError('Unknown decoder service policy')
        sharing = TensorSharingState(sharing_context or {}, batch_size, n_pickup,
                                     num_uav, n_agents, emb_uav.device)
        remaining_deliveries = _pending_deliveries(demands, num_depots, remaining_deliveries)
        if agent_depot_allowed is not None:
            if tuple(agent_depot_allowed.shape) != (batch_size, n_agents, num_depots):
                raise ValueError('Invalid vehicle depot permissions')
            agent_depot_allowed = agent_depot_allowed.to(emb_uav.device).bool()
        route_data = sharing_context or {}
        diagnostic_actions = route_data.get('diagnostic_actions')
        recharge_budget = tensor_recharge_budget(route_data, (batch_size, n_agents), emb_uav.device)
        shift_cap = route_data.get('max_recharges_per_vehicle')
        if shift_cap is not None:
            shift_cap = torch.as_tensor(shift_cap, device=emb_uav.device).reshape(-1, 1)
            if shift_cap.size(0) not in (1, batch_size):
                raise ValueError('Invalid per-instance recharge cap')
            shift_cap = shift_cap.clamp_min(1)
        recharge_depots = recharge_depot_permissions(agent_depot_allowed, recharge_budget, num_depots)
        stop_enabled = service_policy == 'mandatory'
        if sharing.enabled or stop_enabled:
            parallel_select = False
        stopped = torch.zeros((batch_size, n_agents), dtype=torch.bool, device=emb_uav.device)
        delivery_nodes = torch.arange(
            num_depots + n_pickup, num_depots + num_nodes,
            device=emb_uav.device,
        )
        owner_state = torch.full(
            (batch_size, n_pickup),
            -1,
            dtype=torch.long,
            device=emb_uav.device,
        )
        if delivery_owner is not None:
            supplied_owner = delivery_owner.view(batch_size, n_pickup).to(emb_uav.device)
            if ((supplied_owner != supplied_owner.long()) | (supplied_owner < -1)
                    | (supplied_owner >= n_agents)).any():
                raise ValueError('Invalid immutable delivery carrier')
            owner_state = supplied_owner.long().clone()

        def _record_pickup_owners(selected_nodes):
            selected_pickup = (
                (selected_nodes >= num_depots)
                & (selected_nodes < num_depots + n_pickup)
            )
            if actions:
                selected_pickup &= selected_nodes != actions[-1].squeeze(-1)
            if not selected_pickup.any():
                return
            pickup_ids = (selected_nodes - num_depots).clamp(min=0, max=n_pickup - 1)
            for agent_id in range(n_agents):
                rows = selected_pickup[:, agent_id]
                if rows.any():
                    if (owner_state[rows, pickup_ids[rows, agent_id]] >= 0).any():
                        raise ValueError('Pickup cannot overwrite an existing delivery carrier')
                    if sharing.enabled:
                        sharing.reserve(pickup_ids[:, agent_id], rows, agent_id)
                    owner_state[rows, pickup_ids[rows, agent_id]] = agent_id

        def _apply_delivery_owner_mask(mask_tensor):
            agent_ids = torch.arange(n_agents, device=emb_uav.device).view(1, n_agents, 1)
            owner_exp = owner_state.unsqueeze(1)
            delivery_owner_block = (owner_exp >= 0) & (agent_ids != owner_exp)
            mask_tensor = mask_tensor.clone()
            if agent_depot_allowed is not None:
                mask_tensor[:, :, :num_depots] = mask_tensor[:, :, :num_depots].masked_fill(~agent_depot_allowed, 1)
            mask_tensor[:, :, delivery_nodes] = mask_tensor[:, :, delivery_nodes].masked_fill(
                delivery_owner_block, 1
            )
            if sharing.enabled:
                for agent in range(n_agents):
                    allowed = sharing.allowed(agent)
                    mask_tensor[:, agent, num_depots:num_depots + n_pickup] = (
                        mask_tensor[:, agent, num_depots:num_depots + n_pickup].masked_fill(~allowed, 1))
                    mask_tensor[:, agent, delivery_nodes] = (
                        mask_tensor[:, agent, delivery_nodes].masked_fill(~allowed, 1))
            return mask_tensor

        deadline = time_window.clone()
        deadline[:, num_depots:num_depots + n_pickup] = (
            time_window[:, num_depots + n_pickup:num_depots + num_nodes]
        )

        id_emb_uav = self._agent_ids(self.agent_id_emb_uav, num_uav, batch_size)
        id_emb_adr = self._agent_ids(self.agent_id_emb_adr, num_adr, batch_size)

        onboard_owner = route_data.get('onboard_owner')
        if onboard_owner is None:
            onboard_owner = torch.full_like(owner_state, -1)
            if initial_visited is not None:
                picked = initial_visited[:, num_depots:num_depots + n_pickup] > .5
                delivered = initial_visited[:, num_depots + n_pickup:] > .5
                onboard_owner = torch.where(picked & ~delivered, owner_state, onboard_owner)
        else:
            onboard_owner = torch.as_tensor(onboard_owner, device=emb_uav.device)
            if (onboard_owner.shape != owner_state.shape or onboard_owner.dtype != torch.long
                    or ((onboard_owner < -1) | (onboard_owner >= n_agents)).any()
                    or ((onboard_owner >= 0) & (onboard_owner != owner_state)).any()):
                raise ValueError('Invalid onboard delivery ownership')
        slots = torch.arange(n_agents, device=emb_uav.device).view(1, -1, 1)
        onboard = slots == onboard_owner.unsqueeze(1)
        continuation_check = getattr(self, 'continuation_check', 'off')
        check_onboard = continuation_check == 'onboard'
        if 'onboard_owner' in route_data:
            pickup_load = demands[:, num_depots:num_depots + n_pickup].unsqueeze(1)
            physical_load = (full_capacity - capacity).squeeze(-1)
            mismatch = ((onboard * pickup_load).sum(-1) - physical_load).abs() > 1e-4
            if available_mask is not None:
                mismatch = mismatch & available_mask.bool()
            if mismatch.any():
                raise ValueError('Onboard request ledger disagrees with initial vehicle load')
        uav_onboard = onboard[:, :num_uav, :].to(emb_uav.dtype)
        adr_onboard = onboard[:, num_uav:, :].to(emb_adr.dtype)
        onboard_uav_sum = uav_onboard @ emb_uav[:, delivery_nodes, :]
        onboard_uav_cnt = uav_onboard.sum(-1, keepdim=True)
        onboard_adr_sum = adr_onboard @ emb_adr[:, delivery_nodes, :]
        onboard_adr_cnt = adr_onboard.sum(-1, keepdim=True)

        capacity_init = full_capacity.clone().clamp(min=1e-8)
        E_tensor = _full_bat.clamp(min=1e-8)

        log_ps, actions, time_log = [], [], []
        entropies = []
        i = 0
        # Upper bound per instance; override via --max-decode-steps.
        _max_steps = (2 * max(num_nodes, 1) + n_agents * 2 if max_decode_steps is None
                      else int(max_decode_steps))
        if _max_steps < 1:
            raise ValueError('Decoder guard must be positive')
        stopped_no_progress = False

        _batch_range_uav = torch.arange(batch_size, device=emb_uav.device).unsqueeze(1).expand(-1, num_uav)
        _batch_range_adr = torch.arange(batch_size, device=emb_uav.device).unsqueeze(1).expand(-1, num_adr)
        _deadline_exp    = deadline.unsqueeze(1)
        _sel_buf  = emb_uav.new_zeros(batch_size, n_agents, dtype=torch.long)
        _ask_buf  = emb_uav.new_zeros(batch_size, emb_uav.size(1), dtype=torch.bool)
        _perm_greedy = torch.arange(n_agents, device=emb_uav.device)

        while (mask1[:, :, num_depots:].max(dim=1)[0]).eq(0).any() and i < _max_steps:
            if i == 0:
                if start_node is not None:
                    index = start_node.to(emb_uav.device)
                elif n_depots_uav is not None and n_depots_uav > 0:
                    n_depots_adr_val = num_depots - n_depots_uav
                    uav_depot = torch.remainder(
                        torch.arange(num_uav, device=emb_uav.device), n_depots_uav
                    )
                    adr_depot = n_depots_uav + torch.remainder(
                        torch.arange(num_adr, device=emb_uav.device),
                        max(n_depots_adr_val, 1),
                    )
                    index = torch.cat([uav_depot, adr_depot]).unsqueeze(0).expand(batch_size, -1)
                else:
                    index = torch.remainder(
                        torch.arange(n_agents, device=emb_uav.device), num_depots
                    ).unsqueeze(0).expand(batch_size, -1)
                s_t_uav = torch.gather(emb_uav, 1,
                    index[:, :num_uav].unsqueeze(2).expand(-1, -1, H))
                s_t_adr = torch.gather(emb_adr, 1,
                    index[:, num_uav:].unsqueeze(2).expand(-1, -1, H))
                s_t = torch.cat([s_t_uav, s_t_adr], dim=1)
                actions.append(index.unsqueeze(2))

            remaining = (mask1[:, :, num_depots:].max(dim=1)[0] == 0).float()
            # Urgency weights: earlier deadline → higher weight; T_ref = earliest agent time.
            if available_mask is not None:
                available_bool = available_mask > 0
                time_for_ref = T_t.squeeze(-1).masked_fill(~available_bool, float('inf'))
                no_available = torch.isinf(time_for_ref).all(dim=1, keepdim=True)
                T_ref = time_for_ref.min(dim=1, keepdim=True)[0]
                T_ref = torch.where(no_available, T_t.squeeze(-1).min(dim=1, keepdim=True)[0], T_ref)
            else:
                T_ref = T_t.squeeze(-1).min(dim=1, keepdim=True)[0]       # [B, 1]
            dl_slack = (deadline[:, num_depots:] - T_ref).clamp(min=0.0)
            _urg_pool = self.log_urg_pool_scale.exp().clamp(30.0, 600.0)
            urgency = torch.exp(-dl_slack / _urg_pool) * remaining
            urgency_sum = urgency.sum(1, keepdim=True).clamp(min=1e-8)
            pool_weights = urgency / urgency_sum
            pool_uav = (emb_uav[:, num_depots:, :] * pool_weights.unsqueeze(-1)).sum(1)
            pool_adr = (emb_adr[:, num_depots:, :] * pool_weights.unsqueeze(-1)).sum(1)

            if self.dyn_features == 'extended':
                # First 3 scalars match 'basic' for drop-in init compatibility.
                cap_norm = capacity / capacity_init
                bat_norm = battery / E_tensor
                time_norm = T_t / 120.0
                time_norm_long = (T_t / 240.0).clamp(0.0, 1.0)
                remaining_cap = (1.0 - cap_norm).clamp(0.0, 1.0)
                low_bat_flag = (bat_norm < 0.5).float()
                is_uav = torch.zeros_like(cap_norm)
                is_adr = torch.zeros_like(cap_norm)
                if num_uav > 0:
                    is_uav[:, :num_uav, :] = 1.0
                if num_adr > 0:
                    is_adr[:, num_uav:, :] = 1.0
                onboard_cnt = torch.cat([
                    onboard_uav_cnt,
                    onboard_adr_cnt,
                ], dim=1)
                onboard_frac = (onboard_cnt / capacity_init.clamp(min=1.0)).clamp(0.0, 1.0)
                avail_flag = (
                    available_mask.float().unsqueeze(-1)
                    if available_mask is not None
                    else torch.ones_like(cap_norm)
                )
                dyn = torch.cat([
                    cap_norm,
                    time_norm,
                    bat_norm,
                    is_uav,
                    is_adr,
                    remaining_cap,
                    time_norm_long,
                    low_bat_flag,
                    onboard_frac,
                    avail_flag,
                ], dim=-1)
            else:
                dyn = torch.cat([
                    capacity / capacity_init,
                    T_t / 120.0,
                    battery / E_tensor,
                ], dim=-1)

            onboard_uav = torch.where(
                onboard_uav_cnt > 0,
                onboard_uav_sum / onboard_uav_cnt.clamp(min=1.0),
                self.h_empty_uav.expand(batch_size, num_uav, H),
            )
            onboard_adr = torch.where(
                onboard_adr_cnt > 0,
                onboard_adr_sum / onboard_adr_cnt.clamp(min=1.0),
                self.h_empty_adr.expand(batch_size, num_adr, H),
            )

            ctx_uav_emb = self.Wd_uav(
                torch.cat([s_t[:, :num_uav, :] + id_emb_uav, onboard_uav], dim=-1).view(-1, H * 2)
            ).view(batch_size, num_uav, H)
            ctx_adr_emb = self.Wd_adr(
                torch.cat([s_t[:, num_uav:, :] + id_emb_adr, onboard_adr], dim=-1).view(-1, H * 2)
            ).view(batch_size, num_adr, H)
            quota = dyn.new_full((batch_size, n_agents), -1)
            if recharge_budget is not None:
                quota = recharge_budget.to(dyn)
                if shift_cap is not None:
                    quota = quota / shift_cap
            quota = quota.unsqueeze(-1)
            dyn_uav = self.dyn_proj_uav(dyn[:, :num_uav, :])
            dyn_adr = self.dyn_proj_adr(dyn[:, num_uav:, :])
            ctx_uav = self.ln_uav(ctx_uav_emb + dyn_uav
                                  + quota[:, :num_uav] * self.recharge_state_weight[0])
            ctx_adr = self.ln_adr(ctx_adr_emb + dyn_adr
                                  + quota[:, num_uav:] * self.recharge_state_weight[1])

            if num_uav > 0 and num_adr > 0:
                ctx_uav_pre, ctx_adr_pre = ctx_uav, ctx_adr
                cross_uav, _ = self.cross_attn_uav(ctx_uav_pre, ctx_adr_pre, ctx_adr_pre)
                cross_adr, _ = self.cross_attn_adr(ctx_adr_pre, ctx_uav_pre, ctx_uav_pre)
                ctx_uav = self.ln_cross_uav(ctx_uav + cross_uav)
                ctx_adr = self.ln_cross_adr(ctx_adr + cross_adr)

            dec_uav = (pool_uav.unsqueeze(1).expand(-1, num_uav, -1)
                       + self.fc1_uav(ctx_uav))
            dec_adr = (pool_adr.unsqueeze(1).expand(-1, num_adr, -1)
                       + self.fc1_adr(ctx_adr))
            dec_input = torch.cat([dec_uav, dec_adr], dim=1)

            if i == 0:
                mask_index = index
                if available_mask is not None:
                    mask_index = torch.where(
                        available_mask.bool(),
                        index,
                        torch.zeros_like(index),
                    )
                mask, mask1 = update_mask(demands, capacity, mask_index, mask1, battery, num_uav, E, i,
                                          acc_uav, acc_adr,
                                          edge_attr_d=edge_attr_uav, edge_attr_r=edge_attr_adr,
                                          num_depots=num_depots, T_t=T_t, time_window=time_window,
                                          max_capacity=full_capacity, wind=wind, coordinates=coordinates,
                                          physical_config=config, remaining_deliveries=remaining_deliveries,
                                          allowed_depots=recharge_depots,
                                          mask_adj_uav=mask_adj_uav, mask_adj_adr=mask_adj_adr,
                                          continuation_check=continuation_check, onboard=onboard,
                                          recharge_budget=recharge_budget, available=available_mask)
                mask = _apply_delivery_owner_mask(mask)
                if available_mask is not None:
                    unavailable = (available_mask <= 0).unsqueeze(-1)
                    mask = mask.masked_fill(unavailable, 1)
                    mask[:, :, :num_depots] = mask[:, :, :num_depots].masked_fill(unavailable, 0)

            certified_mask = mask.bool().clone() if check_onboard else None
            q_uav = self.mha_uav(dec_input[:, :num_uav, :], K_uav, V_uav, mask[:, :num_uav, :])
            q_adr = self.mha_adr(dec_input[:, num_uav:, :], K_adr, V_adr, mask[:, num_uav:, :])

            dist_uav = edge_attr_uav[_batch_range_uav, index[:, :num_uav]]
            dist_adr = edge_attr_adr[_batch_range_adr, index[:, num_uav:]]
            dist_bias_uav = self.dist_compat_uav(dist_uav.unsqueeze(-1)).squeeze(-1)
            dist_bias_adr = self.dist_compat_adr(dist_adr.unsqueeze(-1)).squeeze(-1)

            # Learned per-fleet urgency scales (replaces hardcoded 120.0).
            _urg_u = self.log_urg_scale_uav.exp().clamp(30.0, 600.0)
            _urg_a = self.log_urg_scale_adr.exp().clamp(30.0, 600.0)

            urgency_bias_uav = self.urgency_proj_uav(
                ((_deadline_exp - T_t)[:, :num_uav, :] / _urg_u).unsqueeze(-1)
            ).squeeze(-1).masked_fill(mask[:, :num_uav, :].bool(), 0.0)
            urgency_bias_adr = self.urgency_proj_adr(
                ((_deadline_exp - T_t)[:, num_uav:, :] / _urg_a).unsqueeze(-1)
            ).squeeze(-1).masked_fill(mask[:, num_uav:, :].bool(), 0.0)

            # Live feasibility margin: per-agent (deadline - T_t - travel_time).
            live_slack_uav = (
                _deadline_exp
                - T_t[:, :num_uav, :]
                - dist_uav / V_UAV_MAX
            ) / _urg_u
            live_slack_adr = (
                _deadline_exp
                - T_t[:, num_uav:, :]
                - dist_adr / V_ADR_MAX
            ) / _urg_a
            live_margin_bias_uav = self.live_margin_proj_uav(
                live_slack_uav.unsqueeze(-1)
            ).squeeze(-1).masked_fill(mask[:, :num_uav, :].bool(), 0.0)
            live_margin_bias_adr = self.live_margin_proj_adr(
                live_slack_adr.unsqueeze(-1)
            ).squeeze(-1).masked_fill(mask[:, num_uav:, :].bool(), 0.0)

            # Post-visit battery-margin bias m_j = bat_frac − dist_cur→j/R − dist_j→depot/R.
            if self.margin_bias:
                _dist_norm = 50.0
                d_j_to_dep_uav = (
                    edge_attr_uav[:, :, :num_depots].min(dim=-1).values
                )
                d_j_to_dep_adr = (
                    edge_attr_adr[:, :, :num_depots].min(dim=-1).values
                )
                bat_frac_u = (battery[:, :num_uav, :] / _full_bat[:, :num_uav, :].clamp(min=1e-6))
                bat_frac_a = (battery[:, num_uav:, :] / _full_bat[:, num_uav:, :].clamp(min=1e-6))
                margin_u = (
                    bat_frac_u
                    - (dist_uav / _dist_norm)
                    - (d_j_to_dep_uav.unsqueeze(1) / _dist_norm)
                )
                margin_a = (
                    bat_frac_a
                    - (dist_adr / _dist_norm)
                    - (d_j_to_dep_adr.unsqueeze(1) / _dist_norm)
                )
                post_visit_bias_uav = self.post_visit_margin_proj_uav(
                    margin_u.unsqueeze(-1)
                ).squeeze(-1).masked_fill(mask[:, :num_uav, :].bool(), 0.0)
                post_visit_bias_adr = self.post_visit_margin_proj_adr(
                    margin_a.unsqueeze(-1)
                ).squeeze(-1).masked_fill(mask[:, num_uav:, :].bool(), 0.0)
            else:
                post_visit_bias_uav = torch.zeros_like(dist_bias_uav)
                post_visit_bias_adr = torch.zeros_like(dist_bias_adr)

            # Battery hard-mask: zero out infeasible customers; depots stay legal.
            # Uses simulator physics (_vrp_leg_metrics) vs. reserve_kj from config.
            if self.battery_mask:
                _N = edge_attr_uav.size(1)
                _n_pick = (_N - num_depots) // 2
                _targets_all = torch.arange(_N, device=edge_attr_uav.device).view(1, 1, _N).expand(batch_size, n_agents, _N)
                _energy_cur_to_tgt, _, _reachable = _vrp_leg_metrics(
                    demands, time_window, T_t, capacity, num_uav, index, _targets_all,
                    edge_attr_uav, edge_attr_adr, num_depots,
                    max_capacity=full_capacity, wind=wind, coordinates=coordinates,
                    physical_config=physical_config, remaining_deliveries=remaining_deliveries,
                )
                _reserve_kj = _vrp_battery_reserve(_full_bat.squeeze(-1), num_uav, config)
                _bat_kj = battery.squeeze(-1)
                _bm_infeas_leg = ((_bat_kj.unsqueeze(-1) - _energy_cur_to_tgt) < _reserve_kj.unsqueeze(-1))
                _bm_infeas_leg = _bm_infeas_leg | (~_reachable)
                # Lookahead: extend the energy chain with pickup→delivery and
                # delivery→nearest-depot so the mask enforces commit-level feasibility.
                if self.battery_mask_lookahead and _n_pick > 0:
                    _pick_idx = torch.arange(num_depots, num_depots + _n_pick,
                                             device=edge_attr_uav.device)
                    _del_idx = _pick_idx + _n_pick
                    # Pessimistic payload for pickup→delivery (upper-bounds chain energy).
                    _sel_pick = _pick_idx.view(1, -1).expand(batch_size, -1)
                    _tgt_del  = _del_idx.view(1, -1, 1).expand(batch_size, -1, 1)
                    _dummy_T = torch.zeros(batch_size, _n_pick, 1, device=T_t.device)
                    _dummy_max_cap_u = full_capacity[:, :1, 0].expand(batch_size, _n_pick)
                    _dummy_max_cap_a = full_capacity[:, -1:, 0].expand(batch_size, _n_pick)
                    _cap_full_u = torch.zeros(batch_size, _n_pick, device=T_t.device)
                    _cap_full_a = torch.zeros(batch_size, _n_pick, device=T_t.device)
                    _e_p2d_uav, _, _r_p2d_uav = _vrp_leg_metrics(
                        demands, time_window, _dummy_T, _cap_full_u, _n_pick,
                        _sel_pick, _tgt_del, edge_attr_uav, edge_attr_adr, num_depots,
                        max_capacity=_dummy_max_cap_u, wind=wind, coordinates=coordinates,
                        physical_config=physical_config, remaining_deliveries=remaining_deliveries,
                    )
                    _e_p2d_adr, _, _r_p2d_adr = _vrp_leg_metrics(
                        demands, time_window, _dummy_T, _cap_full_a, 0,
                        _sel_pick, _tgt_del, edge_attr_uav, edge_attr_adr, num_depots,
                        max_capacity=_dummy_max_cap_a, wind=wind, coordinates=coordinates,
                        physical_config=physical_config, remaining_deliveries=remaining_deliveries,
                    )
                    _e_p2d_uav = _e_p2d_uav.squeeze(-1)
                    _e_p2d_adr = _e_p2d_adr.squeeze(-1)
                    _r_p2d_uav = _r_p2d_uav.squeeze(-1)
                    _r_p2d_adr = _r_p2d_adr.squeeze(-1)
                    _sel_del = _del_idx.view(1, -1).expand(batch_size, -1)
                    _tgt_dep = torch.arange(num_depots, device=edge_attr_uav.device).view(1, 1, -1).expand(batch_size, _n_pick, -1)
                    _cap_empty_u = _dummy_max_cap_u
                    _cap_empty_a = _dummy_max_cap_a
                    _e_d2dep_uav, _, _r_d2dep_uav = _vrp_leg_metrics(
                        demands, time_window, _dummy_T, _cap_empty_u, _n_pick,
                        _sel_del, _tgt_dep, edge_attr_uav, edge_attr_adr, num_depots,
                        max_capacity=_dummy_max_cap_u, wind=wind, coordinates=coordinates,
                        physical_config=physical_config, remaining_deliveries=remaining_deliveries,
                    )
                    _e_d2dep_adr, _, _r_d2dep_adr = _vrp_leg_metrics(
                        demands, time_window, _dummy_T, _cap_empty_a, 0,
                        _sel_del, _tgt_dep, edge_attr_uav, edge_attr_adr, num_depots,
                        max_capacity=_dummy_max_cap_a, wind=wind, coordinates=coordinates,
                        physical_config=physical_config, remaining_deliveries=remaining_deliveries,
                    )
                    # Cheapest reachable depot per vehicle.
                    _e_d2dep_uav = _e_d2dep_uav.masked_fill(~_r_d2dep_uav, float('inf')).min(dim=-1).values
                    _e_d2dep_adr = _e_d2dep_adr.masked_fill(~_r_d2dep_adr, float('inf')).min(dim=-1).values
                    # Two-leg rule (Section 4.4): complete move AND reach a compatible depot.
                    _e_p2dep_uav, _, _r_p2dep_uav = _vrp_leg_metrics(
                        demands, time_window, _dummy_T, _cap_full_u, _n_pick,
                        _sel_pick, _tgt_dep, edge_attr_uav, edge_attr_adr, num_depots,
                        max_capacity=_dummy_max_cap_u, wind=wind, coordinates=coordinates,
                        physical_config=physical_config, remaining_deliveries=remaining_deliveries,
                    )
                    _e_p2dep_adr, _, _r_p2dep_adr = _vrp_leg_metrics(
                        demands, time_window, _dummy_T, _cap_full_a, 0,
                        _sel_pick, _tgt_dep, edge_attr_uav, edge_attr_adr, num_depots,
                        max_capacity=_dummy_max_cap_a, wind=wind, coordinates=coordinates,
                        physical_config=physical_config, remaining_deliveries=remaining_deliveries,
                    )
                    _e_p2dep_uav = _e_p2dep_uav.masked_fill(~_r_p2dep_uav, float('inf')).min(dim=-1).values
                    _e_p2dep_adr = _e_p2dep_adr.masked_fill(~_r_p2dep_adr, float('inf')).min(dim=-1).values
                    _chain_extra = torch.zeros_like(_energy_cur_to_tgt)
                    _chain_extra[:, :num_uav, num_depots:num_depots + _n_pick] = _e_p2dep_uav.unsqueeze(1)
                    _chain_extra[:, num_uav:, num_depots:num_depots + _n_pick] = _e_p2dep_adr.unsqueeze(1)
                    _chain_extra[:, :num_uav, num_depots + _n_pick:num_depots + 2 * _n_pick] = _e_d2dep_uav.unsqueeze(1)
                    _chain_extra[:, num_uav:, num_depots + _n_pick:num_depots + 2 * _n_pick] = _e_d2dep_adr.unsqueeze(1)
                    _total_energy = _energy_cur_to_tgt + _chain_extra
                    _bm_infeas_chain = ((_bat_kj.unsqueeze(-1) - _total_energy) < _reserve_kj.unsqueeze(-1))
                    if getattr(self, 'use_chain_lookahead_in_mask', True):
                        _bm_infeas_combined = _bm_infeas_leg | _bm_infeas_chain
                    else:
                        _bm_infeas_combined = _bm_infeas_leg
                else:
                    _bm_infeas_combined = _bm_infeas_leg
                _bm_infeas_u = _bm_infeas_combined[:, :num_uav, :]
                _bm_infeas_a = _bm_infeas_combined[:, num_uav:, :]
                # Depot physics-feasibility is allowed to flow through so
                # force-unmask only ever chooses physically-reachable depots.
                _bm_overlay = torch.cat([_bm_infeas_u, _bm_infeas_a], dim=1)
                mask = (mask.bool() | _bm_overlay).to(mask.dtype)
                # Force-unmask depots only when every non-depot is blocked,
                # to avoid the simulator's mask2 depot-persistence trap.
                if self.battery_mask_lookahead and num_depots > 0:
                    _sim_mask_bool = mask.bool()
                    _blocked_u = _sim_mask_bool[:, :num_uav, num_depots:]
                    _blocked_a = _sim_mask_bool[:, num_uav:, num_depots:]
                    _all_infeas_u = _blocked_u.all(dim=-1, keepdim=True)
                    _all_infeas_a = _blocked_a.all(dim=-1, keepdim=True)
                    _all_infeas = torch.cat([_all_infeas_u, _all_infeas_a], dim=1)
                    _all_infeas_expand = _all_infeas.expand(-1, -1, num_depots)
                    _depot_slice = mask[:, :, :num_depots]
                    # Unmask only agent-allowed depots or simulator rejects with 'Forbidden depot'.
                    if recharge_depots is not None:
                        _unmask = _all_infeas_expand & recharge_depots
                    else:
                        _unmask = _all_infeas_expand.clone()
                    # Also AND with mask_adj_* row at the agent's current node so
                    # we never unmask a depot the mode-adjacency explicitly forbids.
                    _adj_depots_combined = torch.ones(
                        batch_size, n_agents, num_depots,
                        device=_unmask.device, dtype=torch.bool,
                    )
                    if mask_adj_uav is not None and num_uav > 0:
                        _adj_u = mask_adj_uav.reshape(batch_size, _N, _N)
                        _cur_u = index[:, :num_uav]
                        _adj_rows_u = _adj_u.gather(
                            1, _cur_u.unsqueeze(-1).expand(-1, -1, _N)
                        )
                        _adj_depots_combined[:, :num_uav] = (
                            _adj_rows_u[:, :, :num_depots] >= 0.5
                        )
                    if mask_adj_adr is not None and num_adr > 0:
                        _adj_a = mask_adj_adr.reshape(batch_size, _N, _N)
                        _cur_a = index[:, num_uav:]
                        _adj_rows_a = _adj_a.gather(
                            1, _cur_a.unsqueeze(-1).expand(-1, -1, _N)
                        )
                        _adj_depots_combined[:, num_uav:] = (
                            _adj_rows_a[:, :, :num_depots] >= 0.5
                        )
                    _unmask = _unmask & _adj_depots_combined
                    # Also gate on single-leg physics so an unreachable depot
                    # is never unmasked (else update_state rejects it).
                    _phys_feas_depots = ~_bm_infeas_leg[:, :, :num_depots]
                    _unmask = _unmask & _phys_feas_depots
                    # Depot-cycle fallback: unmask when all non-depots blocked —
                    # otherwise an agent with zero legal actions stalls the softmax
                    # degenerate case should no longer dominate because most
                    # when all customers are infeasible at step 0.
                    mask[:, :, :num_depots] = torch.where(
                        _unmask,
                        torch.zeros_like(_depot_slice),
                        _depot_slice,
                    )
                # Mandatory-recharge trigger: at low battery allow only depots
                # and the agent's own pending delivery (if carrying).
                if self.battery_mandatory_recharge and num_depots > 0:
                    _bat_frac_u = (battery[:, :num_uav, 0] / _full_bat[:, :num_uav, 0].clamp(min=1e-6))
                    _bat_frac_a = (battery[:, num_uav:, 0] / _full_bat[:, num_uav:, 0].clamp(min=1e-6))
                    _trig_u = _bat_frac_u < (self.battery_mask_reserve_frac_uav + self.battery_mandatory_recharge_buffer)
                    _trig_a = _bat_frac_a < (self.battery_mask_reserve_frac_adr + self.battery_mandatory_recharge_buffer)
                    _trig = torch.cat([_trig_u, _trig_a], dim=1)
                    if _trig.any():
                        _cap_now = capacity.squeeze(-1)
                        _cap_full = full_capacity.squeeze(-1)
                        _carrying = _cap_now < (_cap_full - 1e-5)
                        # allowed = depots ∪ (own delivery while carrying).
                        _allowed = torch.zeros_like(mask, dtype=torch.bool)
                        _allowed[:, :, :num_depots] = True
                        if recharge_depots is not None:
                            _allowed[:, :, :num_depots] &= recharge_depots
                        # Read live `owner_state` (not the immutable `delivery_owner` arg)
                        # so pickups from earlier decoder steps propagate here.
                        if n_pickup > 0:
                            _d_slice_start = num_depots + n_pickup
                            _d_slice_end = num_depots + 2 * n_pickup
                            _ag_ids = torch.arange(n_agents, device=mask.device).view(1, -1, 1)
                            _owner = owner_state.view(batch_size, 1, n_pickup)
                            _owns_delivery = (_owner == _ag_ids)
                            _allowed[:, :, _d_slice_start:_d_slice_end] = (
                                _carrying.unsqueeze(-1) & _owns_delivery
                            )
                        _trig_exp = _trig.unsqueeze(-1).expand_as(mask)
                        mask = torch.where(
                            _trig_exp & ~_allowed,
                            torch.ones_like(mask),
                            mask,
                        )
                        # Force-unmask depots in trig rows, gated by mask_adj_* at
                        # the agent's current node (otherwise the physical-adjacency
                        # guard rejects the move).
                        _adj_depots_mand = torch.ones(
                            batch_size, n_agents, num_depots,
                            device=mask.device, dtype=torch.bool,
                        )
                        if mask_adj_uav is not None and num_uav > 0:
                            _adj_u_m = mask_adj_uav.reshape(batch_size, _N, _N)
                            _cur_u_m = index[:, :num_uav]
                            _adj_rows_u_m = _adj_u_m.gather(
                                1, _cur_u_m.unsqueeze(-1).expand(-1, -1, _N)
                            )
                            _adj_depots_mand[:, :num_uav] = (
                                _adj_rows_u_m[:, :, :num_depots] >= 0.5
                            )
                        if mask_adj_adr is not None and num_adr > 0:
                            _adj_a_m = mask_adj_adr.reshape(batch_size, _N, _N)
                            _cur_a_m = index[:, num_uav:]
                            _adj_rows_a_m = _adj_a_m.gather(
                                1, _cur_a_m.unsqueeze(-1).expand(-1, -1, _N)
                            )
                            _adj_depots_mand[:, num_uav:] = (
                                _adj_rows_a_m[:, :, :num_depots] >= 0.5
                            )
                        mask[:, :, :num_depots] = torch.where(
                            _trig.unsqueeze(-1).expand(-1, -1, num_depots)
                              & _allowed[:, :, :num_depots]
                              & _adj_depots_mand,
                            torch.zeros_like(mask[:, :, :num_depots]),
                            mask[:, :, :num_depots],
                        )

                # Paper-strict depot-as-fallback overlay (Section 4.4).
                if self.paper_strict_depot_mask and num_depots > 0:
                    _customer_slice = mask[:, :, num_depots:]
                    _any_cust_feas = (~_customer_slice.bool()).any(dim=-1, keepdim=True)
                    _ps_bat_frac_u = (battery[:, :num_uav, 0]
                                      / _full_bat[:, :num_uav, 0].clamp(min=1e-6))
                    _ps_bat_frac_a = (battery[:, num_uav:, 0]
                                      / _full_bat[:, num_uav:, 0].clamp(min=1e-6))
                    _ps_safe_u = _ps_bat_frac_u > (
                        self.battery_mask_reserve_frac_uav
                        + self.paper_strict_depot_margin)
                    _ps_safe_a = _ps_bat_frac_a > (
                        self.battery_mask_reserve_frac_adr
                        + self.paper_strict_depot_margin)
                    _ps_safe = torch.cat([_ps_safe_u, _ps_safe_a], dim=1).unsqueeze(-1)
                    _ps_should_mask = _any_cust_feas & _ps_safe
                    _ps_should_mask_dep = _ps_should_mask.expand(-1, -1, num_depots)
                    mask = mask.clone()
                    mask[:, :, :num_depots] = torch.where(
                        _ps_should_mask_dep,
                        torch.ones_like(mask[:, :, :num_depots]),
                        mask[:, :, :num_depots],
                    )
                    try:
                        self._diag['dyn_depot_fires'] += int(
                            _ps_should_mask.sum().item())
                        self._diag['dyn_depot_total_slots'] += int(
                            _ps_should_mask.numel())
                    except Exception:
                        pass

            if certified_mask is not None:
                mask = mask.masked_fill(certified_mask, 1)
            if self.decoder_biases:
                u_uav = (self._compatibility(q_uav, K_all_uav, mask[:, :num_uav, :], T)
                         + dist_bias_uav.masked_fill(mask[:, :num_uav, :].bool(), 0.0)
                         + urgency_bias_uav
                         + live_margin_bias_uav
                         + post_visit_bias_uav)
                u_adr = (self._compatibility(q_adr, K_all_adr, mask[:, num_uav:, :], T)
                         + dist_bias_adr.masked_fill(mask[:, num_uav:, :].bool(), 0.0)
                         + urgency_bias_adr
                         + live_margin_bias_adr
                         + post_visit_bias_adr)
            else:
                u_uav = self._compatibility(q_uav, K_all_uav, mask[:, :num_uav, :], T)
                u_adr = self._compatibility(q_adr, K_all_adr, mask[:, num_uav:, :], T)
            u = torch.cat([u_uav, u_adr], dim=1)
            if available_mask is not None:
                unavailable = (available_mask <= 0).unsqueeze(-1)
                u = u.masked_fill(unavailable, float('-inf'))
                u[:, :, 0] = u[:, :, 0].masked_fill(unavailable.squeeze(-1), 0.0)

            if parallel_select:
                index, log_p, step_entropy = _parallel_select(
                    u, num_depots, greedy, current_nodes=actions[-1].squeeze(2))
                if _DECODER_TRACE_SINK is not None:
                    with torch.no_grad():
                        _legal_p = (u > -1e8)
                        _lp_p = F.log_softmax(u.masked_fill(~_legal_p, float('-inf')).clamp(min=-1e9), dim=-1)
                        _trace_push({
                            'path': 'parallel',
                            'episode_step': int(i),
                            'num_uav': int(num_uav),
                            'num_adr': int(num_adr),
                            'num_depots': int(num_depots),
                            'num_pickup': int(n_pickup),
                            'greedy': bool(greedy),
                            'legal_mask': _legal_p.detach().cpu().clone(),
                            'log_probs': _lp_p.detach().cpu().clone(),
                            'probs': _lp_p.exp().detach().cpu().clone(),
                            'selected': index.detach().cpu().clone(),
                            'argmax_greedy': u.argmax(dim=-1).detach().cpu().clone(),
                        })
            else:
                # Reuse pre-allocated buffers (avoid per-step malloc)
                _sel_buf.zero_()
                _ask_buf.zero_()
                log_p_list = [None] * n_agents
                entropy_list = [None] * n_agents
                if stop_enabled:
                    stop_scores = self._stop_logits(ctx_uav, ctx_adr)
                    can_stop = (capacity.squeeze(-1) >= full_capacity.squeeze(-1) - 1e-5)
                    can_stop = can_stop & ~stopped
                    if available_mask is not None:
                        can_stop = can_stop & available_mask.bool()
                perm = (_perm_greedy if greedy
                        else torch.randperm(n_agents, device=emb_uav.device))
                for step_i in range(n_agents):
                    ag = int(perm[step_i].item())
                    mu = u[:, ag, :].masked_fill(_ask_buf.clone(), float('-inf'))
                    if sharing.enabled:
                        mu[:, num_depots:num_depots + n_pickup] = (
                            mu[:, num_depots:num_depots + n_pickup].masked_fill(~sharing.allowed(ag), float('-inf')))
                    inactive = stopped[:, ag].clone()
                    if available_mask is not None:
                        inactive = inactive | ~available_mask[:, ag].bool()
                    mu = mu.masked_fill(inactive.unsqueeze(1), float('-inf'))
                    all_masked_ag = (mu == float('-inf')).all(dim=-1, keepdim=True)
                    # Blocked agent stays put; no fabricated depot return.
                    mu_safe = mu.clone()
                    stay = actions[-1][:, ag, 0]
                    safe_stay = mu_safe.gather(1, stay.unsqueeze(1))
                    mu_safe = mu_safe.scatter(1, stay.unsqueeze(1), torch.where(all_masked_ag, 0.0, safe_stay))
                    if stop_enabled:
                        stop_legal = can_stop[:, ag] & ~all_masked_ag.squeeze(1)
                        stop_score = stop_scores[:, ag].masked_fill(~stop_legal, float('-inf'))
                        mu_safe = torch.cat((mu_safe, stop_score.unsqueeze(1)), dim=1)
                    log_probs = F.log_softmax(mu_safe.clamp(min=-1e9), dim=-1)
                    if not greedy:
                        choice = torch.multinomial(log_probs.detach().exp().float(), 1).squeeze(1)
                    else:
                        choice = mu_safe.max(dim=-1)[1]
                    if diagnostic_actions is not None and (i, ag) in diagnostic_actions:
                        forced = int(diagnostic_actions[i, ag])
                        if not (0 <= forced < mu_safe.size(1)) or not torch.isfinite(mu_safe[:, forced]).all():
                            raise ValueError('Requested diagnostic alternative is masked')
                        choice = torch.full_like(choice, forced)
                    selected_stop = choice == emb_uav.size(1)
                    mi = torch.where(selected_stop, stay, choice)
                    _sel_buf[:, ag] = mi
                    stopped[:, ag] = stopped[:, ag] | selected_stop
                    if _DECODER_TRACE_SINK is not None:
                        with torch.no_grad():
                            _legal = (mu_safe > -1e8)
                            _probs_rec = log_probs.detach().exp().float()
                            _argmax_greedy = mu_safe.argmax(dim=-1)
                            _state_pos = actions[-1][:, ag, 0].detach().cpu().clone() if len(actions) > 0 else None
                            _state_bat = battery[:, ag, 0].detach().cpu().clone()
                            _state_cap = capacity[:, ag, 0].detach().cpu().clone()
                            _state_T   = T_t[:, ag, 0].detach().cpu().clone()
                            _state_full_bat = _full_bat[:, ag, 0].detach().cpu().clone()
                            _state_full_cap = full_capacity[:, ag, 0].detach().cpu().clone()
                            _trace_push({
                                'path': 'sequential',
                                'episode_step': int(i),
                                'auction_step': int(step_i),
                                'agent': int(ag),
                                'agent_type': 'uav' if ag < num_uav else 'adr',
                                'num_uav': int(num_uav),
                                'num_adr': int(num_adr),
                                'num_depots': int(num_depots),
                                'num_pickup': int(n_pickup),
                                'greedy': bool(greedy),
                                'legal_mask': _legal.detach().cpu().clone(),
                                'log_probs': log_probs.detach().cpu().clone(),
                                'probs': _probs_rec.detach().cpu().clone(),
                                'selected': mi.detach().cpu().clone(),
                                'argmax_greedy': _argmax_greedy.detach().cpu().clone(),
                                'all_masked': all_masked_ag.detach().cpu().clone(),
                                'stop_selected': selected_stop.detach().cpu().clone(),
                                'inactive': inactive.detach().cpu().clone(),
                                'state_pos': _state_pos,
                                'state_bat_kj': _state_bat,
                                'state_cap_kg': _state_cap,
                                'state_T_min': _state_T,
                                'state_full_bat_kj': _state_full_bat,
                                'state_full_cap_kg': _state_full_cap,
                            })
                    if sharing.enabled:
                        picking = ((mi >= num_depots) & (mi < num_depots + n_pickup)
                                   & (mi != stay))
                        sharing.reserve((mi - num_depots).clamp(0, max(n_pickup - 1, 0)), picking, ag)
                    lp_action = torch.where(selected_stop, choice, mi)
                    lp = log_probs.gather(1, lp_action.unsqueeze(1)).squeeze(1)
                    lp = lp.masked_fill(all_masked_ag.squeeze(-1), 0.0)
                    log_p_list[ag] = lp
                    _probs_ag = log_probs.exp()
                    _plogp_ag = _probs_ag * log_probs
                    _plogp_ag = torch.where(torch.isfinite(_plogp_ag),
                                              _plogp_ag, torch.zeros_like(_plogp_ag))
                    _ent_ag = -_plogp_ag.sum(dim=-1)
                    _ent_ag = _ent_ag.masked_fill(all_masked_ag.squeeze(-1), 0.0)
                    entropy_list[ag] = _ent_ag
                    reserve = (mi != stay) & (mi >= num_depots) & ~inactive
                    _ask_buf[reserve, mi[reserve]] = True

                log_p = torch.stack(log_p_list, dim=1)
                step_entropy = torch.stack(entropy_list, dim=1)
                index = _sel_buf.clone()
            if available_mask is not None:
                index = torch.where(
                    available_mask.bool(),
                    index,
                    actions[-1].squeeze(2),
                )
            previous_index = actions[-1].squeeze(-1)
            did_move = index != previous_index
            _record_pickup_owners(index)
            actions.append(index.unsqueeze(2))

            # Depot-pick diagnostic (moves only).
            try:
                _depot_sel = (index < num_depots) & did_move
                if _depot_sel.any():
                    self._diag['depot_pick_count'] += int(_depot_sel.sum().item())
                    self._diag['depot_pick_sum_tour_len'] += int(
                        _tour_step[_depot_sel].sum().item())
                    _legal_cust = (~mask[:, :, num_depots:].bool()).sum(dim=-1)
                    self._diag['legal_cust_at_depot_pick_sum'] += int(
                        _legal_cust[_depot_sel].sum().item())
                # Advance tour counter for agents that made a real move.
                _tour_step = _tour_step + did_move.long()
            except Exception:
                pass

            is_done = (mask1[:, :, num_depots:].max(dim=1)[0].sum(1).unsqueeze(1)
                       .expand(batch_size, n_agents) >= (emb_uav.size(1) - num_depots)).float()
            log_p = log_p * (1.0 - is_done)
            step_entropy = step_entropy * (1.0 - is_done)
            if available_mask is not None:
                log_p = log_p * available_mask
                step_entropy = step_entropy * available_mask
            log_ps.append(log_p.unsqueeze(2))
            entropies.append(step_entropy.unsqueeze(2))

            capacity_prev, T_t_prev, battery_prev = capacity, T_t, battery
            capacity, T_t, battery = update_state(
                demands, time_window, battery, T_t, capacity, E,
                num_uav, actions, edge_attr_uav, edge_attr_adr,
                num_depots=num_depots,
                max_capacity=full_capacity, wind=wind, coordinates=coordinates,
                physical_config=config, remaining_deliveries=remaining_deliveries,
                allowed_depots=recharge_depots,
                strict=service_policy == 'mandatory',
            )
            remaining_deliveries = pending_after_moves(
                remaining_deliveries, index, did_move, num_depots)
            if check_onboard and n_pickup:
                customer = did_move & (index >= num_depots)
                picked = index < num_depots + n_pickup
                ids = (index - num_depots).remainder(n_pickup).unsqueeze(-1)
                changed = torch.where(customer, picked, onboard.gather(2, ids).squeeze(-1))
                onboard.scatter_(2, ids, changed.unsqueeze(-1))
            if recharge_budget is not None:
                recharge_budget = recharge_budget - ((index < num_depots) & did_move).long()
                recharge_depots = recharge_depot_permissions(
                    agent_depot_allowed, recharge_budget, num_depots)
            if available_mask is not None:
                available_state = available_mask.unsqueeze(-1).bool()
                capacity = torch.where(available_state.squeeze(-1), capacity, capacity_prev.squeeze(2))
                T_t = torch.where(available_state.squeeze(-1), T_t, T_t_prev.squeeze(2))
                battery = torch.where(available_state.squeeze(-1), battery, battery_prev.squeeze(2))
            capacity = capacity.unsqueeze(2)
            T_t = T_t.unsqueeze(2)
            battery = battery.unsqueeze(2)
            time_log.append(T_t)

            mask_index = index
            if available_mask is not None:
                mask_index = torch.where(
                    available_mask.bool(),
                    index,
                    torch.zeros_like(index),
                )
            mask, mask1 = update_mask(demands, capacity, mask_index, mask1, battery, num_uav, E, i,
                                      acc_uav, acc_adr,
                                      edge_attr_d=edge_attr_uav, edge_attr_r=edge_attr_adr,
                                      num_depots=num_depots, T_t=T_t, time_window=time_window,
                                      max_capacity=full_capacity, wind=wind, coordinates=coordinates,
                                      physical_config=config, remaining_deliveries=remaining_deliveries,
                                      allowed_depots=recharge_depots,
                                      mask_adj_uav=mask_adj_uav, mask_adj_adr=mask_adj_adr,
                                      continuation_check=continuation_check, onboard=onboard,
                                      recharge_budget=recharge_budget, available=available_mask)
            mask = _apply_delivery_owner_mask(mask)
            if available_mask is not None:
                unavailable = (available_mask <= 0).unsqueeze(-1)
                mask = mask.masked_fill(unavailable, 1)
                mask[:, :, :num_depots] = mask[:, :, :num_depots].masked_fill(unavailable, 0)

            s_t_uav = torch.gather(emb_uav, 1,
                index[:, :num_uav].unsqueeze(2).expand(-1, -1, H))
            s_t_adr = torch.gather(emb_adr, 1,
                index[:, num_uav:].unsqueeze(2).expand(-1, -1, H))
            s_t = torch.cat([s_t_uav, s_t_adr], dim=1)

            idx_uav = index[:, :num_uav]
            is_pickup_uav = (idx_uav >= num_depots) & (idx_uav < num_depots + n_pickup) & did_move[:, :num_uav]
            is_delivery_uav = (idx_uav >= (num_depots + n_pickup)) & did_move[:, :num_uav]

            delivery_idx_uav = (idx_uav + n_pickup).clamp(max=emb_uav.size(1) - 1)
            paired_del_emb_uav = torch.gather(
                emb_uav, 1, delivery_idx_uav.unsqueeze(-1).expand(-1, -1, H)
            )
            cur_emb_uav = torch.gather(emb_uav, 1, idx_uav.unsqueeze(-1).expand(-1, -1, H))

            p_u = is_pickup_uav.unsqueeze(-1)
            d_u = is_delivery_uav.unsqueeze(-1)
            onboard_uav_sum = (onboard_uav_sum
                               + paired_del_emb_uav * p_u.float()
                               - cur_emb_uav * d_u.float())
            onboard_uav_cnt = (onboard_uav_cnt
                               + p_u.float()
                               - d_u.float()).clamp(min=0.0)

            idx_adr = index[:, num_uav:]
            is_pickup_adr = (idx_adr >= num_depots) & (idx_adr < num_depots + n_pickup) & did_move[:, num_uav:]
            is_delivery_adr = (idx_adr >= (num_depots + n_pickup)) & did_move[:, num_uav:]

            delivery_idx_adr = (idx_adr + n_pickup).clamp(max=emb_adr.size(1) - 1)
            paired_del_emb_adr = torch.gather(
                emb_adr, 1, delivery_idx_adr.unsqueeze(-1).expand(-1, -1, H)
            )
            cur_emb_adr = torch.gather(emb_adr, 1, idx_adr.unsqueeze(-1).expand(-1, -1, H))

            p_a = is_pickup_adr.unsqueeze(-1)
            d_a = is_delivery_adr.unsqueeze(-1)
            onboard_adr_sum = (onboard_adr_sum
                               + paired_del_emb_adr * p_a.float()
                               - cur_emb_adr * d_a.float())
            onboard_adr_cnt = (onboard_adr_cnt
                               + p_a.float()
                               - d_a.float()).clamp(min=0.0)

            i += 1
            if not did_move.any():
                stopped_no_progress = True
                break

        complete = (mask1[:, :, num_depots:].max(dim=1)[0] > .5).all(dim=1)
        try:
            self._diag['last_episode_steps'] = int(i)
            self._diag['last_max_steps'] = int(_max_steps)
            if i >= _max_steps:
                self._diag['hit_max_steps_batches'] = self._diag.get('hit_max_steps_batches', 0) + 1
        except Exception:
            pass
        eligible_agents = (available_mask.bool() if available_mask is not None
                           else torch.ones_like(stopped))
        all_stopped = eligible_agents.any(1) & (stopped | ~eligible_agents).all(1)
        self.last_decode_status = dict(complete=complete.detach(), stopped=stopped.detach(),
                                      all_available_stopped=all_stopped.detach(),
                                      reason='all_stopped' if stopped_no_progress and all_stopped.all() else
                                      'no_progress' if stopped_no_progress else
                                      'step_guard' if not complete.all() else 'complete',
                                      steps=i, max_steps=_max_steps,
                                      scope='rolling_prefix' if allow_partial else 'complete_episode')
        if service_policy == 'mandatory' and not allow_partial and not complete.all():
            raise IncompleteTensorRouteError('Mandatory decoder ' + self.last_decode_status['reason']
                                             + ': active deliveries remain; this is not an infeasibility proof')
        if not actions:
            if start_node is not None:
                first = start_node.to(emb_uav.device).long()
            elif n_depots_uav:
                first = torch.cat([torch.arange(num_uav, device=emb_uav.device) % n_depots_uav,
                                   n_depots_uav + torch.arange(num_adr, device=emb_uav.device)
                                   % max(num_depots - n_depots_uav, 1)]).expand(batch_size, -1)
            else:
                first = (torch.arange(n_agents, device=emb_uav.device) % num_depots).expand(batch_size, -1)
            empty_logp = emb_uav.new_zeros(batch_size, n_agents)
            if return_entropy:
                empty_ent = emb_uav.new_zeros(batch_size, n_agents)
                return first.unsqueeze(-1), empty_logp, T_t[:, :, :0], empty_ent
            return first.unsqueeze(-1), empty_logp, T_t[:, :, :0]
        actions = torch.cat(actions, dim=2)
        Time = torch.cat(time_log, dim=2)
        log_p_out = torch.cat(log_ps, dim=2).sum(dim=2)
        if return_entropy:
            entropy_out = torch.cat(entropies, dim=2).sum(dim=2) if entropies \
                else emb_uav.new_zeros(batch_size, n_agents)
            return actions, log_p_out, Time, entropy_out
        return actions, log_p_out, Time

class Model(nn.Module):
    # continuation_check writes are auto-propagated to the decoder.
    @property
    def continuation_check(self):
        return getattr(self, '_continuation_check', 'off')

    @continuation_check.setter
    def continuation_check(self, value):
        self._continuation_check = value
        decoder = getattr(self, 'decoder', None)
        if decoder is not None:
            decoder.continuation_check = value

    def __init__(self, input_node_dim=11, hidden_node_dim=128,
                 input_edge_dim=4, hidden_edge_dim=16, conv_layers=3,
                 arch='hetgat',
                 edge_features: bool = True,
                 decoder_biases: bool = True,
                 margin_bias: bool = False,
                 critic: bool = False,
                 dyn_features: str = 'basic',
                 battery_mask: bool = False,
                 battery_mask_reserve_frac: float = 0.10,
                 battery_mask_reserve_frac_uav: float = None,
                 battery_mask_reserve_frac_adr: float = None,
                 battery_mask_lookahead: bool = False,
                 battery_mandatory_recharge: bool = False,
                 battery_mandatory_recharge_buffer: float = 0.10,
                 paper_strict_depot_mask: bool = True,
                 paper_strict_depot_margin: float = 0.10,
                 encoder_attn_mask: bool = True,
                 use_chain_lookahead_in_mask: bool = True,
                 continuation_check: str = 'off'):
        super().__init__()
        self._continuation_check = continuation_check
        self.edge_features = bool(edge_features)
        self.decoder_biases = bool(decoder_biases)
        self.margin_bias = bool(margin_bias)
        self.paper_strict_depot_mask = bool(paper_strict_depot_mask)
        self.paper_strict_depot_margin = float(paper_strict_depot_margin)
        self.encoder_attn_mask = bool(encoder_attn_mask)
        self.use_chain_lookahead_in_mask = bool(use_chain_lookahead_in_mask)
        self.battery_mask = bool(battery_mask)
        self.battery_mask_reserve_frac = float(battery_mask_reserve_frac)
        # Per-fleet overrides fall back to the single-scalar reserve (back-compat).
        self.battery_mask_reserve_frac_uav = (
            float(battery_mask_reserve_frac_uav)
            if battery_mask_reserve_frac_uav is not None else None
        )
        self.battery_mask_reserve_frac_adr = (
            float(battery_mask_reserve_frac_adr)
            if battery_mask_reserve_frac_adr is not None else None
        )
        self.battery_mask_lookahead = bool(battery_mask_lookahead)
        self.battery_mandatory_recharge = bool(battery_mandatory_recharge)
        self.battery_mandatory_recharge_buffer = float(battery_mandatory_recharge_buffer)
        # Optional value head (constructed only when critic=True so old ckpts load clean).
        self.critic = bool(critic)
        self.dyn_features = dyn_features
        if arch == 'simplegat':
            self.encoder = SimpleGatEncoder(input_node_dim, hidden_node_dim,
                                            input_edge_dim, hidden_edge_dim, conv_layers,
                                            encoder_attn_mask=self.encoder_attn_mask)
        else:
            self.encoder = Encoder(input_node_dim, hidden_node_dim,
                                   input_edge_dim, hidden_edge_dim, conv_layers,
                                   encoder_attn_mask=self.encoder_attn_mask)
        self.decoder = CoopDecoder(hidden_node_dim, hidden_node_dim,
                                    decoder_biases=self.decoder_biases,
                                    margin_bias=self.margin_bias,
                                    dyn_features=self.dyn_features,
                                    battery_mask=self.battery_mask,
                                    battery_mask_reserve_frac=self.battery_mask_reserve_frac,
                                    battery_mask_reserve_frac_uav=self.battery_mask_reserve_frac_uav,
                                    battery_mask_reserve_frac_adr=self.battery_mask_reserve_frac_adr,
                                    battery_mask_lookahead=self.battery_mask_lookahead,
                                    battery_mandatory_recharge=self.battery_mandatory_recharge,
                                    battery_mandatory_recharge_buffer=self.battery_mandatory_recharge_buffer,
                                    paper_strict_depot_mask=self.paper_strict_depot_mask,
                                    paper_strict_depot_margin=self.paper_strict_depot_margin,
                                    use_chain_lookahead_in_mask=self.use_chain_lookahead_in_mask)
        self.decoder.continuation_check = self._continuation_check
        if self.critic:
            self.value_head = nn.Sequential(
                nn.Linear(hidden_node_dim * 2, hidden_node_dim),
                nn.ReLU(),
                nn.Linear(hidden_node_dim, 1),
            )

    def forward(self, data, num_uav, num_adr, greedy=False, T=1.0,
                checkpoint_encoder=False, training=False, parallel_select=False,
                initial_visited=None, start_node=None, start_time=None,
                start_battery=None, start_capacity=None, available_mask=None,
                delivery_owner=None, return_entropy=False):
        from reward import native_physical_configuration
        config = native_physical_configuration(data)
        import torch.utils.checkpoint as cp
        if not self.edge_features:
            # Zero only encoder edge features; dist_bias still sees edge_attr_d/r.
            data = dict(data)
            for key in ('edge_attr_uav', 'edge_attr_adr'):
                if key in data and torch.is_tensor(data[key]):
                    data[key] = torch.zeros_like(data[key])
        if checkpoint_encoder and training:
            # Pack initial_visited via data dict so it survives gradient checkpointing.
            _ckpt_data = dict(data)
            if initial_visited is not None:
                _ckpt_data['_iv'] = initial_visited
            x_uav, x_adr = cp.checkpoint(self.encoder, _ckpt_data, use_reentrant=False)
        else:
            x_uav, x_adr = self.encoder(data, initial_visited=initial_visited)

        batch_size = data['x'].shape[0]
        nd = data.get('n_depots', None)
        if nd is not None:
            num_depots = int(nd[0]) if hasattr(nd, '__len__') else int(nd)
        else:
            num_depots = int((data['demand'] == 0).sum().item() // batch_size)
        num_nodes = x_uav.shape[1] - num_depots

        nd_uav = data.get('n_depots_uav', None)
        n_depots_uav = (int(nd_uav[0]) if hasattr(nd_uav, '__len__') else int(nd_uav)) if nd_uav is not None else None

        ea_uav = data['edge_attr_d']
        ea_adr = data['edge_attr_r']

        acc_uav = data['x'][:, :, 8]
        acc_adr = data['x'][:, :, 9]
        if start_node is None:
            start_node = data.get('start_node', None)
        if start_time is None:
            start_time = data.get('start_time', None)
        if start_battery is None:
            start_battery = data.get('start_battery', None)
        if start_capacity is None:
            start_capacity = data.get('start_capacity', None)
        if available_mask is None:
            available_mask = data.get('available_mask', None)
        if delivery_owner is None:
            delivery_owner = data.get('delivery_owner', None)

        _decoder_out = self.decoder(
            x_uav, x_adr,
            data['capacity'], data['demand'], data['battery'],
            data['time_window'], num_depots, num_nodes,
            num_uav, num_adr, ea_uav, ea_adr, T, greedy,
            acc_uav=acc_uav, acc_adr=acc_adr,
            parallel_select=parallel_select,
            n_depots_uav=n_depots_uav,
            initial_visited=initial_visited,
            start_node=start_node,
            start_time=start_time,
            start_battery=start_battery,
            start_capacity=start_capacity,
            available_mask=available_mask,
            delivery_owner=delivery_owner,
            wind=data.get('wind'), coordinates=data.get('coordinates'),
            sharing_context=data,
            service_policy=data.get('service_policy', 'legacy_optional'),
            allow_partial=data.get('route_scope') == 'rolling_prefix',
            max_decode_steps=data.get('max_decode_steps'),
            agent_depot_allowed=data.get('agent_depot_allowed'),
            physical_config=config,
            remaining_deliveries=data.get('remaining_deliveries'),
            mask_adj_uav=data.get('mask_adjacency_uav'),
            mask_adj_adr=data.get('mask_adjacency_adr'),
            return_entropy=return_entropy,
        )
        if return_entropy:
            actions, log_p, time, entropy = _decoder_out
        else:
            actions, log_p, time = _decoder_out
        if data.get('service_policy') == 'mandatory':
            from reward import replay_mandatory_trajectory
            if data.get('route_scope', 'complete_episode') not in ('complete_episode', 'rolling_prefix'):
                raise ValueError('Unknown mandatory route valuation scope')
            replay_mandatory_trajectory(
                data, actions, time, num_uav, num_depots,
                initial_visited=initial_visited,
                start_time=start_time, start_battery=start_battery, start_capacity=start_capacity,
                allow_partial=data.get('route_scope') == 'rolling_prefix',
            )
        # V(s) from pooled encoder; .detach() keeps critic MSE from corrupting
        # the shared encoder's policy gradient (standard POMO/Kool practice).
        value = None
        if self.critic:
            pooled_uav = x_uav.mean(dim=1).detach()
            pooled_adr = x_adr.mean(dim=1).detach()
            value = self.value_head(torch.cat([pooled_uav, pooled_adr], dim=-1)).squeeze(-1)
        if return_entropy:
            if self.critic:
                return actions, log_p, time, entropy, value
            return actions, log_p, time, entropy
        if self.critic:
            return actions, log_p, time, value
        return actions, log_p, time

from vrpUpdate import recharge_depot_permissions, tensor_recharge_budget, update_mask, update_state
