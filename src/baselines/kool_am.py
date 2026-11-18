"""Kool et al. 2019 Attention Model, adapted to the bimodal CE-CPDPTW.

Reference: W. Kool, H. van Hoof, M. Welling. "Attention, Learn to Solve Routing
Problems!" ICLR 2019.

Design intent — this is a *learned-routing baseline*, not a competitor
architecture. It isolates the contribution of the HetGAT encoder by sharing
*everything else* with the main Model: same batch dict interface, same
``CoopDecoder`` physics stack (mandatory-service checks, battery, time
windows, capacity, pickup/delivery pairing, mode-specific access), same
training loss, same reward. The only architectural difference is the encoder:

- **HetGAT encoder** (``VRP_Actor.Encoder``): heterogeneous UAV/ADR streams
  with mode-specific tokens, mode-specific GAT convs, mode-specific edge
  features, cross-fleet interaction via shared parameters.
- **Kool AM encoder** (this file): single-stream multi-head self-attention
  over node features, no edge features, no mode-specific message passing.
  Duplicates the output tensor to the two streams the downstream decoder
  expects. The encoder is fleet-homogeneous; the heterogeneity enters only
  through the per-step feasibility masks the decoder already enforces.

Train with the same ``VRP_Rollout_train.py`` harness by substituting
``KoolAMModel`` for ``Model`` at construction time.
"""
from __future__ import annotations

import math
from pathlib import Path
import sys

import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from VRP_Actor import CoopDecoder  # shared physics-aware decoder


class _KoolMHA(nn.Module):
    """Multi-head self-attention block in the Kool et al. 2019 style.

    Pre-LN residual block: ``x + MHA(LN(x)) → x + FF(LN(x))``. Deterministic
    (no dropout) to keep validation reproducible.
    """

    def __init__(self, hidden_dim: int, n_heads: int = 8, ff_dim: int = 512):
        super().__init__()
        self.n_heads = n_heads
        self.head_dim = hidden_dim // n_heads
        assert hidden_dim % n_heads == 0, (
            f'hidden_dim={hidden_dim} not divisible by n_heads={n_heads}')
        self.norm = 1.0 / math.sqrt(self.head_dim)

        self.q = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.k = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.v = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.out = nn.Linear(hidden_dim, hidden_dim, bias=False)

        self.ln1 = nn.LayerNorm(hidden_dim)
        self.ln2 = nn.LayerNorm(hidden_dim)

        self.ff = nn.Sequential(
            nn.Linear(hidden_dim, ff_dim),
            nn.ReLU(),
            nn.Linear(ff_dim, hidden_dim),
        )

    def _mha(self, x: torch.Tensor) -> torch.Tensor:
        B, N, H = x.shape
        Q = self.q(x).view(B, N, self.n_heads, self.head_dim).transpose(1, 2)
        K = self.k(x).view(B, N, self.n_heads, self.head_dim).transpose(1, 2)
        V = self.v(x).view(B, N, self.n_heads, self.head_dim).transpose(1, 2)
        compat = self.norm * torch.matmul(Q, K.transpose(2, 3))
        attn = F.softmax(compat, dim=-1)
        out = torch.matmul(attn, V).transpose(1, 2).contiguous().view(B, N, H)
        return self.out(out)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self._mha(self.ln1(x))
        x = x + self.ff(self.ln2(x))
        return x


class KoolAMEncoder(nn.Module):
    """Fleet-homogeneous single-stream encoder (Kool et al. 2019).

    Mirrors ``VRP_Actor.Encoder``'s input contract so downstream code is
    unchanged:

    - Reads ``data['x']`` of shape ``[B, N, input_node_dim]`` (depot block first,
      then n_pickup pickups, then n_pickup deliveries).
    - Returns ``(x_uav, x_adr)`` each ``[B, N, hidden_node_dim]`` — the two
      streams are identical. Any downstream mode distinction must come from
      the decoder's masks, not this encoder.

    Deliberately omitted vs. HetGAT ``Encoder``:
    - Separate UAV/ADR mode-specific token or embedding stacks.
    - Edge features (``edge_attr_uav`` / ``edge_attr_adr``).
    - Adaptive-time per-stream encoding.
    - Mode-adjacency masks within attention (pickups and deliveries are
      attended uniformly).
    """

    def __init__(self, input_node_dim: int = 11, hidden_node_dim: int = 128,
                 n_heads: int = 8, n_layers: int = 3, ff_dim: int = 512):
        super().__init__()
        self.input_node_dim = int(input_node_dim)
        self.hidden_node_dim = int(hidden_node_dim)

        # Kool's original uses separate linears per node *type* (depot vs
        # customer); we keep that pattern because the batch dict's node
        # layout is depot-pickup-delivery-partitioned and the pickup/delivery
        # pairing is semantically meaningful. The pickup-paired/delivery-
        # paired linears mirror the HetGAT structure so this baseline sees
        # the same per-node feature information, just without heterogeneous
        # message passing.
        self.W_depot    = nn.Linear(input_node_dim, hidden_node_dim, bias=False)
        self.W_pickup   = nn.Linear(input_node_dim * 2, hidden_node_dim, bias=False)
        self.W_delivery = nn.Linear(input_node_dim * 2, hidden_node_dim, bias=False)
        self.ln_in = nn.LayerNorm(hidden_node_dim)

        self.layers = nn.ModuleList(
            _KoolMHA(hidden_node_dim, n_heads=n_heads, ff_dim=ff_dim)
            for _ in range(n_layers)
        )
        self.ln_out = nn.LayerNorm(hidden_node_dim)

        # Unknown token mirrors Encoder's initial_visited handling so the
        # baseline receives the same information about unrevealed nodes.
        self.unknown_token = nn.Parameter(torch.zeros(hidden_node_dim))
        nn.init.normal_(self.unknown_token, std=0.1)

    def forward(self, data, initial_visited=None):
        if initial_visited is None:
            initial_visited = data.get('_iv', None)

        feat = data['x']
        batch_size = feat.shape[0]
        n_total = feat.shape[1]

        nd = data.get('n_depots', None)
        if nd is not None:
            num_depots = int(nd[0]) if hasattr(nd, '__len__') else int(nd)
        else:
            num_depots = int((data['demand'] == 0).sum().item() // batch_size)
        num_nodes = n_total - num_depots
        n_pickup = num_nodes // 2

        depot_f    = feat[:, :num_depots, :]
        pickup_f   = feat[:, num_depots:num_depots + n_pickup, :]
        delivery_f = feat[:, num_depots + n_pickup:, :]

        emb_depot    = self.W_depot(depot_f.reshape(batch_size * num_depots, -1))
        emb_pickup   = self.W_pickup(
            torch.cat([pickup_f, delivery_f], dim=-1)
                 .reshape(batch_size * n_pickup, -1))
        emb_delivery = self.W_delivery(
            torch.cat([delivery_f, pickup_f], dim=-1)
                 .reshape(batch_size * n_pickup, -1))

        x = torch.cat([
            emb_depot.view(batch_size, num_depots, -1),
            emb_pickup.view(batch_size, n_pickup, -1),
            emb_delivery.view(batch_size, n_pickup, -1),
        ], dim=1)
        x = self.ln_in(x)

        if initial_visited is not None:
            iv = initial_visited.reshape(batch_size, n_total, 1).float()
            unk = self.unknown_token.view(1, 1, -1).expand(batch_size, n_total, -1)
            x = x * (1.0 - iv) + unk * iv

        for layer in self.layers:
            x = layer(x)
        x = self.ln_out(x)

        # Fleet-homogeneous: both streams receive the same embedding tensor.
        # The downstream decoder's mode-specific feasibility masks are what
        # keep UAV/ADR decisions distinct at the policy level.
        return x, x.clone()


class KoolAMModel(nn.Module):
    """Drop-in replacement for ``VRP_Actor.Model`` using the Kool AM encoder.

    Same ``forward`` signature as ``Model``, same return conventions
    (3-tuple default, 4-tuple with ``return_entropy=True``). Trainable with
    the existing ``VRP_Rollout_train.py`` by swapping the actor factory call.
    """

    def __init__(self, input_node_dim: int = 11, hidden_node_dim: int = 128,
                 input_edge_dim: int = 4, hidden_edge_dim: int = 16,
                 conv_layers: int = 3, n_heads: int = 8, ff_dim: int = 512,
                 edge_features: bool = True, decoder_biases: bool = True):
        super().__init__()
        # Edge features / cross-fleet don't apply to Kool AM (no edge features
        # by design, fleet-homogeneous). Accept the kwargs for interface
        # parity with Model so the trainer factory can pass them uniformly.
        self.edge_features = bool(edge_features)
        self.decoder_biases = bool(decoder_biases)
        self.encoder = KoolAMEncoder(input_node_dim, hidden_node_dim,
                                      n_heads=n_heads, n_layers=conv_layers,
                                      ff_dim=ff_dim)
        # Reuse the shared physics-aware decoder — identical operational
        # contract to HetGAT means apples-to-apples comparison downstream.
        self.decoder = CoopDecoder(hidden_node_dim, hidden_node_dim,
                                    decoder_biases=self.decoder_biases)
        # Expose the dims so experiment_manifest's checkpoint inspector can
        # read them off the state_dict in exactly the same way as Model.
        self._input_node_dim = input_node_dim
        self._hidden_node_dim = hidden_node_dim

    def forward(self, data, num_uav, num_adr, greedy=False, T=1.0,
                checkpoint_encoder=False, training=False, parallel_select=False,
                initial_visited=None, start_node=None, start_time=None,
                start_battery=None, start_capacity=None, available_mask=None,
                delivery_owner=None, return_entropy=False):
        from reward import native_physical_configuration
        config = native_physical_configuration(data)
        import torch.utils.checkpoint as cp
        if checkpoint_encoder and training:
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
            actions, log_p, time_, entropy = _decoder_out
        else:
            actions, log_p, time_ = _decoder_out

        if data.get('service_policy') == 'mandatory':
            from reward import replay_mandatory_trajectory
            if data.get('route_scope', 'complete_episode') not in ('complete_episode', 'rolling_prefix'):
                raise ValueError('Unknown mandatory route valuation scope')
            replay_mandatory_trajectory(
                data, actions, time_, num_uav, num_depots,
                initial_visited=initial_visited,
                start_time=start_time, start_battery=start_battery,
                start_capacity=start_capacity,
                allow_partial=data.get('route_scope') == 'rolling_prefix',
            )
        if return_entropy:
            return actions, log_p, time_, entropy
        return actions, log_p, time_
