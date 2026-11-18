"""GANN (Guo et al.) learned-routing baseline, adapted to CE-CPDPTW.

Reference: Guo et al. "Graph Attention / GCN Network" for vehicle routing
with reinforcement learning. The architectural signature here is GCN-style
message passing — *not* attention — over node features only, followed by a
pointer-attention decoder (same decoder family as Kool et al. 2019 AM).

Design intent — this is a *learned-routing baseline*, not a competitor
architecture. Shares the batch dict interface, the physics-aware
``CoopDecoder``, the mandatory-service reward replay, and the training
loss with ``VRP_Actor.Model``. Only the encoder differs.

Faithful choices vs. Guo GANN:

- **Node features only.** No ``edge_attr_uav`` / ``edge_attr_adr`` and no
  ``mask_adjacency_*`` reach the encoder. The GCN adjacency is derived
  purely from node coordinates — ``data['x'][:, :2]`` — as a K-nearest-
  neighbour sparse graph. This mirrors how a GCN baseline without
  pre-computed edge features builds its graph from the raw input.
- **GCN message passing.** Each layer applies the symmetric-normalised
  propagation ``h' = σ(D^-1/2 (A+I) D^-1/2 h W)``. No attention, no edge
  features in the aggregator.

Minimum delta required for CE-CPDPTW:

- **Mode-id embedding** (UAV-only=0, ADR-only=1, shared=2), derived from
  the per-node accessibility flags ``data['x'][:, :, 8]`` and ``[:, :, 9]``
  that the dispatcher already enforces on the decoder side. This is the
  single scalar needed for the baseline to see the two-fleet world; no
  edge features or graph-level info are added.

Train with ``VRP_Rollout_train.py --arch gann`` once the trainer's model
factory is wired to this module.
"""
from __future__ import annotations

from pathlib import Path
import sys

import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from VRP_Actor import CoopDecoder  # shared physics-aware decoder


_MODE_UAV_ONLY, _MODE_ADR_ONLY, _MODE_SHARED = 0, 1, 2


def _derive_mode_ids(acc_uav: torch.Tensor, acc_adr: torch.Tensor) -> torch.Tensor:
    """0 = UAV-only, 1 = ADR-only, 2 = shared. See hetero_attn_pdp._derive_mode_ids."""
    uav_ok = acc_uav > 0.5
    adr_ok = acc_adr > 0.5
    mode = torch.full_like(acc_uav, fill_value=float(_MODE_ADR_ONLY), dtype=torch.long)
    mode = torch.where(uav_ok & ~adr_ok, torch.full_like(mode, _MODE_UAV_ONLY), mode)
    mode = torch.where(uav_ok & adr_ok,  torch.full_like(mode, _MODE_SHARED),  mode)
    return mode


def _knn_normalised_adjacency(coords: torch.Tensor, k: int) -> torch.Tensor:
    """Symmetric K-NN adjacency with self-loops, GCN-normalised.

    Returns ``Â = D^-1/2 (A + I) D^-1/2`` of shape ``[B, N, N]``, where
    ``A`` is the symmetrised K-NN graph over 2-D coordinates. Using
    coordinates — not edge features — keeps the GCN adjacency within the
    "node features only" contract.

    When ``N <= k``, the graph degenerates to a fully-connected graph with
    self-loops, which still yields a valid GCN layer.
    """
    B, N, _ = coords.shape
    # Pairwise Euclidean distance; add self-distance so the K-NN includes
    # the node itself, which we then convert to an explicit self-loop below.
    diff = coords.unsqueeze(2) - coords.unsqueeze(1)   # [B, N, N, 2]
    dist = diff.pow(2).sum(dim=-1).clamp_min(0)        # [B, N, N]

    k_eff = min(int(k), N)
    # Negate so topk returns smallest distances; include self.
    _, nn_idx = torch.topk(-dist, k=k_eff, dim=-1)     # [B, N, k_eff]
    A = torch.zeros(B, N, N, device=coords.device, dtype=coords.dtype)
    A.scatter_(2, nn_idx, 1.0)

    # Symmetrise: edge exists if either direction found it.
    A = ((A + A.transpose(1, 2)) > 0).to(coords.dtype)
    # Self-loops (idempotent against K-NN self-inclusion).
    I_eye = torch.eye(N, device=coords.device, dtype=coords.dtype).expand(B, -1, -1)
    A = ((A + I_eye) > 0).to(coords.dtype)

    deg = A.sum(dim=-1).clamp_min(1.0)                  # [B, N]
    d_inv_sqrt = deg.pow(-0.5)
    norm = d_inv_sqrt.unsqueeze(2) * d_inv_sqrt.unsqueeze(1)  # [B, N, N]
    return A * norm


class _GCNLayer(nn.Module):
    """One GCN propagation step: ``h' = σ(Â h W)`` with pre-LN residual."""

    def __init__(self, hidden_dim: int):
        super().__init__()
        self.W = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.ln = nn.LayerNorm(hidden_dim)

    def forward(self, h: torch.Tensor, A_norm: torch.Tensor) -> torch.Tensor:
        h_in = self.ln(h)
        agg = torch.bmm(A_norm, self.W(h_in))
        return h + F.relu(agg)


class GANNEncoder(nn.Module):
    """GCN-style encoder (Guo GANN family).

    Returns ``(x_uav, x_adr)`` where both streams are identical — fleet
    heterogeneity enters only through the shared decoder's feasibility
    masks. This is the structural claim Guo GANN makes: a plain GCN
    aggregator over node features, no mode-specific message passing, no
    edge features.

    Compared to the HetGAT ``Encoder``, deliberately omitted:
      - Edge features (distance / energy / signed-margin / temporal-gap).
      - Mask-adjacency within attention.
      - Fleet-separated attention heads.
      - Cross-fleet message passing.
      - Decoder biases.
      - Attention mechanism entirely (GANN uses GCN rules).
    Added (minimum delta for CE-CPDPTW): mode-id embedding (UAV-only /
    ADR-only / shared).
    """

    def __init__(
        self,
        input_node_dim: int = 11,
        hidden_node_dim: int = 128,
        n_layers: int = 3,
        knn_k: int = 10,
    ):
        super().__init__()
        self.input_node_dim = int(input_node_dim)
        self.hidden_node_dim = int(hidden_node_dim)
        self.knn_k = int(knn_k)

        # Match the KoolAM / HetGAT state_dict surface so the artefact
        # evaluator infers dims identically across architectures.
        self.W_depot    = nn.Linear(input_node_dim,     hidden_node_dim, bias=False)
        self.W_pickup   = nn.Linear(input_node_dim * 2, hidden_node_dim, bias=False)
        self.W_delivery = nn.Linear(input_node_dim * 2, hidden_node_dim, bias=False)

        # Minimum heterogeneity signal: mode-id embedding only.
        self.mode_emb = nn.Embedding(3, hidden_node_dim)
        nn.init.normal_(self.mode_emb.weight, std=0.02)

        self.ln_in = nn.LayerNorm(hidden_node_dim)
        self.layers = nn.ModuleList(
            _GCNLayer(hidden_node_dim) for _ in range(n_layers)
        )
        self.ln_out = nn.LayerNorm(hidden_node_dim)

        self.unknown_token = nn.Parameter(torch.zeros(hidden_node_dim))
        nn.init.normal_(self.unknown_token, std=0.02)

    def forward(self, data, initial_visited=None):
        if initial_visited is None:
            initial_visited = data.get('_iv', None)

        feat = data['x']
        batch_size, n_total, _ = feat.shape

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

        mode_ids = _derive_mode_ids(feat[:, :, 8], feat[:, :, 9])
        x = x + self.mode_emb(mode_ids)
        x = self.ln_in(x)

        if initial_visited is not None:
            iv = initial_visited.reshape(batch_size, n_total, 1).to(x.dtype)
            unk = self.unknown_token.view(1, 1, -1).expand(batch_size, n_total, -1)
            x = x * (1.0 - iv) + unk * iv

        # Build the GCN adjacency from raw coords only — not from edge
        # features — so the "node features only" constraint is preserved.
        coords = feat[:, :, :2]
        A_norm = _knn_normalised_adjacency(coords, k=self.knn_k)

        for layer in self.layers:
            x = layer(x, A_norm)
        x = self.ln_out(x)

        return x, x.clone()


class GANNModel(nn.Module):
    """Drop-in replacement for ``VRP_Actor.Model`` using the GANN encoder.

    Interface parity with ``Model`` / ``KoolAMModel`` / ``HAPDPModel``:
    same forward signature, same return conventions. Trainable via
    ``VRP_Rollout_train.py`` by selecting ``--arch gann`` once the factory
    is wired.
    """

    def __init__(
        self,
        input_node_dim: int = 11,
        hidden_node_dim: int = 128,
        input_edge_dim: int = 4,   # accepted for factory parity; unused
        hidden_edge_dim: int = 16, # accepted for factory parity; unused
        conv_layers: int = 3,
        n_heads: int = 8,          # accepted for factory parity; GCN has no heads
        ff_dim: int = 512,         # accepted for factory parity; unused
        edge_features: bool = True,
        decoder_biases: bool = True,
        knn_k: int = 10,
    ):
        super().__init__()
        self.edge_features = bool(edge_features)
        self.decoder_biases = bool(decoder_biases)
        self.encoder = GANNEncoder(
            input_node_dim=input_node_dim,
            hidden_node_dim=hidden_node_dim,
            n_layers=conv_layers,
            knn_k=knn_k,
        )
        self.decoder = CoopDecoder(
            hidden_node_dim, hidden_node_dim,
            decoder_biases=self.decoder_biases,
        )
        self._input_node_dim = input_node_dim
        self._hidden_node_dim = hidden_node_dim

    def forward(
        self, data, num_uav, num_adr, greedy=False, T=1.0,
        checkpoint_encoder=False, training=False, parallel_select=False,
        initial_visited=None, start_node=None, start_time=None,
        start_battery=None, start_capacity=None, available_mask=None,
        delivery_owner=None, return_entropy=False,
    ):
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
        n_depots_uav = (int(nd_uav[0]) if hasattr(nd_uav, '__len__') else int(nd_uav)) \
            if nd_uav is not None else None

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
            if data.get('route_scope', 'complete_episode') not in (
                'complete_episode', 'rolling_prefix',
            ):
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
