"""Heterogeneous Attentions PDP-DRL (Li et al. 2019), adapted to CE-CPDPTW.

Reference: J. Li, Y. Ma, R. Gao, Z. Cao, A. Lim, W. Song, J. Zhang. "Deep
Reinforcement Learning for Solving the Heterogeneous Capacitated Vehicle
Routing Problem." (Li 2019) — role-aware "heterogeneous attention" over
pickup/delivery/depot nodes with precedence awareness, no edge features.

Design intent — this is a *learned-routing baseline*, not a competitor
architecture. Like ``baselines.kool_am`` it shares every operational
mechanism with ``VRP_Actor.Model`` (same batch dict interface, same
``CoopDecoder`` physics stack, same training loss, same reward); only the
encoder differs.

Faithful choices vs. Li 2019:

- **Node features only.** No ``edge_attr_uav`` / ``edge_attr_adr`` and no
  ``mask_adjacency_*`` reach the encoder. Mode-pair feasibility is enforced
  by the shared decoder's masks, exactly as it would be if the original
  single-vehicle PDPTW were extended to multi-fleet.
- **Role-aware attention.** Each encoder layer stacks three heterogeneous
  sub-attentions:

    1. *Same-role* attention — queries attend only to K/V of their own role
       (pickup→pickup, delivery→delivery).
    2. *Cross-role* attention — queries attend only to K/V of the opposite
       role (pickup→delivery, delivery→pickup).
    3. *Precedence* attention — a pickup is explicitly told the state of
       its paired delivery (and vice versa) via a direct, learned
       projection of the paired node's hidden state.

  Together these cover Li 2019's "6 heterogeneous attention types" (3
  role-pair groups × 2 directions) with weight sharing within each group.

Minimum delta required for CE-CPDPTW:

- **Role embedding** (depot=0, pickup=1, delivery=2) — derived positionally
  from the batch's depot-pickup-delivery layout.
- **Mode-id embedding** (UAV-only=0, ADR-only=1, shared=2) — derived from
  the per-node accessibility flags (``data['x'][:, :, 8]`` and
  ``[:, :, 9]``) that the dispatcher already enforces on the decoder side.
  This is the single scalar needed for the baseline to *see* the two-fleet
  world; no edge features or graph-level info are added.

Train with ``VRP_Rollout_train.py --arch hetero_attn_pdp`` once the
trainer's model factory is wired to this module.
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


# Role / mode indices (small integer tables, used by nn.Embedding).
_ROLE_DEPOT, _ROLE_PICKUP, _ROLE_DELIVERY = 0, 1, 2
_MODE_UAV_ONLY, _MODE_ADR_ONLY, _MODE_SHARED = 0, 1, 2


def _derive_mode_ids(acc_uav: torch.Tensor, acc_adr: torch.Tensor) -> torch.Tensor:
    """Convert per-node (acc_uav, acc_adr) flags to a mode-id scalar.

    0 = UAV-only, 1 = ADR-only, 2 = shared. Non-accessible nodes (should not
    occur after the depot-accessibility fix) collapse to ADR-only; the
    downstream decoder's feasibility masks veto them regardless.
    """
    uav_ok = acc_uav > 0.5
    adr_ok = acc_adr > 0.5
    # start from ADR-only, flip to UAV-only or shared as appropriate
    mode = torch.full_like(acc_uav, fill_value=float(_MODE_ADR_ONLY), dtype=torch.long)
    mode = torch.where(uav_ok & ~adr_ok, torch.full_like(mode, _MODE_UAV_ONLY), mode)
    mode = torch.where(uav_ok & adr_ok,  torch.full_like(mode, _MODE_SHARED),  mode)
    return mode


class _HAPDPAttention(nn.Module):
    """Three role-pair attention sub-blocks: same-role, cross-role, precedence.

    Pre-LN residual: ``x + concat(SR,CR,PR)(LN(x)) → x + FF(LN(x))``.
    """

    def __init__(self, hidden_dim: int, n_heads: int = 8, ff_dim: int = 512):
        super().__init__()
        assert hidden_dim % n_heads == 0, (
            f'hidden_dim={hidden_dim} not divisible by n_heads={n_heads}')
        self.n_heads = n_heads
        self.head_dim = hidden_dim // n_heads
        self.scale = 1.0 / math.sqrt(self.head_dim)
        self.hidden_dim = hidden_dim

        # Same-role and cross-role each have their own Q/K/V (weight-shared
        # across both directions within the group — e.g. pickup→pickup and
        # delivery→delivery share params in the same-role block).
        self.q_sr = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.k_sr = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.v_sr = nn.Linear(hidden_dim, hidden_dim, bias=False)

        self.q_cr = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.k_cr = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.v_cr = nn.Linear(hidden_dim, hidden_dim, bias=False)

        # Precedence sub-block: a direct gather of the paired node's hidden
        # state (pickup ↔ delivery) through a learned linear. No softmax —
        # Li 2019 treats precedence as a hard precedence link, not a soft
        # attention distribution, which is how we interpret the "u^Pre /
        # u^Succ" slot of the heterogeneous-attention set.
        self.w_pr = nn.Linear(hidden_dim, hidden_dim, bias=False)

        # Fuse the three role-pair streams back into a single hidden vector.
        self.out = nn.Linear(hidden_dim * 3, hidden_dim, bias=False)

        self.ln1 = nn.LayerNorm(hidden_dim)
        self.ln2 = nn.LayerNorm(hidden_dim)
        self.ff = nn.Sequential(
            nn.Linear(hidden_dim, ff_dim),
            nn.ReLU(),
            nn.Linear(ff_dim, hidden_dim),
        )

    @staticmethod
    def _reshape_heads(t: torch.Tensor, n_heads: int, head_dim: int) -> torch.Tensor:
        B, N, _ = t.shape
        return t.view(B, N, n_heads, head_dim).transpose(1, 2)  # [B, H, N, Dh]

    def _masked_attention(
        self,
        q_proj: nn.Linear, k_proj: nn.Linear, v_proj: nn.Linear,
        x: torch.Tensor,
        attn_mask: torch.Tensor,
    ) -> torch.Tensor:
        """Multi-head attention restricted to ``attn_mask`` key positions.

        ``attn_mask`` is ``[B, N, N]`` with 1 at allowed (query, key) pairs
        and 0 elsewhere. Rows with no allowed key degrade to the input (no
        attention update) rather than NaN out through an all-masked softmax.
        """
        B, N, _ = x.shape
        Q = self._reshape_heads(q_proj(x), self.n_heads, self.head_dim)
        K = self._reshape_heads(k_proj(x), self.n_heads, self.head_dim)
        V = self._reshape_heads(v_proj(x), self.n_heads, self.head_dim)

        compat = self.scale * torch.matmul(Q, K.transpose(-2, -1))  # [B, H, N, N]
        # Broadcast [B,1,N,N] against [B,H,N,N]
        allowed = attn_mask.unsqueeze(1).to(compat.dtype)
        neg_inf = torch.finfo(compat.dtype).min
        compat = compat.masked_fill(allowed < 0.5, neg_inf)

        # Rows that have no allowed key — zero them out explicitly so the
        # softmax yields a uniform-but-irrelevant row and the per-row result
        # is overwritten to 0 below.
        row_has_key = (allowed.sum(dim=-1, keepdim=True) > 0).to(compat.dtype)
        safe_compat = torch.where(row_has_key > 0, compat, torch.zeros_like(compat))
        attn = F.softmax(safe_compat, dim=-1)
        out = torch.matmul(attn, V)  # [B, H, N, Dh]
        out = out * row_has_key
        out = out.transpose(1, 2).contiguous().view(B, N, self.hidden_dim)
        return out

    def forward(
        self, x: torch.Tensor,
        mask_same_role: torch.Tensor,
        mask_cross_role: torch.Tensor,
        precedence_index: torch.Tensor,
        precedence_valid: torch.Tensor,
    ) -> torch.Tensor:
        h = self.ln1(x)

        sr = self._masked_attention(self.q_sr, self.k_sr, self.v_sr, h, mask_same_role)
        cr = self._masked_attention(self.q_cr, self.k_cr, self.v_cr, h, mask_cross_role)

        # Precedence: for each node, gather its paired node's hidden state.
        # ``precedence_index`` is [B, N] with the index of the paired node,
        # or an index into the depot slot where no pair exists; the
        # ``precedence_valid`` mask zeros the depot/padding contributions.
        gather_idx = precedence_index.unsqueeze(-1).expand(-1, -1, self.hidden_dim)
        paired = torch.gather(h, dim=1, index=gather_idx)
        pr = self.w_pr(paired) * precedence_valid.unsqueeze(-1).to(h.dtype)

        fused = self.out(torch.cat([sr, cr, pr], dim=-1))
        x = x + fused
        x = x + self.ff(self.ln2(x))
        return x


class HAPDPEncoder(nn.Module):
    """Role-aware heterogeneous-attention encoder (Li et al. 2019).

    Returns ``(x_uav, x_adr)`` where both streams are identical — fleet
    heterogeneity enters only through the per-step feasibility masks the
    shared ``CoopDecoder`` already enforces. This is the structural claim
    Li 2019 makes: the baseline encodes role-pair and precedence structure
    but is otherwise fleet-homogeneous.

    Compared to the HetGAT ``Encoder``, deliberately omitted:
      - Edge features (distance / energy / signed-margin / temporal-gap).
      - Mask-adjacency within attention.
      - Fleet-separated attention heads.
      - Cross-fleet message passing.
      - Decoder biases.
    Added (minimum delta for CE-CPDPTW): role embedding (depot/pickup/
    delivery) and mode-id embedding (UAV-only/ADR-only/shared).
    """

    def __init__(
        self,
        input_node_dim: int = 11,
        hidden_node_dim: int = 128,
        n_heads: int = 8,
        n_layers: int = 3,
        ff_dim: int = 512,
    ):
        super().__init__()
        self.input_node_dim = int(input_node_dim)
        self.hidden_node_dim = int(hidden_node_dim)

        # Match ``VRP_Actor.Encoder`` / ``KoolAMEncoder`` state_dict surface:
        # the Tier-N artefact evaluator infers (hidden, input) dims from
        # ``encoder.W_depot.weight`` identically across architectures.
        self.W_depot    = nn.Linear(input_node_dim,     hidden_node_dim, bias=False)
        self.W_pickup   = nn.Linear(input_node_dim * 2, hidden_node_dim, bias=False)
        self.W_delivery = nn.Linear(input_node_dim * 2, hidden_node_dim, bias=False)

        # Minimum delta for CE-CPDPTW: role and mode-id embeddings.
        self.role_emb = nn.Embedding(3, hidden_node_dim)
        self.mode_emb = nn.Embedding(3, hidden_node_dim)
        nn.init.normal_(self.role_emb.weight, std=0.02)
        nn.init.normal_(self.mode_emb.weight, std=0.02)

        self.ln_in = nn.LayerNorm(hidden_node_dim)
        self.layers = nn.ModuleList(
            _HAPDPAttention(hidden_node_dim, n_heads=n_heads, ff_dim=ff_dim)
            for _ in range(n_layers)
        )
        self.ln_out = nn.LayerNorm(hidden_node_dim)

        self.unknown_token = nn.Parameter(torch.zeros(hidden_node_dim))
        nn.init.normal_(self.unknown_token, std=0.02)

    def _build_role_and_pairs(
        self, batch_size: int, n_total: int, num_depots: int, n_pickup: int,
        device: torch.device,
    ):
        """Role ids, same-/cross-role masks, and precedence index per node.

        Role ids are positional: ``[0..num_depots)`` is depot,
        ``[num_depots..num_depots+n_pickup)`` is pickup, and the rest is
        delivery. The pickup→delivery pairing is index-aligned: pickup *i*
        is paired with delivery ``i``.
        """
        role_ids = torch.empty(n_total, dtype=torch.long, device=device)
        role_ids[:num_depots] = _ROLE_DEPOT
        role_ids[num_depots:num_depots + n_pickup] = _ROLE_PICKUP
        role_ids[num_depots + n_pickup:] = _ROLE_DELIVERY

        is_pickup   = (role_ids == _ROLE_PICKUP)
        is_delivery = (role_ids == _ROLE_DELIVERY)
        is_customer = is_pickup | is_delivery

        # Same-role / cross-role masks are symmetric and depot-exclusive on
        # both axes: depots do not participate as queries or keys in the
        # role-pair stack. Depot embeddings pass through unchanged via the
        # residual connection (and reach the decoder as-is).
        same_role = (
            (is_pickup.view(-1, 1)   & is_pickup.view(1, -1)) |
            (is_delivery.view(-1, 1) & is_delivery.view(1, -1))
        )
        cross_role = (
            (is_pickup.view(-1, 1)   & is_delivery.view(1, -1)) |
            (is_delivery.view(-1, 1) & is_pickup.view(1, -1))
        )
        same_role  = same_role.float().unsqueeze(0).expand(batch_size, -1, -1)
        cross_role = cross_role.float().unsqueeze(0).expand(batch_size, -1, -1)

        # Precedence index: pickup i ↔ delivery (num_depots + n_pickup + i).
        pair_idx = torch.arange(n_total, device=device)
        pickup_slice   = slice(num_depots, num_depots + n_pickup)
        delivery_slice = slice(num_depots + n_pickup, n_total)
        pickup_to_delivery   = torch.arange(n_pickup, device=device) + (num_depots + n_pickup)
        delivery_to_pickup   = torch.arange(n_pickup, device=device) + num_depots
        pair_idx = pair_idx.clone()
        pair_idx[pickup_slice]   = pickup_to_delivery
        pair_idx[delivery_slice] = delivery_to_pickup
        # Depots have no pair — point them at index 0 (any valid index) and
        # zero out via precedence_valid so the gather returns an ignored row.
        precedence_valid = is_customer.float()
        pair_idx = pair_idx.unsqueeze(0).expand(batch_size, -1)
        precedence_valid = precedence_valid.unsqueeze(0).expand(batch_size, -1)
        role_ids = role_ids.unsqueeze(0).expand(batch_size, -1)
        return role_ids, same_role, cross_role, pair_idx, precedence_valid

    def forward(self, data, initial_visited=None):
        if initial_visited is None:
            initial_visited = data.get('_iv', None)

        feat = data['x']
        batch_size, n_total, _ = feat.shape
        device = feat.device

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

        role_ids, same_role, cross_role, pair_idx, prec_valid = (
            self._build_role_and_pairs(
                batch_size, n_total, num_depots, n_pickup, device))

        # The minimum heterogeneity signal: role + mode embeddings.
        mode_ids = _derive_mode_ids(feat[:, :, 8], feat[:, :, 9])
        x = x + self.role_emb(role_ids) + self.mode_emb(mode_ids)
        x = self.ln_in(x)

        if initial_visited is not None:
            iv = initial_visited.reshape(batch_size, n_total, 1).to(x.dtype)
            unk = self.unknown_token.view(1, 1, -1).expand(batch_size, n_total, -1)
            x = x * (1.0 - iv) + unk * iv

        for layer in self.layers:
            x = layer(x, same_role, cross_role, pair_idx, prec_valid)
        x = self.ln_out(x)

        # Fleet-homogeneous output; decoder masks enforce UAV/ADR distinction.
        return x, x.clone()


class HAPDPModel(nn.Module):
    """Drop-in replacement for ``VRP_Actor.Model`` using the HA-PDP encoder.

    Interface parity with ``Model`` / ``KoolAMModel``: same forward
    signature, same return conventions (3-tuple default, 4-tuple with
    ``return_entropy=True``). Trainable via ``VRP_Rollout_train.py`` by
    selecting ``--arch hetero_attn_pdp`` once the factory is wired.
    """

    def __init__(
        self,
        input_node_dim: int = 11,
        hidden_node_dim: int = 128,
        input_edge_dim: int = 4,   # accepted for factory parity; unused
        hidden_edge_dim: int = 16, # accepted for factory parity; unused
        conv_layers: int = 3,
        n_heads: int = 8,
        ff_dim: int = 512,
        edge_features: bool = True,
        decoder_biases: bool = True,
    ):
        super().__init__()
        # HA-PDP has no edge features by design and is fleet-homogeneous.
        # These kwargs are accepted so the trainer's uniform factory signature
        # works; only ``decoder_biases`` reaches the shared decoder.
        self.edge_features = bool(edge_features)
        self.decoder_biases = bool(decoder_biases)
        self.encoder = HAPDPEncoder(
            input_node_dim=input_node_dim,
            hidden_node_dim=hidden_node_dim,
            n_heads=n_heads,
            n_layers=conv_layers,
            ff_dim=ff_dim,
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
