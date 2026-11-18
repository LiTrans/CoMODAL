import math
from typing import Dict, List, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from rl_value_baseline import GLOBAL_FEATURE_DIM
from vrpUpdate import update_state, V_UAV_MAX, V_ADR_MAX, UAV_LAND_TAKEOFF_MIN


WINDOW_CHOICES = torch.tensor([1.0, 2.0, 5.0, 10.0, 15.0, 30.0, 60.0])
ARRIVAL_HISTORY_DIM = 8

# Once-per-process diagnostic for the cooperative capacity safety valve.
# Set via _warn_capacity_overflow; the first occurrence prints full context,
# subsequent occurrences are counted but silent to avoid log flooding.
_CAP_OVERFLOW_WARNED = False
_CAP_OVERFLOW_COUNT = 0


def _warn_capacity_overflow(old_cap, demand_sel, new_cap, overflow):
    """Diagnostic for :func:`AssignmentDecoder._advance_assignment_state`.

    Logs the first observed capacity-overflow sample with full numeric
    context (vehicle capacity, request demand, resulting new_cap) so the
    user can distinguish float32-noise band crossings from genuine logic
    bugs. Subsequent occurrences are counted silently.
    """
    global _CAP_OVERFLOW_WARNED, _CAP_OVERFLOW_COUNT
    _CAP_OVERFLOW_COUNT += int(overflow.sum().item())
    if _CAP_OVERFLOW_WARNED:
        return
    _CAP_OVERFLOW_WARNED = True
    with torch.no_grad():
        bad_idx = overflow.nonzero(as_tuple=False)
        n_show = min(3, bad_idx.shape[0])
        preview = []
        for row in bad_idx[:n_show]:
            b = int(row[0].item())
            preview.append(
                f'[batch={b} cap={float(old_cap[b].item()):+.6f} '
                f'demand={float(demand_sel[b].item()):+.6f} '
                f'new_cap={float(new_cap[b].item()):+.6f}]'
            )
    import sys
    print(
        f'[coop-decoder] Capacity-overflow safety valve engaged '
        f'(n_overflow={int(overflow.sum().item())} in this batch, '
        f'cumulative={_CAP_OVERFLOW_COUNT}). '
        f'Converting offenders to no-op and continuing. '
        f'First samples: {" ".join(preview)}',
        file=sys.stderr, flush=True,
    )


def _as_scalar_int(value, default=None) -> int:
    if value is None:
        if default is None:
            raise ValueError('Missing required scalar value')
        return int(default)
    if torch.is_tensor(value):
        return int(value.flatten()[0].item())
    if hasattr(value, '__len__') and not isinstance(value, (str, bytes)):
        return int(value[0])
    return int(value)


def _masked_edge_dists(batch: Dict[str, torch.Tensor], B: int, N: int):
    # Stamp mask_adjacency-forbidden edges with +inf so distance-based
    # feasibility filters (`raw_dist < 1e8` / `torch.isfinite`) exclude them.
    d_uav = batch['edge_attr_d'].view(B, N, N).float()
    d_adr = batch['edge_attr_r'].view(B, N, N).float()
    m_uav = batch.get('mask_adjacency_uav')
    m_adr = batch.get('mask_adjacency_adr')
    inf = torch.tensor(float('inf'), device=d_uav.device, dtype=d_uav.dtype)
    if isinstance(m_uav, torch.Tensor):
        d_uav = torch.where(m_uav.view(B, N, N) > 0.5, d_uav, inf)
    if isinstance(m_adr, torch.Tensor):
        d_adr = torch.where(m_adr.view(B, N, N) > 0.5, d_adr, inf)
    return d_uav, d_adr


class RequestPairEncoder(nn.Module):
    def __init__(self, node_dim: int, hidden_dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(node_dim * 2 + 10, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
        )

    def forward(self, batch: Dict[str, torch.Tensor], num_depots: int) -> torch.Tensor:
        x = batch['x'].float()
        B, N, Fdim = x.shape
        n_req = (N - num_depots) // 2
        pickup = x[:, num_depots:num_depots + n_req, :]
        delivery = x[:, num_depots + n_req:num_depots + 2 * n_req, :]

        demand = batch['demand'].view(B, N).float()[:, num_depots:num_depots + n_req]
        tw = batch['time_window'].view(B, N).float()
        pickup_tw = tw[:, num_depots:num_depots + n_req]
        delivery_tw = tw[:, num_depots + n_req:num_depots + 2 * n_req]
        coord_delta = delivery[:, :, :2] - pickup[:, :, :2]
        pair_dist = coord_delta.norm(dim=-1)

        acc_uav = (pickup[:, :, 8] * delivery[:, :, 8]).unsqueeze(-1)
        acc_adr = (pickup[:, :, 9] * delivery[:, :, 9]).unsqueeze(-1)
        aux = torch.cat([
            demand.unsqueeze(-1) / 10.0,
            pickup_tw.unsqueeze(-1) / 180.0,
            delivery_tw.unsqueeze(-1) / 180.0,
            pair_dist.unsqueeze(-1),
            coord_delta,
            acc_uav,
            acc_adr,
            (delivery_tw - pickup_tw).unsqueeze(-1) / 180.0,
            torch.ones(B, n_req, 1, device=x.device),
        ], dim=-1)
        return self.net(torch.cat([pickup, delivery, aux], dim=-1))


class VehicleStateEncoder(nn.Module):
    def __init__(self, node_dim: int, hidden_dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(node_dim + 10, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
        )

    def forward(
        self,
        batch: Dict[str, torch.Tensor],
        num_uav: int,
        num_adr: int,
        num_depots: int,
        n_depots_uav: int = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        x = batch['x'].float()
        B, N, _ = x.shape
        A = num_uav + num_adr
        device = x.device
        n_depots_uav = num_depots if n_depots_uav is None else n_depots_uav

        if 'start_node' in batch:
            start_node = batch['start_node'].view(B, A).long().to(device)
        else:
            starts = []
            for ag in range(num_uav):
                starts.append(ag % max(n_depots_uav, 1))
            n_dep_adr = max(num_depots - n_depots_uav, 1)
            for ag in range(num_adr):
                starts.append(n_depots_uav + ag % n_dep_adr)
            start_node = torch.tensor(starts, dtype=torch.long, device=device).view(1, A).expand(B, -1)

        full_cap = batch['capacity'].view(B, A).float().to(device).clamp(min=1.0)
        full_bat = batch['battery'].view(B, A).float().to(device).clamp(min=1.0)
        cap = batch.get('start_capacity', batch['capacity']).view(B, A).float().to(device)
        bat = batch.get('start_battery', batch['battery']).view(B, A).float().to(device)
        start_time = batch.get('start_time')
        if isinstance(start_time, torch.Tensor):
            cur_time = start_time.view(B, A).float().to(device)
        else:
            cur_time = torch.zeros(B, A, device=device)

        available = batch.get('available_mask')
        if isinstance(available, torch.Tensor):
            available = available.view(B, A).float().to(device)
        else:
            available = torch.ones(B, A, device=device)

        node_feat = torch.gather(
            x, 1, start_node.unsqueeze(-1).expand(-1, -1, x.size(-1))
        )
        mode = torch.zeros(B, A, 2, device=device)
        mode[:, :num_uav, 0] = 1.0
        mode[:, num_uav:, 1] = 1.0
        aux = torch.cat([
            mode,
            (cap / full_cap).unsqueeze(-1),
            (bat / full_bat).unsqueeze(-1),
            (cur_time / 180.0).unsqueeze(-1),
            available.unsqueeze(-1),
            (1.0 - cap / full_cap).unsqueeze(-1),
            torch.full((B, A, 1), float(num_uav) / max(A, 1), device=device),
            torch.full((B, A, 1), float(num_adr) / max(A, 1), device=device),
            start_node.float().unsqueeze(-1) / max(float(N - 1), 1.0),
        ], dim=-1)
        return self.net(torch.cat([node_feat, aux], dim=-1)), start_node, cur_time, cap, bat


class GlobalWindowEncoder(nn.Module):
    def __init__(self, hidden_dim: int, history_dim: int = ARRIVAL_HISTORY_DIM):
        super().__init__()
        self.history_gru = nn.GRU(history_dim, hidden_dim, batch_first=True)
        self.global_net = nn.Sequential(
            nn.Linear(GLOBAL_FEATURE_DIM + hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
        )
        self.window_head = nn.Linear(hidden_dim, len(WINDOW_CHOICES))

    def forward(self, batch: Dict[str, torch.Tensor]) -> Tuple[torch.Tensor, torch.Tensor]:
        x = batch['x']
        B = x.shape[0]
        device = x.device
        gf = batch.get('global_features')
        if not isinstance(gf, torch.Tensor):
            gf = torch.zeros(B, GLOBAL_FEATURE_DIM, device=device)
        gf = gf.float().to(device)
        hist = batch.get('arrival_history')
        if not isinstance(hist, torch.Tensor):
            hist = torch.zeros(B, 1, ARRIVAL_HISTORY_DIM, device=device)
        hist = hist.float().to(device)
        _, h = self.history_gru(hist)
        global_token = self.global_net(torch.cat([gf, h[-1]], dim=-1))
        return global_token, self.window_head(global_token)


class RelationAwareCoopEncoder(nn.Module):
    def __init__(self, hidden_dim: int, n_heads: int = 4, n_layers: int = 2):
        super().__init__()
        enc_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim, nhead=n_heads, batch_first=True,
            dim_feedforward=hidden_dim * 4, dropout=0.0, activation='relu',
        )
        self.encoder = nn.TransformerEncoder(enc_layer, num_layers=n_layers)
        self.req_type = nn.Parameter(torch.zeros(1, 1, hidden_dim))
        self.veh_type = nn.Parameter(torch.zeros(1, 1, hidden_dim))
        self.glob_type = nn.Parameter(torch.zeros(1, 1, hidden_dim))
        nn.init.normal_(self.req_type, std=0.02)
        nn.init.normal_(self.veh_type, std=0.02)
        nn.init.normal_(self.glob_type, std=0.02)

    def forward(self, req: torch.Tensor, veh: torch.Tensor,
                glob: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        B, R, H = req.shape
        A = veh.shape[1]
        tokens = torch.cat([
            glob.unsqueeze(1) + self.glob_type,
            veh + self.veh_type,
            req + self.req_type,
        ], dim=1)
        out = self.encoder(tokens)
        return out[:, 1 + A:, :], out[:, 1:1 + A, :], out[:, 0, :]


class AssignmentDecoder(nn.Module):
    def __init__(self, hidden_dim: int):
        super().__init__()
        self.vehicle_proj = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.request_proj = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.global_pair = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.pair_bias = nn.Sequential(
            nn.Linear(8, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )
        self.stop_head = nn.Linear(hidden_dim, 1)

    def _pair_features(self, batch, num_depots, num_uav, num_adr, start_node, cur_time, cap):
        B, A = start_node.shape
        N = batch['x'].shape[1]
        R = (N - num_depots) // 2
        device = start_node.device
        demand = batch['demand'].view(B, N).float()[:, num_depots:num_depots + R]
        tw = batch['time_window'].view(B, N).float()
        pickup_tw = tw[:, num_depots:num_depots + R]
        delivery_tw = tw[:, num_depots + R:num_depots + 2 * R]

        d_uav, d_adr = _masked_edge_dists(batch, B, N)
        req_nodes = torch.arange(num_depots, num_depots + R, device=device)
        del_nodes = torch.arange(num_depots + R, num_depots + 2 * R, device=device)
        bidx = torch.arange(B, device=device).view(B, 1, 1).expand(B, A, R)
        sn = start_node.unsqueeze(-1).expand(B, A, R)
        rn = req_nodes.view(1, 1, R).expand(B, A, R)
        dn = del_nodes.view(1, 1, R).expand(B, A, R)
        raw_dist_u = d_uav[bidx, sn, rn]
        raw_dist_a = d_adr[bidx, sn, rn]
        raw_pd_u = d_uav[bidx, rn, dn]
        raw_pd_a = d_adr[bidx, rn, dn]
        dist_u = raw_dist_u.clamp(max=50.0)
        dist_a = raw_dist_a.clamp(max=50.0)
        pd_u = raw_pd_u.clamp(max=50.0)
        pd_a = raw_pd_a.clamp(max=50.0)
        mode_u = torch.zeros(B, A, 1, device=device)
        mode_u[:, :num_uav] = 1.0
        dist = torch.where(mode_u.bool(), dist_u, dist_a)
        pickup_delivery_dist = torch.where(mode_u.bool(), pd_u, pd_a)
        raw_dist = torch.where(mode_u.bool(), raw_dist_u, raw_dist_a)
        raw_pickup_delivery_dist = torch.where(mode_u.bool(), raw_pd_u, raw_pd_a)
        speed = torch.where(
            mode_u.bool(),
            torch.full_like(dist, V_UAV_MAX),
            torch.full_like(dist, V_ADR_MAX),
        )
        travel = dist / speed.clamp(min=1e-6)
        service = torch.where(
            mode_u.bool(),
            torch.full_like(dist, UAV_LAND_TAKEOFF_MIN),
            torch.zeros_like(dist),
        )
        pickup_arrival = cur_time.unsqueeze(-1) + travel
        pickup_complete = torch.maximum(pickup_arrival, pickup_tw.unsqueeze(1)) + service
        delivery_complete = pickup_complete + pickup_delivery_dist / speed.clamp(min=1e-6) + service
        pickup_slack = (pickup_tw.unsqueeze(1) - pickup_arrival) / 180.0
        delivery_slack = (delivery_tw.unsqueeze(1) - delivery_complete) / 180.0
        cap_margin = (cap.unsqueeze(-1) - demand.unsqueeze(1)) / 10.0
        acc = batch['x'][:, :, 8:10].float()
        p_acc = acc[:, num_depots:num_depots + R, :]
        d_acc = acc[:, num_depots + R:num_depots + 2 * R, :]
        uav_ok = (p_acc[:, :, 0] * d_acc[:, :, 0]).unsqueeze(1).expand(B, A, R)
        adr_ok = (p_acc[:, :, 1] * d_acc[:, :, 1]).unsqueeze(1).expand(B, A, R)
        mode_ok = torch.where(mode_u.bool(), uav_ok, adr_ok)

        pair_feat = torch.stack([
            dist / 50.0,
            travel / 120.0,
            pickup_slack,
            delivery_slack,
            cap_margin,
            demand.unsqueeze(1).expand(B, A, R) / 10.0,
            uav_ok.float(),
            adr_ok.float(),
        ], dim=-1)
        # Capacity feasibility: tightened from ``>= -1e-6`` to
        # ``>= 1e-5`` (post-scaling: demand ≤ cap − 1e-4). Rationale: the
        # downstream sanity check in ``_advance_assignment_state`` trips on
        # ``new_cap < -1e-6`` (demand > cap + 1e-6). The previous threshold
        # left a ~9e-6-wide band where the mask passed but the trip fired —
        # survivable at n=5 (few samples per step) but blew up at n=25 (1250
        # samples/epoch) and in generalised training. The buffer here must
        # stay strictly greater than the trip tolerance so float32 noise
        # along the demand − cap axis cannot cross the gap.
        feasible = (
            mode_ok.bool()
            & (cap_margin >= 1e-5)
            & (raw_dist < 1e8)
            & (raw_pickup_delivery_dist < 1e8)
            & torch.isfinite(raw_dist)
            & torch.isfinite(raw_pickup_delivery_dist)
            & (raw_dist >= 0)
            & (raw_pickup_delivery_dist >= 0)
        )
        return pair_feat, feasible

    def _advance_assignment_state(
        self,
        batch,
        num_depots,
        num_uav,
        start_node,
        cur_time,
        capacity,
        veh_idx,
        req_idx,
        active,
    ):
        B, A = start_node.shape
        N = batch['x'].shape[1]
        R = (N - num_depots) // 2
        device = start_node.device
        safe_veh = veh_idx.clamp(min=0, max=max(A - 1, 0))
        safe_req = req_idx.clamp(min=0, max=max(R - 1, 0))
        bidx = torch.arange(B, device=device)

        cur_node = start_node.gather(1, safe_veh.unsqueeze(1)).squeeze(1)
        old_time = cur_time.gather(1, safe_veh.unsqueeze(1)).squeeze(1)
        old_cap = capacity.gather(1, safe_veh.unsqueeze(1)).squeeze(1)
        pickup_node = num_depots + safe_req

        demand = batch['demand'].view(B, N).float()[:, num_depots:num_depots + R]
        demand_sel = demand[bidx, safe_req]
        pickup_tw = batch['time_window'].view(B, N).float()[bidx, pickup_node]

        d_uav, d_adr = _masked_edge_dists(batch, B, N)
        dist_u = d_uav[bidx, cur_node, pickup_node]
        dist_a = d_adr[bidx, cur_node, pickup_node]
        is_uav = safe_veh < num_uav
        dist = torch.where(is_uav, dist_u, dist_a)
        if (active & (~torch.isfinite(dist) | (dist < 0) | (dist >= 1e9))).any():
            raise ValueError('Infeasible cooperative pickup travel')
        speed = torch.where(
            is_uav,
            torch.full_like(dist, V_UAV_MAX),
            torch.full_like(dist, V_ADR_MAX),
        )
        service = torch.where(
            is_uav,
            torch.full_like(dist, UAV_LAND_TAKEOFF_MIN),
            torch.zeros_like(dist),
        )
        pickup_arrival = old_time + dist / speed.clamp(min=1e-6)
        new_time = torch.maximum(pickup_arrival, pickup_tw) + service
        new_cap = old_cap - demand_sel

        # Capacity safety valve. The feasibility mask in
        # ``_pair_features`` is now tightened enough that this condition
        # should never fire. If a sample slips through anyway (data
        # corruption, mixed-precision edge case, future refactor), convert
        # the offender to a no-op by clearing its active flag — training
        # continues and the first occurrence is logged. The previous
        # behaviour (``raise ValueError``) killed multi-hour runs; a soft
        # fallback is safer for long training without hiding the signal.
        overflow = active & (~torch.isfinite(new_cap) | (new_cap < -1e-6))
        if overflow.any():
            _warn_capacity_overflow(old_cap, demand_sel, new_cap, overflow)
            active = active & ~overflow

        next_node = torch.where(active, pickup_node, cur_node)
        next_time = torch.where(active, new_time, old_time)
        next_cap = torch.where(active, new_cap, old_cap)

        start_node = start_node.clone()
        cur_time = cur_time.clone()
        capacity = capacity.clone()
        start_node.scatter_(1, safe_veh.unsqueeze(1), next_node.unsqueeze(1))
        cur_time.scatter_(1, safe_veh.unsqueeze(1), next_time.unsqueeze(1))
        capacity.scatter_(1, safe_veh.unsqueeze(1), next_cap.unsqueeze(1))
        return start_node, cur_time, capacity

    def forward(
        self,
        req: torch.Tensor,
        veh: torch.Tensor,
        glob: torch.Tensor,
        batch: Dict[str, torch.Tensor],
        num_depots: int,
        num_uav: int,
        num_adr: int,
        start_node: torch.Tensor,
        cur_time: torch.Tensor,
        capacity: torch.Tensor,
        available: torch.Tensor,
        request_open: torch.Tensor = None,
        request_owner: torch.Tensor = None,
        greedy: bool = False,
        max_assignments: int = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        B, R, H = req.shape
        A = veh.shape[1]
        max_assignments = R if max_assignments is None else min(max_assignments, R)
        if request_open is None:
            request_open = torch.ones(B, R, dtype=torch.bool, device=req.device)
        else:
            request_open = request_open.to(req.device).bool().view(B, R)
        if request_owner is not None:
            request_owner = request_owner.to(req.device).long().view(B, R)
            request_owner = request_owner.clamp(min=-1, max=max(A - 1, 0))
            agent_ids = torch.arange(A, device=req.device).view(1, A, 1)
            owner_ok = (
                (request_owner.unsqueeze(1) < 0)
                | (agent_ids == request_owner.unsqueeze(1))
            )
        else:
            owner_ok = torch.ones(B, A, R, dtype=torch.bool, device=req.device)
        assigned = ~request_open
        done = ~request_open.any(dim=1)
        logp_agents = torch.zeros(B, A, device=req.device)
        assignments = torch.full((B, max_assignments, 2), -1, dtype=torch.long, device=req.device)
        stop_idx = A * R

        for step in range(max_assignments):
            qv = self.vehicle_proj(veh)
            kr = self.request_proj(req)
            g = self.global_pair(glob).view(B, 1, 1, H)
            logits = (torch.tanh(qv.unsqueeze(2) + kr.unsqueeze(1) + g).sum(dim=-1)
                      / math.sqrt(H))
            pair_feat, feasible = self._pair_features(
                batch, num_depots, num_uav, num_adr, start_node, cur_time, capacity
            )
            logits = logits + self.pair_bias(pair_feat).squeeze(-1)
            feasible = (
                feasible
                & (~assigned.unsqueeze(1))
                & (available.unsqueeze(-1) > 0.5)
                & owner_ok
                & (~done.view(B, 1, 1))
            )
            flat_logits = logits.reshape(B, A * R)
            flat_valid = feasible.reshape(B, A * R)
            stop_logit = self.stop_head(glob)
            all_logits = torch.cat([flat_logits, stop_logit], dim=1)
            # Mandatory-service stop mask: the stop action is only
            # a valid choice when NO (vehicle, request) pair remains feasible.
            # Without this guard the free stop_head bias dominated on
            # mode-exclusive leftovers (e.g. an ADR-only req after UAVs served
            # everything else) because reward1 under mandatory service sets
            # unserved=0 and ALPHA_DELIV_BONUS=0, leaving only the PBRS k shaping
            # — too small to overcome ADR travel/TW cost. Hard-masking stop
            # while a feasible assignment exists aligns the decoder with the
            # mandatory contract.
            any_feasible = flat_valid.any(dim=1, keepdim=True)
            all_valid = torch.cat([flat_valid, ~any_feasible], dim=1)
            all_logits = all_logits.masked_fill(~all_valid, float('-inf'))
            stop_only = torch.full_like(all_logits, float('-inf'))
            stop_only[:, stop_idx] = 0.0
            all_logits = torch.where(done.unsqueeze(1), stop_only, all_logits)
            log_probs = F.log_softmax(all_logits.clamp(min=-1e9), dim=1)
            if greedy:
                choice = all_logits.argmax(dim=1)
            else:
                choice = torch.multinomial(log_probs.detach().exp().float(), 1).squeeze(1)
            choice = torch.where(done, torch.full_like(choice, stop_idx), choice)
            is_stop = choice == stop_idx
            pair_choice = choice.clamp(max=stop_idx - 1)
            veh_idx = pair_choice // R
            req_idx = pair_choice % R
            active = (~is_stop) & (~done)
            new_stop = is_stop & (~done)
            assignments[:, step, 0] = torch.where(active, veh_idx, torch.full_like(veh_idx, -1))
            assignments[:, step, 1] = torch.where(active, req_idx, torch.full_like(req_idx, -1))
            lp = log_probs.gather(1, choice.unsqueeze(1)).squeeze(1)
            logp_agents.scatter_add_(1, veh_idx.unsqueeze(1), (lp * active.float()).unsqueeze(1))
            if A > 0:
                logp_agents = logp_agents + (lp * new_stop.float()).unsqueeze(1) / float(A)
            assigned.scatter_(1, req_idx.unsqueeze(1), assigned.gather(1, req_idx.unsqueeze(1)) | active.unsqueeze(1))
            start_node, cur_time, capacity = self._advance_assignment_state(
                batch, num_depots, num_uav, start_node, cur_time, capacity,
                veh_idx, req_idx, active,
            )
            done = done | is_stop | assigned.all(dim=1)
            if bool(done.all().item()):
                break
        return assignments, logp_agents


def assignments_to_tours(
    assignments: torch.Tensor,
    batch: Dict[str, torch.Tensor],
    num_depots: int,
    num_uav: int,
    num_adr: int,
    n_depots_uav: int = None,
    onboard_assignments: torch.Tensor = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Convert new-pickup assignments plus optional onboard assignments to tours.

    ``onboard_assignments``: shape ``[B, R, 2]`` with
    ``(vehicle, request)`` pairs for requests currently in-flight. Each
    vehicle's tour PREPENDS the delivery nodes for its onboard requests
    (no pickup needed — already picked up) before the new (pickup, delivery)
    sequence from ``assignments``. Fixes the "Cooperative delivery-only
    residuals are not implemented" gate so coop works in rolling-horizon.
    """
    B = assignments.shape[0]
    A = num_uav + num_adr
    N = batch['x'].shape[1]
    R = (N - num_depots) // 2
    device = batch['x'].device
    if assignments.dtype != torch.long or assignments.dim() != 3 or assignments.shape[-1] != 2:
        raise ValueError('Invalid cooperative assignment shape or type')
    if A <= 0 or assignments.shape[0] != batch['x'].shape[0]:
        raise ValueError('Invalid cooperative assignment fleet')
    active_entries = (assignments[..., 0] >= 0) & (assignments[..., 1] >= 0)
    stopped = (assignments == -1).all(dim=-1)
    if (~(active_entries | stopped)).any() or (assignments[..., 0] >= A).any() or (assignments[..., 1] >= R).any():
        raise ValueError('Invalid cooperative request or vehicle index')
    if R:
        counts = torch.zeros(B, R, device=device, dtype=torch.long)
        counts.scatter_add_(1, assignments[..., 1].clamp(min=0), active_entries.long())
        if (counts > 1).any():
            raise ValueError('A cooperative request cannot be assigned twice')
    # Validate onboard_assignments if given.
    if onboard_assignments is not None:
        if onboard_assignments.shape != (B, R, 2) or onboard_assignments.dtype != torch.long:
            raise ValueError('onboard_assignments must be [B, R, 2] long')
        _ob_active = (onboard_assignments[..., 0] >= 0) & (onboard_assignments[..., 1] >= 0)
        # Onboard requests must not overlap new assignments (double-assign).
        if R:
            _ob_counts = torch.zeros(B, R, device=device, dtype=torch.long)
            _ob_counts.scatter_add_(
                1, onboard_assignments[..., 1].clamp(min=0), _ob_active.long())
            if (counts + _ob_counts > 1).any():
                raise ValueError('Request is both onboard and newly assigned')
    available = batch.get('available_mask')
    if available is not None:
        allowed = available.reshape(B, A).gather(1, assignments[..., 0].clamp(min=0)).bool()
        if (active_entries & ~allowed).any():
            raise ValueError('Cooperative assignment uses an unavailable vehicle')
    n_depots_uav = num_depots if n_depots_uav is None else n_depots_uav
    if 'start_node' in batch:
        start = batch['start_node'].view(B, A).long().to(device)
    else:
        starts = []
        for ag in range(num_uav):
            starts.append(ag % max(n_depots_uav, 1))
        n_dep_adr = max(num_depots - n_depots_uav, 1)
        for ag in range(num_adr):
            starts.append(n_depots_uav + ag % n_dep_adr)
        start = torch.tensor(starts, dtype=torch.long, device=device).view(1, A).expand(B, -1)

    seq_len = 1 + 2 * R
    tours = start.unsqueeze(-1).expand(B, A, seq_len).clone()
    lengths = torch.zeros(B, A, dtype=torch.long, device=device)
    # Onboard-delivery prefix: each vehicle's tour starts with
    # delivery nodes for any onboard requests, THEN its new pickups + new
    # deliveries. onboard_count_per_vehicle shifts the start of the
    # new-pickup segment.
    onboard_count_per_vehicle = torch.zeros(B, A, dtype=torch.long, device=device)
    if onboard_assignments is not None:
        ob_veh = onboard_assignments[:, :, 0]
        ob_req = onboard_assignments[:, :, 1]
        ob_active = (ob_veh >= 0) & (ob_req >= 0)
        if bool(ob_active.any().item()):
            ob_veh_safe = ob_veh.clamp(min=0, max=max(A - 1, 0))
            ob_veh_onehot = F.one_hot(ob_veh_safe, num_classes=A).to(torch.long)
            ob_veh_onehot = ob_veh_onehot * ob_active.unsqueeze(-1).long()
            ob_prev = ob_veh_onehot.cumsum(dim=1) - ob_veh_onehot
            ob_offset = ob_prev.gather(2, ob_veh_safe.unsqueeze(-1)).squeeze(-1)
            onboard_count_per_vehicle = ob_veh_onehot.sum(dim=1)
            ob_pos = 1 + ob_offset
            ob_valid = ob_active & (ob_pos < seq_len)
            if bool(ob_valid.any().item()):
                b_idx = torch.arange(B, device=device).unsqueeze(1).expand_as(ob_valid)[ob_valid]
                v_idx = ob_veh_safe[ob_valid]
                r_idx = ob_req[ob_valid]
                # Onboard requests already picked up — only need delivery node.
                tours[b_idx, v_idx, ob_pos[ob_valid]] = num_depots + R + r_idx
            lengths = lengths + onboard_count_per_vehicle
    veh = assignments[:, :, 0]
    req = assignments[:, :, 1]
    active = (veh >= 0) & (req >= 0)
    if bool(active.any().item()):
        veh_safe = veh.clamp(min=0, max=max(A - 1, 0))
        veh_onehot = F.one_hot(veh_safe, num_classes=A).to(torch.long)
        veh_onehot = veh_onehot * active.unsqueeze(-1).long()
        prev_for_vehicle = veh_onehot.cumsum(dim=1) - veh_onehot
        pickup_offset = prev_for_vehicle.gather(2, veh_safe.unsqueeze(-1)).squeeze(-1)
        total_for_vehicle = veh_onehot.sum(dim=1)
        lengths = lengths + 2 * total_for_vehicle
        assigned_total = total_for_vehicle.gather(1, veh_safe)
        # Shift new pickups/deliveries by this vehicle's onboard count so
        # onboard-delivery prefix is preserved.
        vehicle_onboard = onboard_count_per_vehicle.gather(1, veh_safe)
        pickup_pos = 1 + vehicle_onboard + pickup_offset
        delivery_pos = 1 + vehicle_onboard + assigned_total + pickup_offset
        valid = active & (pickup_pos < seq_len) & (delivery_pos < seq_len)
        if bool(valid.any().item()):
            b_idx = torch.arange(B, device=device).unsqueeze(1).expand_as(valid)[valid]
            v_idx = veh_safe[valid]
            r_idx = req[valid]
            tours[b_idx, v_idx, pickup_pos[valid]] = num_depots + r_idx
            tours[b_idx, v_idx, delivery_pos[valid]] = num_depots + R + r_idx

    # Padding is idle at the last real node, never an invented trip to depot 0.
    last_node = tours.gather(2, lengths.unsqueeze(-1))
    positions = torch.arange(seq_len, device=device).view(1, 1, -1)
    tours = torch.where(positions > lengths.unsqueeze(-1), last_node, tours)

    capacity = batch.get('start_capacity', batch['capacity']).view(B, A, 1).float()
    battery = batch.get('start_battery', batch['battery']).view(B, A, 1).float()
    T_t = batch.get('start_time')
    T_t = T_t.view(B, A, 1).float() if isinstance(T_t, torch.Tensor) else torch.zeros(B, A, 1, device=device)
    full_bat = batch['battery'].view(B, A).float()
    e = full_bat
    actions: List[torch.Tensor] = [tours[:, :, 0].unsqueeze(2)]
    times = []
    # Truncate at the last feasible step instead of hard-raising, since the
    # per-pair feasibility check doesn't verify chain battery margin.
    for s in range(1, seq_len):
        actions.append(tours[:, :, s].unsqueeze(2))
        try:
            cap_next, time_next, bat_next = update_state(
                batch['demand'].view(B, N),
                batch['time_window'].view(B, N),
                battery,
                T_t,
                capacity,
                e,
                num_uav,
                actions,
                batch['edge_attr_d'],
                batch['edge_attr_r'],
                num_depots=num_depots,
                max_capacity=batch['capacity'].view(B, A, 1).float(),
                wind=batch.get('wind'), coordinates=batch.get('coordinates'),
            )
        except ValueError:
            # Infeasible leg: freeze all vehicles at previous node (idle padding).
            prev = tours[:, :, s - 1:s]
            tours = torch.cat([tours[:, :, :s], prev.expand(B, A, seq_len - s)], dim=2)
            # Pad times to [B, A, seq_len - 1] with the last known clock.
            if times:
                last_t = times[-1]
            else:
                last_t = T_t
            while len(times) < seq_len - 1:
                times.append(last_t)
            break
        capacity = cap_next.unsqueeze(2)
        T_t = time_next.unsqueeze(2)
        battery = bat_next.unsqueeze(2)
        times.append(T_t)
    time_tensor = torch.cat(times, dim=2) if times else torch.zeros(B, A, 0, device=device)
    return tours, time_tensor


class CoopActorCritic(nn.Module):
    def __init__(self, input_node_dim: int = 15, hidden_dim: int = 128,
                 n_heads: int = 4, n_layers: int = 2):
        super().__init__()
        self.request_encoder = RequestPairEncoder(input_node_dim, hidden_dim)
        self.vehicle_encoder = VehicleStateEncoder(input_node_dim, hidden_dim)
        self.global_encoder = GlobalWindowEncoder(hidden_dim)
        self.coop_encoder = RelationAwareCoopEncoder(hidden_dim, n_heads=n_heads, n_layers=n_layers)
        self.assignment_decoder = AssignmentDecoder(hidden_dim)
        self.value_head = nn.Sequential(
            nn.Linear(hidden_dim * 3, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, batch: Dict[str, torch.Tensor], num_uav: int, num_adr: int,
                greedy: bool = False, max_assignments: int = None,
                initial_visited: torch.Tensor = None) -> Dict[str, torch.Tensor]:
        B = batch['x'].shape[0]
        num_depots = _as_scalar_int(batch.get('n_depots', None), default=2)
        n_depots_uav = _as_scalar_int(batch.get('n_depots_uav', None), default=num_depots)
        req = self.request_encoder(batch, num_depots)
        veh, start_node, cur_time, cap, _ = self.vehicle_encoder(
            batch, num_uav, num_adr, num_depots, n_depots_uav=n_depots_uav
        )
        glob, window_logits = self.global_encoder(batch)
        req, veh, glob = self.coop_encoder(req, veh, glob)
        available = batch.get('available_mask')
        if isinstance(available, torch.Tensor):
            available = available.view(B, num_uav + num_adr).float().to(batch['x'].device)
        else:
            available = torch.ones(B, num_uav + num_adr, device=batch['x'].device)
        if initial_visited is None:
            initial_visited = batch.get('initial_visited', None)
        request_open = None
        onboard_assignments = None  # Mid-shift onboard support
        if isinstance(initial_visited, torch.Tensor):
            N = batch['x'].shape[1]
            R = (N - num_depots) // 2
            iv = initial_visited.view(B, N).to(batch['x'].device)
            pickup_visited = iv[:, num_depots:num_depots + R] > .5
            delivery_pending = iv[:, num_depots + R:num_depots + 2 * R] <= 0.5
            onboard = pickup_visited & delivery_pending   # [B, R]
            # New-assignment pool EXCLUDES onboard requests (they're already
            # committed to their current carrier).
            request_open = delivery_pending & ~onboard
            if onboard.any():
                # Build (vehicle, request) pairs from delivery_owner so
                # assignments_to_tours can prepend onboard deliveries.
                _owner = batch.get('delivery_owner')
                if _owner is None:
                    raise ValueError(
                        'Mid-shift coop needs delivery_owner to know which vehicle '
                        'holds each onboard request. Dispatcher must populate this field.')
                _owner = _owner.to(req.device).long().view(B, R)
                onboard_assignments = torch.full((B, R, 2), -1,
                                                  dtype=torch.long,
                                                  device=req.device)
                onboard_assignments[..., 0] = torch.where(onboard, _owner,
                                                           torch.full_like(_owner, -1))
                onboard_assignments[..., 1] = torch.where(
                    onboard,
                    torch.arange(R, device=req.device).unsqueeze(0).expand(B, R),
                    torch.full((B, R), -1, dtype=torch.long, device=req.device))
        request_owner = batch.get('delivery_owner', None)
        assignments, log_p = self.assignment_decoder(
            req, veh, glob, batch, num_depots, num_uav, num_adr,
            start_node, cur_time, cap, available,
            request_open=request_open, request_owner=request_owner,
            greedy=greedy, max_assignments=max_assignments,
        )
        tours, time_tensor = assignments_to_tours(
            assignments, batch, num_depots, num_uav, num_adr, n_depots_uav=n_depots_uav,
            onboard_assignments=onboard_assignments,
        )
        value = self.value_head(torch.cat([
            req.mean(dim=1),
            veh.mean(dim=1),
            glob,
        ], dim=1)).squeeze(1)
        return {
            'assignments': assignments,
            'tour_idx': tours,
            'log_p': log_p,
            'time': time_tensor,
            'value': value,
            'window_logits': window_logits,
        }
