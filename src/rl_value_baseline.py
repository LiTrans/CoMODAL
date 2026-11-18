import torch
import torch.nn as nn


GLOBAL_FEATURE_DIM = 12


def attach_window_context(
    batch: dict,
    initial_visited,
    n_vis,
    n_req: int,
    n_uav: int,
    n_adr: int,
    t_current: float,
    window_length: float,
    shift_minutes: float = 120.0,
) -> dict:
    """Attach compact rolling-window context used by the learned critic.

    This is intentionally a small feature vector so it improves state value
    learning without forcing the actor to learn a large extra interface yet.
    """
    if not isinstance(batch.get('x'), torch.Tensor):
        return batch

    out = dict(batch)
    x = out['x']
    device = x.device
    B = x.shape[0]
    n_agents = max(n_uav + n_adr, 1)

    if isinstance(n_vis, torch.Tensor):
        n_vis_f = n_vis.to(device).float().view(B)
    else:
        n_vis_f = torch.full((B,), float(n_vis), device=device)

    if initial_visited is not None:
        active_nodes = (initial_visited.to(device) <= 0.5).float()
        # depots are not requests; count visible customer nodes only
        active_customer_frac = active_nodes[:, -2 * n_req:].mean(dim=1)
    else:
        active_customer_frac = torch.ones(B, device=device)

    demand = out.get('demand')
    if isinstance(demand, torch.Tensor):
        d = demand.to(device).view(B, -1)
        positive = d.clamp(min=0.0)
        active_req = positive > 0
        mean_demand = positive.sum(dim=1) / active_req.float().sum(dim=1).clamp(min=1.0)
        heavy_frac = (positive > 5.0).float().sum(dim=1) / active_req.float().sum(dim=1).clamp(min=1.0)
    else:
        mean_demand = torch.zeros(B, device=device)
        heavy_frac = torch.zeros(B, device=device)

    available = out.get('available_mask')
    if isinstance(available, torch.Tensor):
        avail = available.to(device).float().view(B, n_agents)
    else:
        avail = torch.ones(B, n_agents, device=device)

    uav_avail = avail[:, :n_uav].mean(dim=1) if n_uav > 0 else torch.zeros(B, device=device)
    adr_avail = avail[:, n_uav:].mean(dim=1) if n_adr > 0 else torch.zeros(B, device=device)

    cap = out.get('start_capacity', out.get('capacity'))
    bat = out.get('start_battery', out.get('battery'))
    full_cap = out.get('capacity')
    full_bat = out.get('battery')
    if isinstance(cap, torch.Tensor) and isinstance(full_cap, torch.Tensor):
        cap_ratio = cap.to(device).view(B, n_agents) / full_cap.to(device).view(B, n_agents).clamp(min=1.0)
        cap_mean = cap_ratio.mean(dim=1)
    else:
        cap_mean = torch.ones(B, device=device)
    if isinstance(bat, torch.Tensor) and isinstance(full_bat, torch.Tensor):
        bat_ratio = bat.to(device).view(B, n_agents) / full_bat.to(device).view(B, n_agents).clamp(min=1.0)
        bat_mean = bat_ratio.mean(dim=1)
    else:
        bat_mean = torch.ones(B, device=device)

    out['global_features'] = torch.stack([
        torch.full((B,), float(t_current) / shift_minutes, device=device),
        torch.full((B,), float(window_length) / shift_minutes, device=device),
        n_vis_f / max(float(n_req), 1.0),
        n_vis_f / float(n_agents),
        active_customer_frac,
        avail.mean(dim=1),
        uav_avail,
        adr_avail,
        cap_mean,
        bat_mean,
        mean_demand / 10.0,
        heavy_frac,
    ], dim=1)
    pickup_tw = out.get('time_window')
    if isinstance(pickup_tw, torch.Tensor):
        tw = pickup_tw.to(device).view(B, -1)
        customer_tw = tw[:, -2 * n_req:] if n_req > 0 else tw
        active_customer = active_nodes[:, -2 * n_req:] if initial_visited is not None else torch.ones_like(customer_tw)
        active_tw = customer_tw.masked_fill(active_customer <= 0.5, float('inf'))
        min_slack = (active_tw.min(dim=1)[0] - float(t_current)) / shift_minutes
        min_slack = torch.where(torch.isfinite(min_slack), min_slack, torch.ones_like(min_slack))
        min_slack = min_slack.clamp(-1.0, 1.0)
    else:
        min_slack = torch.ones(B, device=device)
    out['arrival_history'] = torch.stack([
        torch.full((B,), float(window_length) / shift_minutes, device=device),
        torch.full((B,), float(window_length) / max(float(n_req), 1.0), device=device).clamp(0.0, 1.0),
        n_vis_f / max(float(n_req), 1.0),
        active_customer_frac,
        torch.zeros(B, device=device),
        avail.mean(dim=1),
        min_slack,
        torch.full((B,), float(t_current) / shift_minutes, device=device),
    ], dim=1).unsqueeze(1)
    return out


class RollingStateCritic(nn.Module):
    """Permutation-tolerant value baseline for rolling dispatch states."""

    def __init__(self, input_node_dim: int, hidden_dim: int = 128,
                 global_dim: int = GLOBAL_FEATURE_DIM):
        super().__init__()
        self.node_proj = nn.Sequential(
            nn.Linear(input_node_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
        )
        self.global_proj = nn.Sequential(
            nn.Linear(global_dim + 8, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
        )
        self.value = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, batch: dict, initial_visited=None,
                num_uav: int = 0, num_adr: int = 0) -> torch.Tensor:
        x = batch['x'].float()
        B, N, _ = x.shape
        device = x.device

        if initial_visited is not None:
            active = (initial_visited.to(device).view(B, N) <= 0.5).float()
        else:
            active = torch.ones(B, N, device=device)
        node_emb = self.node_proj(x)
        node_pool = (node_emb * active.unsqueeze(-1)).sum(dim=1) / active.sum(dim=1, keepdim=True).clamp(min=1.0)

        n_agents = max(num_uav + num_adr, 1)
        available = batch.get('available_mask')
        if isinstance(available, torch.Tensor):
            avail = available.to(device).float().view(B, n_agents)
        else:
            avail = torch.ones(B, n_agents, device=device)

        start_time = batch.get('start_time')
        if isinstance(start_time, torch.Tensor):
            time_mean = start_time.to(device).view(B, n_agents).mean(dim=1) / 120.0
        else:
            time_mean = torch.zeros(B, device=device)

        capacity = batch.get('start_capacity', batch.get('capacity'))
        full_capacity = batch.get('capacity')
        battery = batch.get('start_battery', batch.get('battery'))
        full_battery = batch.get('battery')
        if isinstance(capacity, torch.Tensor) and isinstance(full_capacity, torch.Tensor):
            cap_ratio = capacity.to(device).view(B, n_agents) / full_capacity.to(device).view(B, n_agents).clamp(min=1.0)
            cap_mean = cap_ratio.mean(dim=1)
        else:
            cap_mean = torch.ones(B, device=device)
        if isinstance(battery, torch.Tensor) and isinstance(full_battery, torch.Tensor):
            bat_ratio = battery.to(device).view(B, n_agents) / full_battery.to(device).view(B, n_agents).clamp(min=1.0)
            bat_mean = bat_ratio.mean(dim=1)
        else:
            bat_mean = torch.ones(B, device=device)

        fleet_context = torch.stack([
            avail.mean(dim=1),
            avail[:, :num_uav].mean(dim=1) if num_uav > 0 else torch.zeros(B, device=device),
            avail[:, num_uav:].mean(dim=1) if num_adr > 0 else torch.zeros(B, device=device),
            cap_mean,
            bat_mean,
            time_mean,
            torch.full((B,), float(num_uav) / float(n_agents), device=device),
            torch.full((B,), float(num_adr) / float(n_agents), device=device),
        ], dim=1)

        global_features = batch.get('global_features')
        if not isinstance(global_features, torch.Tensor):
            global_features = torch.zeros(B, GLOBAL_FEATURE_DIM, device=device)
        gf = torch.cat([global_features.to(device).float(), fleet_context], dim=1)
        global_emb = self.global_proj(gf)
        return self.value(torch.cat([node_pool, global_emb], dim=1)).squeeze(1)
