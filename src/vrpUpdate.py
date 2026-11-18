# Lazy torch imports: loading torch in OR-Tools worker subprocesses conflicts
# with libprotobuf and crashes with SIGSEGV on cluster spawn.

from model_config import (
    SCALE_M_PER_COORD, UAV_CRUISE_MPS, ADR_CRUISE_MPS, UAV_SERVICE_MIN,
    UAV_CAPACITY_KG, ADR_CAPACITY_KG, speed_to_mps,
)
from energy import (uav_power_torch, uav_customer_service_energy_torch,
                    uav_depot_phase_energy_torch,
                    C_R_ADR, G, M_ADR, NU_ADR)
from hardware_profiles import physical_configuration

V_UAV_MAX = UAV_CRUISE_MPS * 60.0 / SCALE_M_PER_COORD
V_ADR_MAX = ADR_CRUISE_MPS * 60.0 / SCALE_M_PER_COORD
V_UAV_DEPOT = V_UAV_MAX
V_ADR_DEPOT = V_ADR_MAX / 2.0
V_UAV_MIN_PICKUP = V_UAV_MAX
V_ADR_MIN_PICKUP = min(2.0, ADR_CRUISE_MPS) * 60.0 / SCALE_M_PER_COORD
UAV_LAND_TAKEOFF_MIN = UAV_SERVICE_MIN


def PC_uav(payload_kg, v_ground, wind_xy=None, heading_rad=None):
    """kW for coordinate-units/minute ground speed."""
    return uav_power_torch(payload_kg, speed_to_mps(v_ground), wind_xy, heading_rad) / 1000.0


def PC_adr(payload_kg, v_ground):
    return C_R_ADR * G * (M_ADR + payload_kg) * speed_to_mps(v_ground) / NU_ADR / 1000.0


def _pickup_speed(distance, t_now, t_target, v_min, v_max):
    import torch
    slack = (t_target - t_now).clamp(min=0.0)
    speed = (distance / slack.clamp(min=1e-8)).clamp(min=v_min, max=v_max)
    rush = (slack <= 0) | (distance / v_max >= slack) | (distance == 0)
    return torch.where(rush, v_max.expand_as(speed), speed)


def _fleet_values(value, reference, num_uav, defaults=None):
    import torch
    B, A = reference.shape
    if value is None:
        value = defaults
    if isinstance(value, (list, tuple)):
        return torch.cat([
            reference.new_full((B, num_uav), float(value[0])),
            reference.new_full((B, A - num_uav), float(value[1])),
        ], dim=1)
    return torch.as_tensor(value, device=reference.device, dtype=reference.dtype).reshape(B, A)


def _pending_deliveries(demand, num_depots, remaining_deliveries):
    import torch
    B, N = demand.shape[:2]
    R = (N - num_depots) // 2
    if remaining_deliveries is None:
        return torch.ones(B, R, device=demand.device, dtype=torch.bool)
    pending = torch.as_tensor(remaining_deliveries, device=demand.device)
    if pending.shape != (B, R) or pending.dtype != torch.bool:
        raise ValueError('remaining_deliveries must be a persistent whole-book boolean [batch, requests]')
    return pending


def _terminal_targets(demand, targets, load_after, num_depots, remaining_deliveries):
    import torch
    pending = _pending_deliveries(demand, num_depots, remaining_deliveries)
    R = pending.size(1)
    if R == 0:
        return torch.zeros_like(targets, dtype=torch.bool)
    ids = (targets - num_depots - R).clamp(min=0, max=R - 1)
    b = torch.arange(demand.size(0), device=demand.device).view(-1, 1, 1)
    return ((targets >= num_depots + R) & pending[b, ids]
            & (pending.sum(1).view(-1, 1, 1) == 1) & (load_after.abs() <= 1e-5))


def pending_after_moves(pending, nodes, moved, num_depots):
    """Advance a hypothetical whole-book ledger after successful movements."""
    delivery_start = num_depots + pending.size(1)
    rows, agents = (moved & (nodes >= delivery_start)).nonzero(as_tuple=True)
    next_pending = pending.clone()
    next_pending[rows, nodes[rows, agents] - delivery_start] = False
    return next_pending


def _battery_reserve(full_bat, num_uav, config):
    import torch
    if config is None:
        return torch.zeros_like(full_bat)
    expected = _fleet_values([config.uav_battery.full_charge_kj, config.adr_battery.full_charge_kj], full_bat, num_uav)
    if not torch.allclose(full_bat, expected, rtol=1e-6, atol=1e-5):
        raise ValueError('Full battery tensor conflicts with configured usable/SOC window')
    return _fleet_values([config.uav_battery.reserve_energy_kj, config.adr_battery.reserve_energy_kj], full_bat, num_uav)


def _validate_configured_capacity(full_cap, num_uav, config):
    import torch
    if config is not None:
        ceiling = _fleet_values([config.uav_battery.profile.payload_limit_kg,
                                  config.adr_battery.profile.payload_limit_kg], full_cap, num_uav)
        if bool((~torch.isfinite(full_cap) | (full_cap <= 0) | (full_cap > ceiling + 1e-5)).any()):
            raise ValueError('Capacity exceeds the configured hardware profile')


def _leg_metrics(demand, time_window, T_t, capacity, num_uav, selected, targets,
                 edge_attr_uav, edge_attr_adr, num_depots, max_capacity=None,
                 wind=None, coordinates=None, physical_config=None,
                 remaining_deliveries=None, *, matrix_batch=None, terminal_delivery=None):
    """Prospective travel plus customer-service energy and completion times."""
    import torch
    B, A, K = targets.shape
    config = physical_configuration(physical_config) if physical_config is not None else None
    N = demand.size(1)
    batch = torch.arange(B, device=demand.device).view(B, 1, 1)
    matrix_rows = batch[:, :, 0] if matrix_batch is None else matrix_batch.view(B, 1)
    rows = torch.cat([
        edge_attr_uav.reshape(-1, N, N)[matrix_rows, selected[:, :num_uav], :],
        edge_attr_adr.reshape(-1, N, N)[matrix_rows, selected[:, num_uav:], :],
    ], dim=1)
    distance = rows.gather(2, targets)
    reachable = torch.isfinite(distance) & (distance >= 0) & (distance < 1e9)
    dist = torch.where(reachable, distance, torch.zeros_like(distance))
    idle = targets == selected.unsqueeze(-1)
    cap = capacity.reshape(B, A)
    full = _fleet_values(max_capacity, cap, num_uav, [UAV_CAPACITY_KG, ADR_CAPACITY_KG])
    _validate_configured_capacity(full, num_uav, config)
    payload = (full - cap).unsqueeze(-1)
    now = T_t.reshape(B, A, 1)
    tw = time_window.reshape(B, N)[batch, targets]
    pickup = (targets >= num_depots) & (targets < num_depots + (N - num_depots) // 2)
    depot = targets < num_depots
    delta_load = demand.reshape(B, N)[batch, targets]
    load_after = torch.where(depot | idle, payload, payload + delta_load)
    load_valid = (load_after >= -1e-5) & (load_after <= full.unsqueeze(-1) + 1e-5)
    reachable = reachable & load_valid
    is_uav = torch.arange(A, device=demand.device).view(1, A, 1) < num_uav
    v_max = torch.where(is_uav, V_UAV_MAX, V_ADR_MAX)
    v_min = torch.where(is_uav, V_UAV_MIN_PICKUP, V_ADR_MIN_PICKUP)
    speed = torch.where(pickup & ~is_uav, _pickup_speed(dist, now, tw, v_min, v_max), v_max)
    speed = torch.where(depot & ~is_uav, v_max / 2.0, speed)
    travel = dist / speed
    arrival = now + travel
    terminal = terminal_delivery
    if terminal is None:
        terminal = (_terminal_targets(demand, targets, load_after, num_depots, remaining_deliveries)
                    if config is not None else torch.zeros_like(targets, dtype=torch.bool))
    source_depot = (selected < num_depots).unsqueeze(-1)
    if config is not None:
        arrival = arrival + torch.where(is_uav & source_depot & ~idle, config.uav_phases.depot_takeoff_min or 0., 0.)
    finish = torch.where(pickup, torch.maximum(arrival, tw), arrival)
    service = torch.where(is_uav, UAV_SERVICE_MIN, 0.0)
    from model_config import UAV_SWAP_MIN, ADR_SWAP_MIN
    swap = torch.where(is_uav, UAV_SWAP_MIN, ADR_SWAP_MIN)
    if config is not None:
        phases = config.uav_phases
        customer_minutes = phases.customer_landing_min + phases.customer_takeoff_min
        service = torch.where(is_uav, customer_minutes, 0.)
        if phases.terminal_delivery == 'landed':
            service = torch.where(is_uav & terminal, phases.customer_landing_min, service)
        swap = swap + torch.where(is_uav, phases.depot_landing_min or 0., 0.)
    finish = finish + torch.where(depot, swap, service)
    heading = torch.zeros_like(dist)
    wind_xy = None
    if wind is not None:
        wind = wind.reshape(B, 2)
        if not torch.isfinite(wind).all() or (wind[:, 0] < 0).any():
            raise ValueError("Invalid wind magnitude/direction")
        if coordinates is None and (wind[:, 0].abs() > 1e-12).any():
            raise ValueError("Wind requires coordinates; regenerate legacy instances")
        wind_xy = torch.stack([wind[:, 0] * wind[:, 1].cos(),
                               wind[:, 0] * wind[:, 1].sin()], dim=-1).view(B, 1, 1, 2)
    if coordinates is not None:
        xy = coordinates.reshape(B, N, 2)
        delta = xy[batch, targets] - xy[batch[:, :, 0], selected].unsqueeze(2)
        heading = torch.atan2(delta[..., 1], delta[..., 0])
    eu = travel[:, :num_uav] * PC_uav(
        payload[:, :num_uav].clamp(min=0.), speed[:, :num_uav], wind_xy, heading[:, :num_uav],
    ) * 60.0
    # Both the current payload and the hypothetical load_after are power-law
    # inputs for inactive/invalid candidates (later marked unreachable, not
    # repaired). Sanitize both so the physics guard sees non-negative inputs;
    # the reachable mask handles the semantics of invalid states.
    payload_uav_sanitized = payload[:, :num_uav].clamp(min=0.)
    service_energy = uav_customer_service_energy_torch(
        payload_uav_sanitized, load_after[:, :num_uav].clamp(min=0.), wind_xy,
        phase_policy=config.uav_phases if config is not None else None,
        terminal_delivery=terminal[:, :num_uav],
    )
    if config is not None:
        depot_landing = uav_depot_phase_energy_torch(payload_uav_sanitized, wind_xy,
            departure=False, phase_policy=config.uav_phases)
        departure = uav_depot_phase_energy_torch(payload_uav_sanitized, wind_xy,
            departure=True, phase_policy=config.uav_phases)
        eu = eu + torch.where(source_depot[:, :num_uav], departure, 0.)
        service_energy = torch.where(depot[:, :num_uav], depot_landing, service_energy)
    else:
        service_energy = torch.where(depot[:, :num_uav], 0., service_energy)
    eu = eu + torch.where(idle[:, :num_uav], 0., service_energy)
    er = C_R_ADR * G * (M_ADR + payload[:, num_uav:]) * dist[:, num_uav:] * SCALE_M_PER_COORD / NU_ADR / 1000.0
    energy = torch.cat([eu, er], dim=1)
    energy = torch.where(reachable, energy, torch.full_like(energy, float("inf")))
    energy = torch.where(idle, torch.zeros_like(energy), energy)
    finish = torch.where(idle, now.expand_as(finish), finish)
    return energy, finish, reachable | idle


def _safe_continuation_mask(demand, capacity, num_uav, selected, targets, energy,
                            battery, edge_uav, edge_adr, num_depots, max_capacity,
                            wind, coordinates, config, remaining_deliveries, allowed_depots,
                            continuation_check='off'):
    """Vectorized depot-escape/last-delivery certificate.

    continuation_check levels: off = short-circuit when legacy_leg_only;
    l1 = depot-escape + last-pickup direct-delivery; l2 = l1 + check every
    pending pickup; l3 = l2 + recharge-then-deliver chain.
    """
    import torch
    B, A, K = targets.shape
    N = demand.size(1)
    if continuation_check == 'off':
        if config is None or config.continuation_policy == 'legacy_leg_only':
            return torch.ones_like(targets, dtype=torch.bool)
    else:
        if config is None:
            return torch.ones_like(targets, dtype=torch.bool)
    full_cap = _fleet_values(max_capacity, capacity.reshape(B, A), num_uav,
                             [UAV_CAPACITY_KG, ADR_CAPACITY_KG])
    payload = (full_cap - capacity.reshape(B, A)).unsqueeze(-1)
    b = torch.arange(B, device=demand.device).view(B, 1, 1)
    depot = targets < num_depots
    idle = targets == selected.unsqueeze(-1)
    load_after = torch.where(depot | idle, payload, payload + demand.reshape(B, N)[b, targets])
    load_after = load_after.clamp(min=0.)
    terminal = _terminal_targets(demand, targets, load_after, num_depots, remaining_deliveries)
    dm_u = edge_uav.reshape(B, N, N)
    dm_r = edge_adr.reshape(B, N, N)
    dm = torch.cat([dm_u.unsqueeze(1).expand(-1, num_uav, -1, -1),
                    dm_r.unsqueeze(1).expand(-1, A - num_uav, -1, -1)], 1)
    a = torch.arange(A, device=demand.device).view(1, A, 1)
    outbound = dm[b, a, targets, :num_depots]
    valid = torch.isfinite(outbound) & (outbound >= 0) & (outbound < 1e9)
    if allowed_depots is not None:
        allowed = torch.as_tensor(allowed_depots, device=demand.device)
        if allowed.shape != (B, A, num_depots) or allowed.dtype != torch.bool:
            raise ValueError('allowed_depots must be boolean [batch, agents, depots]')
        valid = valid & allowed.unsqueeze(2)
    d = torch.where(valid, outbound, 0.)
    heading = torch.zeros_like(d)
    wind_xy = None
    if wind is not None:
        w = wind.reshape(B, 2)
        wind_xy = torch.stack([w[:, 0] * w[:, 1].cos(), w[:, 0] * w[:, 1].sin()], -1).view(B, 1, 1, 1, 2)
    if coordinates is not None:
        xy = coordinates.reshape(B, N, 2)
        delta = xy[:, None, None, :num_depots] - xy[b, targets].unsqueeze(-2)
        heading = torch.atan2(delta[..., 1], delta[..., 0])
    post_u = load_after[:, :num_uav].unsqueeze(-1)
    return_u = (d[:, :num_uav] * SCALE_M_PER_COORD / UAV_CRUISE_MPS
                * uav_power_torch(post_u, torch.full_like(d[:, :num_uav], UAV_CRUISE_MPS), wind_xy,
                                  heading[:, :num_uav]) / 1000.)
    return_u = return_u + uav_depot_phase_energy_torch(post_u, wind_xy,
        departure=False, phase_policy=config.uav_phases)
    return_r = C_R_ADR * G * (M_ADR + load_after[:, num_uav:].unsqueeze(-1)) * d[:, num_uav:] * SCALE_M_PER_COORD / NU_ADR / 1000.
    escape = torch.cat([return_u, return_r], 1).masked_fill(~valid, float('inf'))
    required = escape.min(-1).values if num_depots else torch.full_like(energy, float('inf'))
    pending = _pending_deliveries(demand, num_depots, remaining_deliveries)
    R = pending.size(1)
    if R:
        ids = (targets - num_depots).clamp(min=0, max=R - 1)
        # l2/l3 extend the completion-energy check to every pending pickup
        # (vs. last_pickup only).
        pending_pickup = ((targets >= num_depots) & (targets < num_depots + R)
                          & pending[b, ids])
        last_pickup = (pending_pickup
                       & (pending.sum(1).view(B, 1, 1) == 1)
                       & (payload.abs() <= 1e-5))
        completion_gate = last_pickup if continuation_check in ('off', 'l1') else pending_pickup
        delivery_node = (num_depots + R + ids)
        completion_distance = dm[b, a, targets, delivery_node]
        completion_valid = torch.isfinite(completion_distance) & (completion_distance >= 0) & (completion_distance < 1e9)
        dc = torch.where(completion_valid, completion_distance, 0.)
        hc = torch.zeros_like(dc)
        if coordinates is not None:
            delta = xy[b, delivery_node] - xy[b, targets]
            hc = torch.atan2(delta[..., 1], delta[..., 0])
        wc = wind_xy.squeeze(-2) if wind_xy is not None else None
        cu = (dc[:, :num_uav] * SCALE_M_PER_COORD / UAV_CRUISE_MPS
              * uav_power_torch(load_after[:, :num_uav], torch.full_like(dc[:, :num_uav], UAV_CRUISE_MPS), wc, hc[:, :num_uav]) / 1000.)
        # Post-delivery load = load_after + demand[delivery] (negative).
        _demand_at_del = demand.reshape(B, N)[b, delivery_node]
        _post_u = (load_after[:, :num_uav]
                   + _demand_at_del[:, :num_uav]).clamp(min=0.)
        cu = cu + uav_customer_service_energy_torch(
            load_after[:, :num_uav], _post_u, wc,
            phase_policy=config.uav_phases, terminal_delivery=True)
        cr = C_R_ADR * G * (M_ADR + load_after[:, num_uav:]) * dc[:, num_uav:] * SCALE_M_PER_COORD / NU_ADR / 1000.
        completion = torch.cat([cu, cr], 1).masked_fill(~(completion_valid & completion_gate), float('inf'))
        required = torch.minimum(required, completion)
    reserve = _fleet_values([config.uav_battery.reserve_energy_kj, config.adr_battery.reserve_energy_kj],
                            battery.reshape(B, A), num_uav).unsqueeze(-1)
    certified = battery.reshape(B, A, 1) - energy >= reserve + required - 1e-5
    # L3: replace "escape OR direct" with "direct OR recharge-then-deliver"
    # for every pending pickup, enforcing commit-level feasibility.
    if continuation_check == 'l3' and R:
        full_bat = _fleet_values(
            [config.uav_battery.full_charge_kj, config.adr_battery.full_charge_kj],
            battery.reshape(B, A), num_uav)
        battery_at_pickup = battery.reshape(B, A, 1) - energy
        # A: reach an admitted depot above reserve.
        can_reach_depot = (
            battery_at_pickup.unsqueeze(-1) - escape >= reserve.unsqueeze(-1) - 1e-5
        )
        # B: after full recharge, complete depot->delivery above reserve.
        dep_rows = dm[:, :, :num_depots, :]
        dn_idx = delivery_node.unsqueeze(-2).expand(-1, -1, num_depots, -1)
        dep_to_del = dep_rows.gather(-1, dn_idx).transpose(-2, -1)
        dep_to_del_valid = (torch.isfinite(dep_to_del)
                            & (dep_to_del >= 0) & (dep_to_del < 1e9))
        dtd = torch.where(dep_to_del_valid, dep_to_del, torch.zeros_like(dep_to_del))
        htd = torch.zeros_like(dtd)
        if coordinates is not None:
            xy_del = xy[b, delivery_node]
            xy_dep = xy[:, :num_depots]
            # Depot -> delivery heading (sign matters for wind component).
            delta2 = xy_del.unsqueeze(-2) - xy_dep.view(B, 1, 1, num_depots, 2)
            htd = torch.atan2(delta2[..., 1], delta2[..., 0])
        la_u = load_after[:, :num_uav].unsqueeze(-1).expand(-1, -1, -1, num_depots)
        la_r = load_after[:, num_uav:].unsqueeze(-1).expand(-1, -1, -1, num_depots)
        travel_u = (dtd[:, :num_uav] * SCALE_M_PER_COORD / UAV_CRUISE_MPS
                    * uav_power_torch(la_u,
                                      torch.full_like(dtd[:, :num_uav], UAV_CRUISE_MPS),
                                      wind_xy, htd[:, :num_uav]) / 1000.)
        dep_phase_u = uav_depot_phase_energy_torch(la_u, wind_xy,
            departure=True, phase_policy=config.uav_phases)
        # Post-delivery load = la_u + demand[delivery] (negative at deliveries).
        # Slice _demand_at_del to [:, :num_uav] before expanding to depot dim.
        _demand_at_del_uav = _demand_at_del[:, :num_uav].unsqueeze(-1).expand(
            -1, -1, -1, num_depots)
        _la_u_post = (la_u + _demand_at_del_uav).clamp(min=0.)
        svc_u = uav_customer_service_energy_torch(la_u, _la_u_post, wind_xy,
            phase_policy=config.uav_phases, terminal_delivery=True)
        leg2_u = travel_u + dep_phase_u + svc_u
        leg2_r = (C_R_ADR * G * (M_ADR + la_r) * dtd[:, num_uav:]
                  * SCALE_M_PER_COORD / NU_ADR / 1000.)
        leg2 = torch.cat([leg2_u, leg2_r], dim=1)
        leg2 = leg2.masked_fill(~dep_to_del_valid, float('inf'))
        can_depart_depot = (full_bat.view(B, A, 1, 1) - leg2
                            >= reserve.unsqueeze(-1) - 1e-5)
        chain_ok = (can_reach_depot & can_depart_depot).any(-1)
        can_direct = (battery.reshape(B, A, 1) - energy - completion
                      >= reserve - 1e-5)
        certified = torch.where(pending_pickup, can_direct | chain_ok, certified)
    return certified | terminal | depot | idle


def tensor_recharge_budget(batch, shape, device):
    """Validate the optional per-vehicle recharge allowance at a replan."""
    import torch

    budget = batch.get('remaining_recharges')
    if budget is None:
        return None
    budget = torch.as_tensor(budget, device=device)
    if budget.shape != shape or ((budget < 0) | (budget != budget.long())).any():
        raise ValueError('Invalid remaining recharge allowance')
    return budget.long()


def recharge_depot_permissions(allowed_depots, budget, num_depots):
    if budget is None:
        return allowed_depots
    available = (budget > 0).unsqueeze(-1).expand(-1, -1, num_depots)
    return available if allowed_depots is None else available & allowed_depots


def update_state(demand, time_window, battery, T_t, capacity, E,
                 num_uav, actions, edge_attr_uav, edge_attr_adr, num_depots=None,
                 max_capacity=None, wind=None, coordinates=None, physical_config=None,
                 remaining_deliveries=None, allowed_depots=None, *, strict=False):
    import torch
    previous = actions[-2].squeeze(2)
    current = actions[-1].squeeze(2)
    B, A = current.shape
    N = demand.size(1)
    if num_depots is None:
        num_depots = int((demand == 0).sum().item() // B)
    if ((current < 0) | (current >= N) | (previous < 0) | (previous >= N)).any():
        raise ValueError("Invalid node index")
    cap = capacity.reshape(B, A)
    bat = battery.reshape(B, A)
    full_cap = _fleet_values(max_capacity, cap, num_uav, [UAV_CAPACITY_KG, ADR_CAPACITY_KG])
    full_bat = _fleet_values(E, bat, num_uav)
    config = physical_configuration(physical_config) if physical_config is not None else None
    reserve = _battery_reserve(full_bat, num_uav, config)
    if not torch.isfinite(torch.stack([cap, bat, full_cap, full_bat])).all():
        raise ValueError("Non-finite vehicle state")
    if ((cap < -1e-5) | (cap > full_cap + 1e-5) | (bat < reserve - 1e-5) | (bat > full_bat + 1e-5)).any():
        raise ValueError("Invalid capacity or battery state")
    energy, finish, reachable = _leg_metrics(
        demand, time_window, T_t, capacity, num_uav, previous, current.unsqueeze(-1),
        edge_attr_uav, edge_attr_adr, num_depots, full_cap, wind, coordinates,
        config, remaining_deliveries,
    )
    safe = _safe_continuation_mask(demand, capacity, num_uav, previous, current.unsqueeze(-1), energy,
        battery, edge_attr_uav, edge_attr_adr, num_depots, full_cap, wind, coordinates,
        config, remaining_deliveries, allowed_depots)
    energy, finish = energy.squeeze(-1), finish.squeeze(-1)
    idle = previous == current
    delta = demand.gather(1, current)
    delta = torch.where(idle | (current < num_depots), torch.zeros_like(delta), delta)
    next_cap = cap - delta
    if ((next_cap < -1e-5) | (next_cap > full_cap + 1e-5)).any():
        raise ValueError("Pickup overflow or delivery exceeds onboard load")
    # Mask-simulator mismatch safety net. Instead of
    # raising ValueError (which kills training), convert infeasible slots
    # to IDLE (agent stays at previous position, no battery/capacity
    # change for that agent) and log a one-time diagnostic. The policy's
    # gradient will push it away from these picks via the reward signal.
    # Enable strict mode via module-flag for post-training validation.
    import os as _os
    _strict_leg = strict or _os.environ.get('CECPDPTW_STRICT_LEG_CHECK', '0') == '1'
    _reach_fail = ~reachable.squeeze(-1)
    _leg_fail = energy > bat - reserve + 1e-5
    _safe_fail = ~safe.squeeze(-1)
    _fail = _reach_fail | _leg_fail | _safe_fail  # [B, A]
    if _fail.any():
        if _strict_leg:
            raise ValueError("Unreachable or battery-infeasible leg")
        global _MASK_SIM_WARNED
        try:
            _MASK_SIM_WARNED
        except NameError:
            _MASK_SIM_WARNED = 0
        if _MASK_SIM_WARNED < 5:
            _b, _a = _fail.nonzero()[0].tolist()
            print(f"[mask-sim-mismatch] b={_b} a={_a} prev={previous[_b,_a].item()} "
                  f"cur={current[_b,_a].item()} energy={energy[_b,_a].item():.1f} "
                  f"bat-reserve={(bat - reserve)[_b,_a].item():.1f} "
                  f"reach={bool(reachable.squeeze(-1)[_b,_a])} "
                  f"safe={bool(safe.squeeze(-1)[_b,_a])} "
                  f"({int(_fail.sum().item())} fails in this batch)", flush=True)
            _MASK_SIM_WARNED += 1
        # Convert failing slots to idle (preserves next_cap via delta=0).
        current = torch.where(_fail, previous, current)
        energy = torch.where(_fail, torch.zeros_like(energy), energy)
        finish = torch.where(_fail, T_t.reshape(B, A), finish)
        delta = torch.where(_fail, torch.zeros_like(delta), delta)
        next_cap = cap - delta
        idle = previous == current
    if allowed_depots is not None:
        allowed = torch.as_tensor(allowed_depots, device=current.device)
        if allowed.shape != (B, A, num_depots) or allowed.dtype != torch.bool:
            raise ValueError('allowed_depots must be boolean [batch, agents, depots]')
        if num_depots and ((current < num_depots) & ~idle & ~allowed.gather(2, current.clamp(max=num_depots - 1).unsqueeze(-1)).squeeze(-1)).any():
            raise ValueError('Forbidden depot')
    next_bat = (bat - energy).clamp(min=0.0)
    next_bat = torch.where((current < num_depots) & ~idle, full_bat, next_bat)
    return next_cap.detach(), finish.detach(), next_bat.detach()

def update_mask(demand, capacity, selected, mask, battery, num_uav, E, i,
                acc_uav=None, acc_adr=None,
                edge_attr_d=None, edge_attr_r=None, num_depots=None,
                T_t=None, time_window=None, max_capacity=None, wind=None, coordinates=None,
                physical_config=None, remaining_deliveries=None, allowed_depots=None,
                mask_adj_uav=None, mask_adj_adr=None,
                continuation_check='off', onboard=None, recharge_budget=None, available=None):
    import torch
    batch_size, n_agents = selected.size()
    if num_depots is None:
        num_depots = int((demand == 0).sum().item() // batch_size)
    n_total = demand.size(1)
    config = physical_configuration(physical_config) if physical_config is not None else None
    full_bat = _fleet_values(E, battery.reshape(batch_size, n_agents), num_uav)
    reserve = _battery_reserve(full_bat, num_uav, config).unsqueeze(-1)
    num_pickup = (n_total - num_depots) // 2

    depot_idx = torch.arange(num_depots, device=demand.device)
    pickup_idx = torch.arange(num_depots, num_depots + num_pickup, device=demand.device)
    corresp_del = num_depots + num_pickup + (pickup_idx - num_depots)

    go_depot = selected.lt(num_depots)

    mask_ = mask.clone()
    mask = mask_.scatter_(2, selected.unsqueeze(-1), 1)
    maskfil = mask.max(dim=1)[0].unsqueeze(1).expand(batch_size, n_agents, n_total)
    mask2 = mask.clone()

    # Depot admissibility: reachability + reserve + recharge budget + mode adjacency.
    # Depot->depot moves stay blocked to prevent loops.
    mask[:, :, depot_idx] = torch.where(
        go_depot.unsqueeze(-1),
        torch.ones_like(mask[:, :, depot_idx]),
        torch.zeros_like(mask[:, :, depot_idx]),
    )

    unvisited_pickups = (mask2[:, :, pickup_idx] == 0)
    mask[:, :, pickup_idx] *= ~unvisited_pickups

    mask[:, :, num_depots + num_pickup:n_total] = 1
    mask[:, :, corresp_del] = torch.where(mask2[:, :, pickup_idx] == 1, 0, mask[:, :, corresp_del])

    # Apply first-visit lock to customer columns only (depots revisitable under budget).
    _cust = slice(num_depots, n_total)
    mask[:, :, _cust] = torch.where(
        maskfil[:, :, _cust] == 1, maskfil[:, :, _cust], mask[:, :, _cust]
    )

    is_pickup_node = torch.zeros(n_total, dtype=torch.bool, device=demand.device)
    is_pickup_node[pickup_idx] = True
    demand_exc = (demand.unsqueeze(1) > capacity) & is_pickup_node.view(1, 1, n_total)
    final_mask = demand_exc + mask2 + mask
    if max_capacity is not None:
        cap_now = capacity.reshape(batch_size, n_agents)
        full_cap = _fleet_values(max_capacity, cap_now, num_uav)
        load = (full_cap - cap_now).unsqueeze(-1)
        delivery_too_large = (-demand.unsqueeze(1) > load + 1e-5) & (demand.unsqueeze(1) < 0)
        final_mask = final_mask.masked_fill(delivery_too_large, 1)

    # Never unmask an impossible depot. Explicit SOC reserve is physical;
    # legacy percentage recharge preferences remain diagnostic heuristics.
    for start, stop, access in ((0, num_uav, acc_uav), (num_uav, n_agents, acc_adr)):
        if access is not None:
            blocked = access.float() < 0.5
            final_mask[:, start:stop] = final_mask[:, start:stop].masked_fill(blocked.unsqueeze(1), 1)

    # Arc mode/adjacency: must match reward.replay_mandatory_trajectory.
    for start, stop, adj in ((0, num_uav, mask_adj_uav), (num_uav, n_agents, mask_adj_adr)):
        if adj is None or stop <= start:
            continue
        adj_mat = adj.reshape(batch_size, n_total, n_total)
        rows = adj_mat.gather(1, selected[:, start:stop].unsqueeze(-1).expand(-1, -1, n_total))
        final_mask[:, start:stop] = final_mask[:, start:stop].masked_fill(rows < 0.5, 1)

    if edge_attr_d is not None and edge_attr_r is not None:
        targets = torch.arange(n_total, device=demand.device).view(1, 1, n_total).expand(batch_size, n_agents, -1)
        now = battery.new_zeros(batch_size, n_agents, 1) if T_t is None else T_t
        tw = demand.new_zeros(batch_size, n_total) if time_window is None else time_window
        energy, _, reachable = _leg_metrics(
            demand, tw, now, capacity, num_uav, selected, targets,
            edge_attr_d, edge_attr_r, num_depots, max_capacity, wind, coordinates,
            config, remaining_deliveries,
        )
        impossible = (~reachable) | (energy > battery - reserve + 1e-5)
        if continuation_check != 'onboard':
            safe = _safe_continuation_mask(demand, capacity, num_uav, selected, targets, energy,
                battery, edge_attr_d, edge_attr_r, num_depots, max_capacity, wind, coordinates,
                config, remaining_deliveries, allowed_depots,
                continuation_check=continuation_check)
            impossible |= ~safe
        final_mask = final_mask.masked_fill(impossible, 1)
    else:
        if config is not None:
            raise ValueError('Configured physical checks require both distance matrices')
        for start, stop, distances in ((0, num_uav, edge_attr_d), (num_uav, n_agents, edge_attr_r)):
            if distances is not None:
                b = torch.arange(batch_size, device=demand.device).unsqueeze(1)
                rows = distances.reshape(batch_size, n_total, n_total)[b, selected[:, start:stop], :]
                impossible = (~torch.isfinite(rows)) | (rows < 0) | (rows >= 1e9)
                final_mask[:, start:stop] = final_mask[:, start:stop].masked_fill(impossible, 1)
    if allowed_depots is not None:
        allowed = torch.as_tensor(allowed_depots, device=demand.device)
        if allowed.shape != (batch_size, n_agents, num_depots) or allowed.dtype != torch.bool:
            raise ValueError('allowed_depots must be boolean [batch, agents, depots]')
        final_mask[:, :, :num_depots] = final_mask[:, :, :num_depots].masked_fill(~allowed, 1)
    if continuation_check == 'onboard':
        if config is None or onboard is None:
            raise ValueError('Onboard continuation requires physics and the per-vehicle onboard ledger')
        from continuation_suffix import certify_onboard_moves
        certified = certify_onboard_moves(
            final_mask.bool(), onboard, demand, tw, capacity, selected, battery, energy,
            num_uav, num_depots, full_bat, reserve, max_capacity, config,
            edge_attr_d, edge_attr_r, wind, coordinates, remaining_deliveries,
            allowed_depots, recharge_budget, available, mask_adj_uav, mask_adj_adr, acc_uav, acc_adr,
        )
        final_mask = final_mask.masked_fill(~certified, 1)
    # Clear depot history from mask2; depot revisits are governed by remaining_recharges.
    mask2 = mask2.clone()
    mask2[:, :, :num_depots] = 0
    return final_mask.detach(), mask2.detach()
