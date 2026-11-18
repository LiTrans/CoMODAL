from __future__ import annotations

import copy
import dataclasses
import math
import random
import time
from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np

from reward import request_penalty, operating_cost_per_minute, battery_penalty, ALPHA_U
from reward import (objective_from_instance, pickup_penalty, delivery_penalty,
                    delivery_event_time, accept_service_insertion, mandatory_service)
from energy import (coord_edge_energy_kj, arc_heading, uav_customer_service_energy_kj,
                    uav_depot_phase_energy_kj)
from hardware_profiles import physical_configuration
from model_config import UAV_BATTERY_KJ, ADR_BATTERY_KJ, UAV_SWAP_MIN, ADR_SWAP_MIN
from vrpUpdate import (
    V_UAV_MAX,
    V_ADR_MAX,
    V_UAV_DEPOT,
    V_ADR_DEPOT,
    V_UAV_MIN_PICKUP,
    V_ADR_MIN_PICKUP,
    UAV_LAND_TAKEOFF_MIN,
)

@dataclass
class Request:
    req_id: int
    t_arrival: float
    t_pickup: float
    t_delivery: float
    pickup_node: int
    delivery_node: int
    demand: float
    status: str = "unrevealed"
    assigned_vehicle: Optional[int] = None
    T_pickup_actual: Optional[float] = None
    T_delivery_actual: Optional[float] = None
    firm_owner: Optional[str] = None
    market_group: str = 'shared'
    allowed_modes: Tuple[str, ...] = ('uav', 'adr')

    VALID_TRANSITIONS = {
        "unrevealed": {"waiting_for_pickup"},
        "waiting_for_pickup": {"pickup_committed"},
        "pickup_committed": {"onboard"},
        "onboard": {"delivery_committed"},
        "delivery_committed": {"delivered"},
        "delivered": set(),
    }

    def transition(self, new_status: str) -> None:
        if new_status not in self.VALID_TRANSITIONS[self.status]:
            raise ValueError(f"Invalid transition {self.status} -> {new_status} for request {self.req_id}")
        self.status = new_status

@dataclass
class Leg:
    request_id: int
    vehicle_id: int
    leg_type: str
    from_node: int
    to_node: int
    t_depart: float
    t_arrive: float
    travel_time: float = 0.0
    operating_cost: float = 0.0
    t_arrive_raw: float = 0.0
    energy_used: float = 0.0
    travel_energy_used: float = 0.0
    service_energy_used: float = 0.0
    t_arrive_committed: Optional[float] = None
    load_before: float = 0.0
    load_after: float = 0.0
    battery_before: float = 0.0
    battery_after_travel: float = 0.0
    battery_after: float = 0.0

@dataclass
class Vehicle:
    vehicle_id: int
    mode: str
    current_node: int
    current_time: float
    battery: float
    load: float
    capacity: float
    committed_leg: Optional[Leg] = None
    onboard_request: Optional[int] = None
    battery_init: Optional[float] = None
    allowed_depots: Optional[Tuple[int, ...]] = None
    onboard_requests: Tuple[int, ...] = field(default_factory=tuple)
    recharge_count: int = 0

    def normalized_mode(self) -> str:
        mode = self.mode.lower()
        if mode not in {"uav", "adr"}:
            raise ValueError(f"Unsupported vehicle mode: {self.mode}")
        return mode

    def onboard_set(self) -> set[int]:
        onboard = set(self.onboard_requests)
        if self.onboard_request is not None:
            onboard.add(int(self.onboard_request))
        return onboard

    def set_onboard_set(self, onboard: Iterable[int]) -> None:
        values = tuple(sorted(set(int(x) for x in onboard)))
        self.onboard_requests = values
        self.onboard_request = values[0] if len(values) == 1 else None

@dataclass(frozen=True)
class SyntheticALNSConfig:
    n_uav: int
    n_adr: int
    n_depots_uav: int
    n_depots_adr: int
    depot_sharing: bool = True
    delta_minutes: float = 5.0
    shift_minutes: Optional[float] = None
    alpha_p_ratio: Optional[float] = None

def _to_numpy(value: Any) -> np.ndarray:
    if hasattr(value, "detach"):
        return value.detach().cpu().numpy()
    if hasattr(value, "cpu"):
        return value.cpu().numpy()
    return np.asarray(value)

def _get_distance_matrix(full_instance: Dict[str, Any], mode: str) -> np.ndarray:
    mode = mode.lower()
    key = "edge_attr_d" if mode == "uav" else "edge_attr_r"
    alt_key = "d_uav" if mode == "uav" else "d_adr"

    if key in full_instance:
        raw = _to_numpy(full_instance[key]).squeeze()
        n_total = int(round(math.sqrt(raw.size)))
        if n_total * n_total != raw.size:
            raise ValueError(f"{key} cannot be reshaped into a square matrix.")
        return raw.reshape(n_total, n_total).astype(float)

    if alt_key in full_instance:
        return _to_numpy(full_instance[alt_key]).astype(float)

    raise KeyError(f"No distance matrix found for mode={mode}")

def _get_mask_matrix(full_instance: Dict[str, Any], mode: str) -> Optional[np.ndarray]:
    mode = mode.lower()
    key = "mask_adjacency_uav" if mode == "uav" else "mask_adjacency_adr"

    if key not in full_instance:
        return None

    raw = _to_numpy(full_instance[key]).squeeze()
    n_total = int(round(math.sqrt(raw.size)))
    if n_total * n_total != raw.size:
        return None
    return raw.reshape(n_total, n_total).astype(float)

def _is_arc_feasible(full_instance: Dict[str, Any], mode: str, from_node: int, to_node: int) -> bool:
    dm = _get_distance_matrix(full_instance, mode)
    if from_node < 0 or to_node < 0 or from_node >= dm.shape[0] or to_node >= dm.shape[1]:
        return False

    distance = float(dm[from_node, to_node])
    if not math.isfinite(distance) or distance < 0 or distance >= 1e9:
        return False

    mask = _get_mask_matrix(full_instance, mode)
    if mask is not None and mask[from_node, to_node] <= 0.5:
        return False

    return True

def _wind_vec(full_instance: Dict[str, Any]) -> np.ndarray:
    wind = full_instance.get("wind")
    if wind is None:
        return np.zeros(2, dtype=float)

    wind = _to_numpy(wind).reshape(-1).astype(float)
    if full_instance.get('physical_config') is not None:
        if len(wind) != 2 or not np.isfinite(wind).all() or wind[0] < 0:
            raise ValueError('Invalid configured wind magnitude/direction')
    if len(wind) < 2:
        return np.zeros(2, dtype=float)

    wind_mag, wind_dir = float(wind[0]), float(wind[1])
    return np.array([wind_mag * math.cos(wind_dir), wind_mag * math.sin(wind_dir)], dtype=float)

def _cruise_speed(mode: str) -> float:
    return V_UAV_MAX if mode.lower() == "uav" else V_ADR_MAX

def _depot_speed(mode: str) -> float:
    return V_UAV_DEPOT if mode.lower() == "uav" else V_ADR_DEPOT

def _pickup_speed(distance: float, t_now: float, t_target: float, mode: str) -> float:
    if mode.lower() == "uav":
        return V_UAV_MAX
    else:
        v_min, v_max = V_ADR_MIN_PICKUP, V_ADR_MAX

    slack = max(t_target - t_now, 0.0)
    if slack <= 0.0 or distance <= 1e-9:
        return v_max
    if distance / max(v_max, 1e-9) >= slack:
        return v_max
    return max(v_min, min(v_max, distance / max(slack, 1e-9)))

def _service_time(mode: str) -> float:
    return UAV_LAND_TAKEOFF_MIN if mode.lower() == "uav" else 0.0

def _battery_threshold(vehicle: Vehicle) -> float:
    base = _full_battery(vehicle)
    frac = 0.25 if vehicle.normalized_mode() == "uav" else 0.20
    return frac * float(base)

def _recharge_minutes(vehicle: Vehicle) -> float:
    return UAV_SWAP_MIN if vehicle.normalized_mode() == "uav" else ADR_SWAP_MIN

def _full_battery(vehicle: Vehicle, full_instance=None) -> float:
    if full_instance is not None and 'physical_config' in full_instance:
        policy = physical_configuration(full_instance['physical_config']).battery_for(vehicle.normalized_mode())
        if vehicle.battery_init is not None and not math.isclose(
                float(vehicle.battery_init), policy.full_charge_kj, rel_tol=1e-6, abs_tol=1e-5):
            raise ValueError('Vehicle full pack conflicts with configured usable/SOC window')
        return policy.full_charge_kj
    nominal = UAV_BATTERY_KJ if vehicle.normalized_mode() == 'uav' else ADR_BATTERY_KJ
    return float(vehicle.battery_init if vehicle.battery_init is not None else nominal)


def _physical_config(full_instance):
    # No metadata preserves model-7 diagnostics. Explicit configuration opts in.
    return physical_configuration(full_instance['physical_config']) if 'physical_config' in full_instance else None


def remaining_delivery_ids(full_instance):
    """Persistent unresolved whole-book IDs, including unrevealed orders.

    The executor must refresh this set at completed deliveries. Without it,
    configured safety conservatively treats the entire original book as pending.
    An optimizer's short residual book is not a terminal-service certificate.
    """
    if 'n_req' in full_instance:
        n_req = int(full_instance['n_req'])
    else:
        mode = 'uav' if 'd_uav' in full_instance or 'edge_attr_d' in full_instance else 'adr'
        n_req = (_get_distance_matrix(full_instance, mode).shape[0] - int(full_instance['n_depots'])) // 2
    raw = full_instance.get('remaining_delivery_ids', range(n_req))
    values = _to_numpy(list(raw) if isinstance(raw, (set, range)) else raw).reshape(-1)
    if (not np.isfinite(values).all() or np.any(values != np.floor(values))
            or np.any(values < 0) or np.any(values >= n_req) or len(set(values)) != len(values)):
        raise ValueError('Invalid persistent remaining_delivery_ids')
    return {int(x) for x in values}


def _terminal_delivery(vehicle, req, leg_type, full_instance, load_after, onboard_after):
    return (leg_type == 'delivery' and req is not None and not onboard_after
            and abs(load_after) <= 1e-9 and remaining_delivery_ids(full_instance) == {req.req_id})


def safe_continuation_certificate(vehicle, full_instance, *, last_pickup=None):
    """Sufficient local escape/completion certificate, not global feasibility.

    A nonterminal customer can reach an allowed swap depot with its actual load
    and protected reserve. The sole remaining onboard order may instead be
    completed directly, without a final depot return. Directed arcs and depot
    permissions are checked; failure may reject a feasible multi-customer chain.
    """
    config = _physical_config(full_instance)
    if config is None or config.continuation_policy == 'legacy_leg_only':
        return dict(certified=True, basis='legacy_leg_only_diagnostic', required_kj=0.)
    policy = config.battery_for(vehicle.normalized_mode())
    minimum = policy.reserve_energy_kj
    depot = _nearest_feasible_depot(vehicle.current_node, vehicle, full_instance)
    if depot is not None:
        return dict(certified=True, basis='allowed_depot_escape', depot=depot,
                    required_kj=_depot_arrival_energy(vehicle, depot, full_instance) + minimum)
    if (last_pickup is not None and vehicle.onboard_set() == {last_pickup.req_id}
            and remaining_delivery_ids(full_instance) == {last_pickup.req_id}):
        leg, _, ok = _simulate_leg(vehicle, last_pickup, 'delivery', last_pickup.delivery_node, full_instance)
        if ok:
            return dict(certified=True, basis='last_delivery_completion',
                        required_kj=leg.energy_used + minimum)
    return dict(certified=False, basis='no_direct_escape_or_terminal_completion', required_kj=float('inf'))


def _depot_arrival_energy(vehicle, depot, full_instance):
    mode = vehicle.normalized_mode()
    wind = _wind_vec(full_instance)
    required = _edge_energy_kj(
        mode=mode, distance=float(_get_distance_matrix(full_instance, mode)[vehicle.current_node, depot]),
        payload=vehicle.load, wind_vec=wind, speed=_depot_speed(mode),
        heading_rad=arc_heading(full_instance, vehicle.current_node, depot))
    config = _physical_config(full_instance)
    if mode == 'uav' and config is not None:
        if vehicle.current_node < int(full_instance['n_depots']):
            required += uav_depot_phase_energy_kj(vehicle.load, wind, departure=True, phase_policy=config.uav_phases)
        required += uav_depot_phase_energy_kj(vehicle.load, wind, departure=False, phase_policy=config.uav_phases)
    return required

def _edge_energy_kj(
    *,
    mode: str,
    distance: float,
    payload: float,
    wind_vec: np.ndarray,
    speed: float,
    heading_rad: float = 0.0,
) -> float:
    return coord_edge_energy_kj(mode, distance, payload, wind_vec, speed, heading_rad)

def remaining_recharges(vehicle: Vehicle, full_instance: Dict[str, Any]) -> Optional[int]:
    """Remaining shift-wide activations, or None for an uncapped shift."""
    cap = full_instance.get("max_recharges_per_vehicle")
    if cap is None or int(cap) <= 0:
        return None
    return max(0, int(cap) - vehicle.recharge_count)


def _nearest_feasible_depot(from_node: int, vehicle: Vehicle, full_instance: Dict[str, Any]) -> Optional[int]:
    if remaining_recharges(vehicle, full_instance) == 0:
        return None
    dm = _get_distance_matrix(full_instance, vehicle.normalized_mode())
    n_depots = int(full_instance["n_depots"])

    candidates = vehicle.allowed_depots if vehicle.allowed_depots is not None else tuple(range(n_depots))

    best_node = None
    best_distance = float("inf")

    for depot in candidates:
        if depot < 0 or depot >= n_depots or depot == from_node:
            continue
        if not _is_arc_feasible(full_instance, vehicle.normalized_mode(), from_node, depot):
            continue

        distance = float(dm[from_node, depot])
        if from_node != vehicle.current_node:
            raise ValueError('Depot search origin must match the vehicle state')
        required = _depot_arrival_energy(vehicle, depot, full_instance)
        config = _physical_config(full_instance)
        # Reserve: mirror _simulate_leg's fallback so we don't pick a depot the
        # simulator will later reject. Without this the forced-recharge path in
        # _make_forced_delivery_or_depot_route can crash under V3 (reserve
        # threaded via reserve_frac_by_mode) at tight capacities (e.g. 30% cap).
        if config is not None:
            reserve = config.battery_for(vehicle.normalized_mode()).reserve_energy_kj
        else:
            _frac_map = full_instance.get('reserve_frac_by_mode')
            _mode = vehicle.normalized_mode()
            _frac = float(_frac_map[_mode]) if _frac_map is not None and _mode in _frac_map else 0.
            reserve = _frac * _full_battery(vehicle, full_instance)
        if required + reserve > vehicle.battery + 1e-9:
            continue
        if distance < best_distance:
            best_distance = distance
            best_node = int(depot)

    return best_node

def _simulate_leg(
    vehicle_state: Vehicle,
    req: Optional[Request],
    leg_type: str,
    to_node: int,
    full_instance: Dict[str, Any],
) -> Tuple[Optional[Leg], Vehicle, bool]:
    mode = vehicle_state.normalized_mode()
    dm = _get_distance_matrix(full_instance, mode)

    from_node = int(vehicle_state.current_node)
    if not _is_arc_feasible(full_instance, mode, from_node, to_node):
        return None, vehicle_state, False

    distance = float(dm[from_node, to_node])
    n_depots = int(full_instance["n_depots"])

    depart = float(vehicle_state.current_time)
    is_depot_leg = to_node < n_depots
    config = _physical_config(full_instance)
    full_battery = _full_battery(vehicle_state, full_instance)
    # Reserve: prefer configured hardware profile when present; else fall back
    # to full_instance['reserve_frac_by_mode'] (set by bench runner / configs)
    # so the simulator honors the same planning reserve as the MIP. Final
    # fallback 0 preserves legacy diagnostic traces.
    if config is not None:
        reserve = config.battery_for(mode).reserve_energy_kj
    else:
        frac_map = full_instance.get('reserve_frac_by_mode')
        reserve_frac = float(frac_map[mode]) if frac_map is not None and mode in frac_map else 0.
        reserve = reserve_frac * full_battery

    values = [vehicle_state.load, vehicle_state.capacity, vehicle_state.battery,
              full_battery, depart]
    if not np.isfinite(values).all() or vehicle_state.capacity <= 0:
        return None, vehicle_state, False
    if config is not None and vehicle_state.capacity > config.battery_for(mode).profile.payload_limit_kg + 1e-9:
        return None, vehicle_state, False
    if not 0 <= vehicle_state.load <= vehicle_state.capacity:
        return None, vehicle_state, False
    if not reserve - 1e-9 <= vehicle_state.battery <= full_battery + 1e-5:
        return None, vehicle_state, False
    if is_depot_leg:
        # Self-loop depot visits (from_node == to_node) are stay-at-depot
        # recharge motions used by the exact MIP's extended formulation
        # (synthetic recharge arcs between depot copies). They are legitimate
        # idle moves: vehicle sits on its pad for the recharge duration without
        # taking off. Accepting them lets the shared-physics audit replay the
        # MIP's recharge arcs and keeps RL↔classical formulation parity.
        if leg_type != "depot" or req is not None:
            return None, vehicle_state, False
        # C2: uniform recharge cap. full_instance['max_recharges_per_vehicle']
        # caps how many recharges one vehicle may activate per shift. None/0
        # disables the cap. Enforced here so ALNS/VNS/heuristics/RL see the
        # same budget as the MIP's depot_visit_limit.
        if remaining_recharges(vehicle_state, full_instance) == 0:
            return None, vehicle_state, False
        if vehicle_state.allowed_depots is not None and to_node not in vehicle_state.allowed_depots:
            return None, vehicle_state, False
    else:
        if req is None or leg_type not in {"pickup", "delivery"}:
            return None, vehicle_state, False
        from coalition_contract import request_mode_allowed
        if not request_mode_allowed(req, mode, full_instance):
            return None, vehicle_state, False
        target_node = req.pickup_node if leg_type == "pickup" else req.delivery_node
        if not np.isfinite([req.demand, req.t_arrival, req.t_pickup, req.t_delivery]).all():
            return None, vehicle_state, False
        if to_node != target_node or req.demand <= 0 or req.t_arrival > depart:
            return None, vehicle_state, False
        if req.assigned_vehicle is not None and req.assigned_vehicle != vehicle_state.vehicle_id:
            return None, vehicle_state, False

    if is_depot_leg:
        speed = _depot_speed(mode)
        travel_time = distance / max(speed, 1e-9)
        raw_arrive = depart + travel_time
        complete = raw_arrive
        complete += _recharge_minutes(vehicle_state)
    elif leg_type == "pickup":
        target = req.t_pickup if req is not None else depart
        speed = _pickup_speed(distance, depart, target, mode)
        travel_time = distance / max(speed, 1e-9)
        raw_arrive = depart + travel_time
        complete = max(raw_arrive, target) + _service_time(mode)
    else:
        speed = _cruise_speed(mode)
        travel_time = distance / max(speed, 1e-9)
        raw_arrive = depart + travel_time
        complete = raw_arrive + _service_time(mode)

    load_before = float(vehicle_state.load)
    load_after = load_before

    onboard = vehicle_state.onboard_set()

    if leg_type == "pickup" and req is not None:
        if req.req_id in onboard:
            return None, vehicle_state, False
        load_after = load_before + float(req.demand)
        if load_after > vehicle_state.capacity + 1e-9:
            return None, vehicle_state, False
        onboard.add(req.req_id)

    elif leg_type == "delivery" and req is not None:
        if req.req_id not in onboard:
            return None, vehicle_state, False
        load_after = load_before - float(req.demand)
        if load_after < -1e-9:
            return None, vehicle_state, False
        load_after = max(0.0, load_after)
        onboard.remove(req.req_id)

    terminal = (config is not None and _terminal_delivery(
        vehicle_state, req, leg_type, full_instance, load_after, onboard))
    # Arc energy uses the *onboard load during the arc*, which is
    # ``load_before`` — the vehicle's load after the previous node's service
    # but before this leg's pickup/delivery. This is the physically correct
    # payload for the from_node→to_node traversal; the target-node demand
    # q_j does not enter here. Mirrored on the exact-MIP side by
    # ce_cpdptw_ortools.energy_interval_bounds taking (lo, hi) payload
    # intervals bracketing the possible mid-route loads.
    payload = load_before
    travel_energy = _edge_energy_kj(
        mode=mode,
        distance=distance,
        payload=payload,
        wind_vec=_wind_vec(full_instance),
        speed=speed,
        heading_rad=arc_heading(full_instance, from_node, to_node),
    )
    service_energy = 0.
    if mode == 'uav':
        wind = _wind_vec(full_instance)
        if config is None:
            service_energy = uav_customer_service_energy_kj(load_before, load_after, wind) if not is_depot_leg else 0.
        else:
            phases = config.uav_phases
            # Transit includes depot departure, so battery_after_travel remains
            # the charge immediately before destination service. No new fields
            # are required by the old dispatcher/trace battery identity.
            # Self-loop depot recharge visits are pure dwell: the UAV sits on
            # its pad without taking off, so takeoff/landing phase energy and
            # time are both zero. Non-self-loop depot visits still pay full
            # takeoff + landing phase energy as before.
            is_self_loop_recharge = is_depot_leg and from_node == to_node
            departure_min = (phases.depot_takeoff_min or 0.) if (
                from_node < n_depots and not is_self_loop_recharge) else 0.
            if from_node < n_depots and not is_self_loop_recharge:
                travel_energy += uav_depot_phase_energy_kj(load_before, wind, departure=True, phase_policy=phases)
            raw_arrive += departure_min
            if is_depot_leg:
                if is_self_loop_recharge:
                    service_energy = 0.0
                    complete = raw_arrive + _recharge_minutes(vehicle_state)
                else:
                    service_energy = uav_depot_phase_energy_kj(load_before, wind, departure=False, phase_policy=phases)
                    complete = raw_arrive + (phases.depot_landing_min or 0.) + _recharge_minutes(vehicle_state)
            else:
                service_energy = uav_customer_service_energy_kj(
                    load_before, load_after, wind, phase_policy=phases, terminal_delivery=terminal)
                complete = (max(raw_arrive, req.t_pickup) if leg_type == 'pickup' else raw_arrive) + phases.customer_minutes(terminal)
    energy_used = travel_energy + service_energy
    battery_after_travel = float(vehicle_state.battery) - travel_energy
    battery_after_service = float(vehicle_state.battery) - energy_used
    if battery_after_service < reserve - 1e-9:
        return None, vehicle_state, False

    next_state = copy.deepcopy(vehicle_state)
    next_state.current_node = int(to_node)
    next_state.current_time = float(complete)
    next_state.load = float(load_after)
    next_state.set_onboard_set(onboard)

    if is_depot_leg:
        next_state.battery = full_battery
        # C2: count the recharge activation.
        next_state.recharge_count = int(vehicle_state.recharge_count) + 1
    else:
        next_state.battery = max(0.0, battery_after_service)

    if config is not None and config.continuation_policy != 'legacy_leg_only' and not is_depot_leg and not terminal:
        certificate = safe_continuation_certificate(
            next_state, full_instance, last_pickup=req if leg_type == 'pickup' else None)
        if not certificate['certified']:
            return None, vehicle_state, False

    # C1: recharge arcs incur operating cost on BOTH travel time and dwell time.
    # Self-loop recharge arcs have travel_time = 0; dwell time 10/20 min then
    # becomes the paid cost that mirrors the MIP's α · ρ · depot_visit term.
    # Default on; disable via full_instance['recharge_cost_includes_dwell']=False
    # for diagnostic reproduction of pre-C1 numbers.
    _dwell_cost_on = bool(full_instance.get('recharge_cost_includes_dwell', True))
    _op_time = float(travel_time)
    if is_depot_leg and _dwell_cost_on:
        _op_time = _op_time + float(_recharge_minutes(vehicle_state))
    leg = Leg(
        request_id=-1 if req is None else int(req.req_id),
        vehicle_id=int(vehicle_state.vehicle_id),
        leg_type="depot" if is_depot_leg else leg_type,
        from_node=from_node,
        to_node=int(to_node),
        t_depart=depart,
        t_arrive=float(complete),
        travel_time=float(travel_time),
        operating_cost=objective_from_instance(full_instance).operating(mode) * _op_time,
        t_arrive_raw=float(raw_arrive),
        energy_used=float(energy_used),
        travel_energy_used=float(travel_energy),
        service_energy_used=float(service_energy),
        load_before=float(load_before),
        load_after=float(load_after),
        battery_before=float(vehicle_state.battery),
        battery_after_travel=max(0.0, battery_after_travel),
        battery_after=float(next_state.battery),
    )

    return leg, next_state, True

def _simulate_planned_leg(vehicle, request, leg, full_instance):
    """Validate a planned departure against the vehicle's live clock."""
    if full_instance.get("allow_delayed_departure", False):
        departure = float(leg.t_depart)
        if not math.isfinite(departure):
            return None, vehicle, False
        if departure > vehicle.current_time:
            vehicle = dataclasses.replace(vehicle, current_time=departure)
    return _simulate_leg(vehicle, request, leg.leg_type, leg.to_node, full_instance)

def _schedule_route_departure(legs, requests, full_instance):
    """Minimize timing cost over a fixed nominal route's initial hold.

    Each food-ready wait absorbs an equal amount of the initial delay.
    Timing cost is piecewise linear in that delay, so only readiness and
    lateness breakpoints can improve on departing immediately.
    """
    if not legs or not full_instance.get("allow_delayed_departure", False):
        return legs, 0.0

    weights = objective_from_instance(full_instance)
    waits = []
    events = []
    candidates = {0.0}
    absorbed = 0.0
    for leg in legs:
        before = absorbed
        if leg.leg_type == "pickup":
            target = requests[leg.request_id].t_pickup
            event = leg.t_arrive_raw
            absorbed += max(0.0, target - event)
            events.append((before, event, target, weights.pickup_early, weights.pickup_late))
        elif leg.leg_type == "delivery":
            target = requests[leg.request_id].t_delivery
            event = delivery_event_time(leg, full_instance)
            events.append((before, event, target, 0.0, weights.delivery_late))
        else:
            waits.append((before, absorbed))
            continue
        candidates.update((before, before + max(0.0, target - event)))
        waits.append((before, absorbed))

    def timing_cost(delay):
        total = 0.0
        for before, event, target, early_weight, late_weight in events:
            delta = event + max(0.0, delay - before) - target
            total += early_weight * max(0.0, -delta) + late_weight * max(0.0, delta)
        return total

    original_cost = best_cost = timing_cost(0.0)
    best_delay = 0.0
    for delay in sorted(candidates):
        cost = timing_cost(delay)
        if cost < best_cost - 1e-9:
            best_cost, best_delay = cost, delay
    if best_delay == 0.0:
        return legs, 0.0

    scheduled = [
        dataclasses.replace(leg,
            t_depart=leg.t_depart + max(0.0, best_delay - before),
            t_arrive_raw=leg.t_arrive_raw + max(0.0, best_delay - before),
            t_arrive=leg.t_arrive + max(0.0, best_delay - after))
        for leg, (before, after) in zip(legs, waits)
    ]
    return scheduled, best_cost - original_cost

def _needs_depot_before_next_leg(
    vehicle_state: Vehicle,
    req: Request,
    leg_type: str,
    full_instance: Dict[str, Any],
) -> bool:
    n_depots = int(full_instance["n_depots"])
    if vehicle_state.current_node < n_depots or remaining_recharges(vehicle_state, full_instance) == 0:
        return False

    config = _physical_config(full_instance)
    if config is not None and config.continuation_policy != 'legacy_leg_only':
        target = req.pickup_node if leg_type == 'pickup' else req.delivery_node
        # A percentage heuristic is not a physical reserve or escape check.
        return not _simulate_leg(vehicle_state, req, leg_type, target, full_instance)[2]

    to_node = req.pickup_node if leg_type == "pickup" else req.delivery_node
    mode = vehicle_state.normalized_mode()
    if not _is_arc_feasible(full_instance, mode, vehicle_state.current_node, to_node):
        return False

    dm = _get_distance_matrix(full_instance, mode)
    distance = float(dm[vehicle_state.current_node, to_node])

    if leg_type == "pickup":
        speed = _pickup_speed(distance, vehicle_state.current_time, req.t_pickup, mode)
    else:
        speed = _cruise_speed(mode)

    projected_energy = _edge_energy_kj(
        mode=mode,
        distance=distance,
        payload=float(vehicle_state.load),
        wind_vec=_wind_vec(full_instance),
        speed=speed,
        heading_rad=arc_heading(full_instance, vehicle_state.current_node, to_node),
    )
    load_after = vehicle_state.load + (req.demand if leg_type == 'pickup' else -req.demand)
    if not 0 <= load_after <= vehicle_state.capacity + 1e-9:
        return False  # The leg simulator will reject the invalid load transition.
    if mode == 'uav':
        projected_energy += uav_customer_service_energy_kj(vehicle_state.load, max(0., load_after),
                                                          _wind_vec(full_instance))

    threshold = _battery_threshold(vehicle_state)
    return (
        float(vehicle_state.battery) <= threshold
        or float(vehicle_state.battery) - projected_energy < threshold
    )

def _materialize_ops(
    vehicle: Vehicle,
    ops: Sequence[Tuple[int, str]],
    requests: Dict[int, Request],
    full_instance: Dict[str, Any],
) -> Tuple[List[Leg], float, bool]:
    full_instance = dict(full_instance)
    if 'sharing_contract' in full_instance:
        full_instance['sharing_assignments'] = dict(full_instance.get('sharing_assignments', {}))
    if 'physical_config' in full_instance:
        pending = remaining_delivery_ids(full_instance)
        pending.difference_update(rid for rid, req in requests.items() if req.status == 'delivered')
        full_instance['remaining_delivery_ids'] = pending
    vehicle_state = copy.deepcopy(vehicle)
    if vehicle_state.battery_init is None:
        vehicle_state.battery_init = _full_battery(vehicle_state, full_instance)

    legs: List[Leg] = []
    total_cost = 0.0
    # Objective parity: weights are pulled from the same
    # reward.objective_from_instance source as the MIP builder
    # (ce_cpdptw_ortools.solve_ce_cpdptw_ortools lines 677-678). Under
    # mandatory service (bench default) `unserved` and `battery_shortfall`
    # are both forced to 0 by mandatory_objective_weights (reward.py:224),
    # so the `weights.battery_cost(...)` calls below evaluate to 0 and
    # total_cost reduces to the same 5-cost structure Gurobi/OR-Tools
    # minimize: operating + α_pe·early + α_pl·late_pickup + α_dl·late_delivery.
    weights = objective_from_instance(full_instance)
    seen_ops = set()

    for req_id, leg_type in ops:
        if (req_id, leg_type) in seen_ops:
            return [], float('inf'), False
        seen_ops.add((req_id, leg_type))
        if req_id not in requests:
            return [], float("inf"), False
        if leg_type not in {"pickup", "delivery"}:
            return [], float("inf"), False

        req = requests[req_id]
        to_node = int(req.pickup_node if leg_type == "pickup" else req.delivery_node)

        if _needs_depot_before_next_leg(vehicle_state, req, leg_type, full_instance):
            depot = _nearest_feasible_depot(vehicle_state.current_node, vehicle_state, full_instance)
            if depot is None:
                return [], float("inf"), False

            depot_leg, vehicle_state, ok = _simulate_leg(
                vehicle_state=vehicle_state,
                req=None,
                leg_type="depot",
                to_node=depot,
                full_instance=full_instance,
            )
            if not ok or depot_leg is None:
                return [], float("inf"), False

            legs.append(depot_leg)
            total_cost += depot_leg.operating_cost
            total_cost += weights.battery_cost(vehicle_state.battery, _full_battery(vehicle_state), vehicle_state.normalized_mode())

        leg, vehicle_state, ok = _simulate_leg(
            vehicle_state=vehicle_state,
            req=req,
            leg_type=leg_type,
            to_node=to_node,
            full_instance=full_instance,
        )
        if not ok or leg is None:
            return [], float("inf"), False

        legs.append(leg)
        total_cost += leg.operating_cost
        total_cost += weights.battery_cost(vehicle_state.battery, _full_battery(vehicle_state), vehicle_state.normalized_mode())

        if leg_type == "pickup":
            contract = full_instance.get('sharing_contract')
            if contract is not None and not contract.reserve(req_id, vehicle_state.normalized_mode(), full_instance['sharing_assignments']):
                return [], float('inf'), False
            total_cost += pickup_penalty(leg.t_arrive_raw, req.t_pickup, weights)
        else:
            total_cost += delivery_penalty(delivery_event_time(leg, full_instance), req.t_delivery, weights)
            if 'physical_config' in full_instance:
                full_instance['remaining_delivery_ids'].discard(req_id)

    legs, timing_change = _schedule_route_departure(legs, requests, full_instance)
    return legs, total_cost + timing_change, True

def _route_ops_from_vehicle(vehicle: Vehicle, active_requests: Dict[int, Request]) -> List[Tuple[int, str]]:
    ops: List[Tuple[int, str]] = []
    if vehicle.committed_leg is not None:
        return ops

    for req_id in sorted(vehicle.onboard_set()):
        req = active_requests.get(req_id)
        if req is not None and req.status in {"onboard", "delivery_committed", "pickup_committed"}:
            ops.append((req.req_id, "delivery"))

    return ops

def _insert_pickup_delivery(
    ops: Sequence[Tuple[int, str]],
    req_id: int,
    pickup_pos: int,
    delivery_pos_after_pickup: int,
) -> List[Tuple[int, str]]:
    candidate = list(ops)
    candidate[pickup_pos:pickup_pos] = [(req_id, "pickup")]
    candidate[delivery_pos_after_pickup:delivery_pos_after_pickup] = [(req_id, "delivery")]
    return candidate

def _all_pickup_delivery_insertions(
    ops: Sequence[Tuple[int, str]],
    req_id: int,
) -> Iterable[List[Tuple[int, str]]]:
    base_len = len(ops)
    for pickup_pos in range(base_len + 1):
        with_pickup = list(ops)
        with_pickup[pickup_pos:pickup_pos] = [(req_id, "pickup")]

        for delivery_pos in range(pickup_pos + 1, len(with_pickup) + 1):
            candidate = list(with_pickup)
            candidate[delivery_pos:delivery_pos] = [(req_id, "delivery")]
            yield candidate

def _best_insertion(
    vehicle: Vehicle,
    ops: Sequence[Tuple[int, str]],
    req: Request,
    active_requests: Dict[int, Request],
    full_instance: Dict[str, Any],
    fleet_ops=None,
    vehicles=None,
) -> Tuple[float, Optional[List[Tuple[int, str]]]]:
    _, base_cost, ok_base = _materialize_ops(vehicle, ops, active_requests, full_instance)
    if not ok_base:
        base_cost = 0.0

    best_delta = float("inf")
    best_ops: Optional[List[Tuple[int, str]]] = None

    for candidate_ops in _all_pickup_delivery_insertions(ops, req.req_id):
        if fleet_ops is not None:
            from coalition_contract import ops_permitted
            candidate_fleet = dict(fleet_ops)
            candidate_fleet[vehicle.vehicle_id] = candidate_ops
            if not ops_permitted(candidate_fleet, vehicles, active_requests, full_instance):
                continue
        _, candidate_cost, ok = _materialize_ops(vehicle, candidate_ops, active_requests, full_instance)
        if not ok:
            continue

        delta = candidate_cost - base_cost
        if delta < best_delta:
            best_delta = delta
            best_ops = candidate_ops

    return best_delta, best_ops

class ALNSSolution:
    def __init__(self, ops: Dict[int, List[Tuple[int, str]]], unassigned: Sequence[int]):
        self.ops = {int(vid): list(route_ops) for vid, route_ops in ops.items()}
        self.unassigned = list(dict.fromkeys(int(rid) for rid in unassigned))
        self._cost_cache: Optional[float] = None

    def copy(self) -> "ALNSSolution":
        return ALNSSolution(self.ops, self.unassigned)

    def invalidate(self) -> None:
        self._cost_cache = None

    def route_cost(
        self,
        vehicles: Dict[int, Vehicle],
        requests: Dict[int, Request],
        full_instance: Dict[str, Any],
    ) -> float:
        # Vehicle state, weights and the coalition ledger can change between
        # calls; a route-only cache is not a valid objective cache.
        from coalition_contract import ops_permitted
        if not ops_permitted(self.ops, vehicles, requests, full_instance):
            self._cost_cache = float('inf')
            return self._cost_cache

        total = 0.0
        for vid, vehicle in vehicles.items():
            _, cost, ok = _materialize_ops(vehicle, self.ops.get(vid, []), requests, full_instance)
            if not ok:
                self._cost_cache = float("inf")
                return self._cost_cache
            total += cost

        self._cost_cache = total
        return total

    def objective(
        self,
        vehicles: Dict[int, Vehicle],
        requests: Dict[int, Request],
        full_instance: Dict[str, Any],
    ) -> float:
        # Objective parity: ALNS search optimizes the same operational
        # objective as MILP at λ_U=0. No undelivered-penalty term — mandatory
        # service is enforced by the dispatcher, not by the search objective.
        return self.route_cost(vehicles, requests, full_instance)

def destroy_random(sol: ALNSSolution, n_remove: int, rng: random.Random) -> Tuple[ALNSSolution, List[int]]:
    sol = sol.copy()

    pickup_ids = sorted({
        req_id
        for ops in sol.ops.values()
        for req_id, leg_type in ops
        if leg_type == "pickup"
    })

    n_remove = min(n_remove, len(pickup_ids))
    removed_ids = rng.sample(pickup_ids, n_remove) if n_remove > 0 else []
    removed = set(removed_ids)

    for vid in sol.ops:
        sol.ops[vid] = [op for op in sol.ops[vid] if op[0] not in removed]

    sol.invalidate()
    return sol, removed_ids

def destroy_worst(
    sol: ALNSSolution,
    n_remove: int,
    vehicles: Dict[int, Vehicle],
    requests: Dict[int, Request],
    full_instance: Dict[str, Any],
) -> Tuple[ALNSSolution, List[int]]:
    sol = sol.copy()
    contributions: List[Tuple[float, int]] = []

    for vid, vehicle in vehicles.items():
        base_ops = sol.ops.get(vid, [])
        _, base_cost, ok = _materialize_ops(vehicle, base_ops, requests, full_instance)
        if not ok:
            continue

        route_req_ids = sorted({req_id for req_id, leg_type in base_ops if leg_type == "pickup"})
        for req_id in route_req_ids:
            pruned = [op for op in base_ops if op[0] != req_id]
            _, pruned_cost, ok_pruned = _materialize_ops(vehicle, pruned, requests, full_instance)
            if ok_pruned:
                contributions.append((base_cost - pruned_cost, req_id))

    contributions.sort(reverse=True)
    removed_ids: List[int] = []
    seen: set[int] = set()

    for _, req_id in contributions:
        if req_id not in seen:
            removed_ids.append(req_id)
            seen.add(req_id)
        if len(removed_ids) >= n_remove:
            break

    removed = set(removed_ids)
    for vid in sol.ops:
        sol.ops[vid] = [op for op in sol.ops[vid] if op[0] not in removed]

    sol.invalidate()
    return sol, removed_ids

def destroy_related(
    sol: ALNSSolution,
    n_remove: int,
    requests: Dict[int, Request],
    rng: random.Random,
) -> Tuple[ALNSSolution, List[int]]:
    pickup_ids = sorted({
        req_id
        for ops in sol.ops.values()
        for req_id, leg_type in ops
        if leg_type == "pickup"
    })

    if not pickup_ids:
        return sol.copy(), []

    seed_req_id = rng.choice(pickup_ids)
    seed_req = requests.get(seed_req_id)
    if seed_req is None:
        return sol.copy(), []

    def related_score(req_id: int) -> Tuple[float, float]:
        req = requests[req_id]
        temporal = abs(req.t_pickup - seed_req.t_pickup) + abs(req.t_delivery - seed_req.t_delivery)
        spatial = abs(req.pickup_node - seed_req.pickup_node) + abs(req.delivery_node - seed_req.delivery_node)
        return temporal, spatial

    removed_ids = sorted(pickup_ids, key=related_score)[:n_remove]
    removed = set(removed_ids)

    sol = sol.copy()
    for vid in sol.ops:
        sol.ops[vid] = [op for op in sol.ops[vid] if op[0] not in removed]

    sol.invalidate()
    return sol, removed_ids

def repair_greedy(
    sol: ALNSSolution,
    repair_ids: Sequence[int],
    vehicles: Dict[int, Vehicle],
    requests: Dict[int, Request],
    full_instance: Dict[str, Any],
) -> ALNSSolution:
    sol = sol.copy()
    sol.unassigned = []

    pending = [requests[req_id] for req_id in dict.fromkeys(repair_ids) if req_id in requests]
    pending.sort(key=lambda r: (r.t_delivery, r.t_pickup, r.req_id))

    for req in pending:
        best_delta = float("inf")
        best_vid: Optional[int] = None
        best_ops: Optional[List[Tuple[int, str]]] = None

        for vid, vehicle in vehicles.items():
            if vehicle.committed_leg is not None:
                continue
            delta, candidate_ops = _best_insertion(
                vehicle=vehicle,
                ops=sol.ops.get(vid, []),
                req=req,
                active_requests=requests,
                full_instance=full_instance,
                fleet_ops=sol.ops, vehicles=vehicles,
            )
            if candidate_ops is not None and delta < best_delta:
                best_delta = delta
                best_vid = vid
                best_ops = candidate_ops

        if best_vid is not None and best_ops is not None and accept_service_insertion(best_delta, full_instance):
            sol.ops[best_vid] = best_ops
        else:
            sol.unassigned.append(req.req_id)

    sol.invalidate()
    return sol

def repair_regret2(
    sol: ALNSSolution,
    repair_ids: Sequence[int],
    vehicles: Dict[int, Vehicle],
    requests: Dict[int, Request],
    full_instance: Dict[str, Any],
) -> ALNSSolution:
    sol = sol.copy()
    sol.unassigned = []

    pending = [requests[req_id] for req_id in dict.fromkeys(repair_ids) if req_id in requests]

    while pending:
        regrets = []

        for req in pending:
            choices = []
            for vid, vehicle in vehicles.items():
                if vehicle.committed_leg is not None:
                    continue
                delta, candidate_ops = _best_insertion(
                    vehicle=vehicle,
                    ops=sol.ops.get(vid, []),
                    req=req,
                    active_requests=requests,
                    full_instance=full_instance,
                    fleet_ops=sol.ops, vehicles=vehicles,
                )
                if candidate_ops is not None:
                    choices.append((delta, vid, candidate_ops))

            choices.sort(key=lambda item: item[0])

            if not choices:
                regrets.append((-float("inf"), req, None, None, float("inf")))
                continue

            c1 = choices[0][0]
            c2 = choices[1][0] if len(choices) > 1 else c1
            regret = c2 - c1
            regrets.append((regret, req, choices[0][1], choices[0][2], c1))

        regrets.sort(key=lambda item: -item[0])
        _, req, best_vid, best_ops, best_delta = regrets[0]
        pending.remove(req)

        if best_vid is not None and best_ops is not None and accept_service_insertion(best_delta, full_instance):
            sol.ops[int(best_vid)] = best_ops
        else:
            sol.unassigned.append(req.req_id)

    sol.invalidate()
    return sol

class ALNSBaseline:
    def __init__(
        self,
        *,
        seed: int = 42,
        time_budget_small_s: float = 3.0,
        time_budget_large_s: float = 8.0,
        large_threshold: int = 30,
        max_remove: int = 5,
        strong: bool = False,
    ):
        self._rng = random.Random(seed)
        self.time_budget_small_s = float(time_budget_small_s)
        self.time_budget_large_s = float(time_budget_large_s)
        self.large_threshold = int(large_threshold)
        self.max_remove = int(max_remove)
        # Default ALNS reaches greedy fixed point on the v8k50 benches
        # (ALNS cost ≡ greedy cost on every instance). The weakness has two
        # causes: tight max_remove cap (5) + fast SA
        # cooldown (temp halves every ~70 iters, reducing acceptance to greedy
        # within the 8s per-cycle budget). strong=True raises destroy neighborhood
        # size dynamically (max(10, n_active//3)) and slows cooldown (/500
        # instead of /100). Default off preserves backward-compat with existing
        # bench CSVs.
        self.strong = bool(strong)

    def initial_solution(self, residual: Dict[str, Any]) -> ALNSSolution:
        vehicles: Dict[int, Vehicle] = residual["vehicles"]
        active_requests: Dict[int, Request] = residual["active_requests"]
        full_instance = residual["full_instance"]

        ops = {
            vid: _route_ops_from_vehicle(vehicle, active_requests)
            for vid, vehicle in vehicles.items()
        }

        waiting = [
            req for req in active_requests.values()
            if req.status == "waiting_for_pickup"
        ]
        waiting.sort(key=lambda r: (r.t_delivery, r.t_pickup, r.req_id))

        sol = ALNSSolution(ops, [])

        for req in waiting:
            best_delta = float("inf")
            best_vid = None
            best_ops = None

            for vid, vehicle in vehicles.items():
                if vehicle.committed_leg is not None:
                    continue
                delta, candidate_ops = _best_insertion(
                    vehicle=vehicle,
                    ops=sol.ops.get(vid, []),
                    req=req,
                    active_requests=active_requests,
                    full_instance=full_instance,
                    fleet_ops=sol.ops, vehicles=vehicles,
                )
                if candidate_ops is not None and delta < best_delta:
                    best_delta = delta
                    best_vid = vid
                    best_ops = candidate_ops

            if best_vid is not None and best_ops is not None and accept_service_insertion(best_delta, full_instance):
                sol.ops[best_vid] = best_ops
            else:
                sol.unassigned.append(req.req_id)

        sol.invalidate()
        return sol

    def solve(self, residual: Dict[str, Any]) -> Dict[int, List[Leg]]:
        vehicles: Dict[int, Vehicle] = residual["vehicles"]
        active_requests: Dict[int, Request] = residual["active_requests"]
        full_instance = residual["full_instance"]

        n_active = len(active_requests)
        time_budget = self.time_budget_small_s if n_active <= self.large_threshold else self.time_budget_large_s

        t_start = time.perf_counter()
        current = self.initial_solution(residual)
        best = current.copy()
        best_obj = best.objective(vehicles, active_requests, full_instance)
        required_service = mandatory_service(full_instance)
        best_key = (len(best.unassigned), best_obj)

        destroy_weights = [1.0, 1.0, 1.0]
        repair_weights = [1.0, 1.0]

        n_iter = 0

        while time.perf_counter() - t_start < time_budget:
            assigned_pickups = {
                req_id
                for ops in current.ops.values()
                for req_id, leg_type in ops
                if leg_type == "pickup"
            }

            if not assigned_pickups and not current.unassigned:
                break

            # strong=True raises destroy neighborhood size from 5 to max(10, n/3)
            _max_rm = max(10, n_active // 3) if self.strong else self.max_remove
            n_remove = max(1, min(max(n_active // 4, 1), _max_rm, len(assigned_pickups) or 1))

            destroy_choice = self._rng.choices(range(3), weights=destroy_weights)[0]
            if destroy_choice == 0:
                candidate, removed_ids = destroy_random(current, n_remove, self._rng)
            elif destroy_choice == 1:
                candidate, removed_ids = destroy_worst(current, n_remove, vehicles, active_requests, full_instance)
            else:
                candidate, removed_ids = destroy_related(current, n_remove, active_requests, self._rng)

            repair_ids = list(dict.fromkeys(list(removed_ids) + list(candidate.unassigned)))
            candidate.unassigned = []

            repair_choice = self._rng.choices(range(2), weights=repair_weights)[0]
            if repair_choice == 0:
                candidate = repair_greedy(candidate, repair_ids, vehicles, active_requests, full_instance)
            else:
                candidate = repair_regret2(candidate, repair_ids, vehicles, active_requests, full_instance)

            cand_obj = candidate.objective(vehicles, active_requests, full_instance)
            curr_obj = current.objective(vehicles, active_requests, full_instance)

            delta = cand_obj - curr_obj
            # strong=True slows SA cooldown so acceptance stays non-greedy for
            # more iterations, letting ALNS escape the greedy-insertion basin.
            _cool = 500.0 if self.strong else 100.0
            temp = max(1.0, curr_obj * 0.01 * math.exp(-n_iter / _cool))
            accept = math.isfinite(cand_obj) and (not math.isfinite(curr_obj) or delta <= 0 or self._rng.random() < math.exp(-delta / temp))
            if required_service and len(candidate.unassigned) != len(current.unassigned):
                accept = math.isfinite(cand_obj) and len(candidate.unassigned) < len(current.unassigned)

            if accept:
                current = candidate

                cand_key = (len(candidate.unassigned), cand_obj)
                if (cand_key < best_key if required_service else cand_obj < best_obj):
                    best = candidate.copy()
                    best_obj = cand_obj
                    best_key = cand_key
                    destroy_weights[destroy_choice] = min(destroy_weights[destroy_choice] * 1.1, 5.0)
                    repair_weights[repair_choice] = min(repair_weights[repair_choice] * 1.1, 5.0)

            n_iter += 1

        routes: Dict[int, List[Leg]] = {}
        for vid, vehicle in vehicles.items():
            if vehicle.committed_leg is not None:
                routes[vid] = [vehicle.committed_leg]
                continue

            legs, _, ok = _materialize_ops(vehicle, best.ops.get(vid, []), active_requests, full_instance)
            routes[vid] = legs if ok else []

        return routes

def build_residual_instance(
    requests: Dict[int, Request],
    vehicles: Dict[int, Vehicle],
    current_time: float,
    full_instance: Dict[str, Any],
) -> Dict[str, Any]:
    active_requests = {
        req_id: req
        for req_id, req in requests.items()
        if req.status not in {"unrevealed", "delivered"}
    }

    return {
        "full_instance": full_instance,
        "active_requests": active_requests,
        "vehicles": vehicles,
        "current_time": current_time,
    }

def pick_first_uncommitted_leg(route: Sequence[Leg], requests: Dict[int, Request], n_depots: int) -> Optional[Leg]:
    for leg in route:
        if leg.to_node < n_depots:
            return leg

        req = requests.get(leg.request_id)
        if req is None:
            continue

        if leg.leg_type == "pickup" and req.status == "waiting_for_pickup":
            return leg
        if leg.leg_type == "delivery" and req.status in {"onboard", "delivery_committed"}:
            return leg

    return None

def advance_time(
    t_start: float,
    t_end: float,
    vehicles: Dict[int, Vehicle],
    full_instance: Dict[str, Any],
    requests: Optional[Dict[int, Request]] = None,
) -> Tuple[List[Leg], Dict[int, Vehicle], float, float, float, float]:
    from dispatch_sim import advance_time as execute_legs
    return execute_legs(t_start, t_end, vehicles, full_instance, requests)

def _count_undelivered(requests: Dict[int, Request]) -> int:
    return sum(1 for req in requests.values() if req.status not in {"delivered", "unrevealed"})

def _count_revealed(requests: Dict[int, Request]) -> int:
    return sum(1 for req in requests.values() if req.status != "unrevealed")

class RollingHorizonDispatcher:
    def __init__(
        self,
        solver: ALNSBaseline,
        delta_minutes: float,
        shift_minutes: float = 120.0,
        alpha_p_ratio: Optional[float] = None,
        service_policy: str = 'mandatory',
    ):
        self.solver = solver
        self.delta = float(delta_minutes)
        self.T = float(shift_minutes)
        self.alpha_p_ratio = alpha_p_ratio
        self.service_policy = service_policy

    def run_shift(
        self,
        arrival_stream: Sequence[Dict[str, Any]],
        fleet_init: Dict[int, Vehicle],
        full_instance: Dict[str, Any],
    ) -> List[Dict[str, Any]]:
        from dispatch_sim import RollingHorizonDispatcher as SharedDispatcher
        return SharedDispatcher(
            self.solver, self.delta, self.T, alpha_p_ratio=self.alpha_p_ratio,
            service_policy=self.service_policy,
        ).run_shift(arrival_stream, fleet_init, full_instance)

def build_static_arrival_stream_from_instance(full_instance: Dict[str, Any]) -> List[Dict[str, Any]]:
    n_depots = int(full_instance["n_depots"])
    n_req = int(full_instance["n_req"])

    demand = _to_numpy(full_instance["demand"]).reshape(-1).astype(float)
    time_window = _to_numpy(full_instance["time_window"]).reshape(-1).astype(float)

    stream = []
    for req_id in range(n_req):
        pickup_node = n_depots + req_id
        delivery_node = n_depots + n_req + req_id

        stream.append({
            "req_id": req_id,
            "t_arrival": 0.0,
            "t_pickup": float(time_window[pickup_node]),
            "t_delivery": float(time_window[delivery_node]),
            "pickup_node": pickup_node,
            "delivery_node": delivery_node,
            "demand": abs(float(demand[pickup_node])),
        })

    return stream


def build_dynamic_arrival_stream_from_instance(full_instance: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Like the static version but uses pre-generated t_arrival per request (staggered arrivals).

    Falls back to t_arrival=0.0 if the instance has no 't_arrival' key (backward compat).
    """
    n_depots = int(full_instance["n_depots"])
    n_req = int(full_instance["n_req"])

    demand = _to_numpy(full_instance["demand"]).reshape(-1).astype(float)
    time_window = _to_numpy(full_instance["time_window"]).reshape(-1).astype(float)

    if "t_arrival" in full_instance:
        t_arrival_arr = _to_numpy(full_instance["t_arrival"]).reshape(-1).astype(float)
    else:
        t_arrival_arr = None

    stream = []
    for req_id in range(n_req):
        pickup_node = n_depots + req_id
        delivery_node = n_depots + n_req + req_id
        t_arr = float(t_arrival_arr[req_id]) if t_arrival_arr is not None else 0.0
        stream.append({
            "req_id": req_id,
            "t_arrival": t_arr,
            "t_pickup": float(time_window[pickup_node]),
            "t_delivery": float(time_window[delivery_node]),
            "pickup_node": pickup_node,
            "delivery_node": delivery_node,
            "demand": abs(float(demand[pickup_node])),
        })

    return sorted(stream, key=lambda x: x["t_arrival"])


def build_initial_fleet_from_instance(
    full_instance: Dict[str, Any],
    config: SyntheticALNSConfig,
) -> Dict[int, Vehicle]:
    n_depots = int(full_instance["n_depots"])
    expected_depots = config.n_depots_uav + config.n_depots_adr
    if expected_depots != n_depots:
        raise ValueError(f"Depot split mismatch: config={expected_depots}, instance={n_depots}")

    capacity = _to_numpy(full_instance["capacity"]).reshape(-1).astype(float)
    battery = _to_numpy(full_instance["battery"]).reshape(-1).astype(float)
    start_battery = _to_numpy(full_instance.get('start_battery', battery)).reshape(-1).astype(float)

    n_agents = config.n_uav + config.n_adr
    if len(capacity) != n_agents or len(battery) != n_agents or len(start_battery) != n_agents:
        raise ValueError("Vehicle count mismatch with capacity/battery arrays.")
    physical = _physical_config(full_instance)
    if physical is not None:
        policies = [physical.uav_battery] * config.n_uav + [physical.adr_battery] * config.n_adr
        expected_full = np.array([p.full_charge_kj for p in policies])
        if not np.allclose(battery, expected_full, rtol=1e-6, atol=1e-5):
            raise ValueError('Instance battery arrays conflict with physical_config')
        if 'start_battery' not in full_instance:
            start_battery = np.array([p.initial_charge_kj for p in policies])
        reserve = np.array([p.reserve_energy_kj for p in policies])
        if np.any(start_battery < reserve - 1e-5) or np.any(start_battery > expected_full + 1e-5):
            raise ValueError('Initial SOC outside configured window')

    all_depots = tuple(range(n_depots))
    uav_depots = tuple(range(config.n_depots_uav))
    adr_depots = tuple(range(config.n_depots_uav, n_depots))

    fleet: Dict[int, Vehicle] = {}

    for i in range(config.n_uav):
        allowed = all_depots if config.depot_sharing else uav_depots
        start_node = allowed[i % len(allowed)]
        fleet[i] = Vehicle(
            vehicle_id=i,
            mode="uav",
            current_node=int(start_node),
            current_time=0.0,
            battery=float(start_battery[i]),
            load=0.0,
            capacity=float(capacity[i]),
            battery_init=float(battery[i]),
            allowed_depots=allowed,
        )

    for j in range(config.n_adr):
        idx = config.n_uav + j
        allowed = all_depots if config.depot_sharing else adr_depots
        start_node = allowed[j % len(allowed)]
        fleet[idx] = Vehicle(
            vehicle_id=idx,
            mode="adr",
            current_node=int(start_node),
            current_time=0.0,
            battery=float(start_battery[idx]),
            load=0.0,
            capacity=float(capacity[idx]),
            battery_init=float(battery[idx]),
            allowed_depots=allowed,
        )

    return fleet

def infer_shift_minutes(full_instance: Dict[str, Any], extra_minutes: float = 120.0) -> float:
    time_window = _to_numpy(full_instance["time_window"]).reshape(-1).astype(float)
    return float(np.nanmax(time_window) + extra_minutes)

def solve_static_instance_with_alns(
    full_instance: Dict[str, Any],
    config: SyntheticALNSConfig,
    *,
    solver: Optional[ALNSBaseline] = None,
) -> List[Dict[str, Any]]:
    solver = solver if solver is not None else ALNSBaseline(seed=42)
    arrival_stream = build_dynamic_arrival_stream_from_instance(full_instance)
    fleet = build_initial_fleet_from_instance(full_instance, config)

    shift_minutes = config.shift_minutes
    if shift_minutes is None:
        shift_minutes = infer_shift_minutes(full_instance)

    # Honour the instance's service_policy so the
    # lambda_u_sweep (which sets service_policy='legacy_optional' with
    # objective_weights.unserved=λ) actually solves the legacy-optional
    # problem instead of forcing mandatory and tripping the five-cost
    # mandatory validator in reward.mandatory_objective_weights.
    dispatcher = RollingHorizonDispatcher(
        solver=solver,
        delta_minutes=config.delta_minutes,
        shift_minutes=shift_minutes,
        alpha_p_ratio=config.alpha_p_ratio,
        service_policy=str(full_instance.get('service_policy', 'mandatory')),
    )

    return dispatcher.run_shift(
        arrival_stream=arrival_stream,
        fleet_init=fleet,
        full_instance=full_instance,
    )

def summarize_episode_log(episode_log: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    if not episode_log:
        return {}

    summary = dict(episode_log[-1]) if episode_log[-1].get("summary") else {}
    if not summary:
        return {}

    return {
        "total_revealed": summary.get("total_revealed"),
        "total_delivered": summary.get("total_delivered"),
        "undelivered": summary.get("undelivered"),
        "operating_cost": summary.get("operating_cost"),
        "penalty_cost": summary.get("penalty_cost"),
        "battery_penalty_cost": summary.get("battery_penalty_cost"),
        "total_cost": summary.get("total_cost"),
        "request_history": summary.get("request_history"),
    }

def build_toy_full_instance() -> Dict[str, Any]:
    coordinates = np.array([
        [0.0, 0.0],
        [10.0, 0.0],
        [1.0, 0.0],
        [1.2, 0.1],
        [2.0, 0.0],
        [2.2, 0.1],
    ], dtype=np.float32)

    n_depots = 2
    n_req = 2
    n_total = n_depots + 2 * n_req

    diff = coordinates[:, None, :] - coordinates[None, :, :]
    d_uav = np.sqrt((diff ** 2).sum(-1)).astype(np.float32)
    d_adr = d_uav.copy()

    mask = np.ones((n_total, n_total), dtype=np.float32)
    np.fill_diagonal(mask, 0.0)

    demand = np.zeros(n_total, dtype=np.float32)
    demand[n_depots + 0] = 1.0
    demand[n_depots + 1] = 1.2
    demand[n_depots + n_req + 0] = -1.0
    demand[n_depots + n_req + 1] = -1.2

    time_window = np.zeros(n_total, dtype=np.float32)
    time_window[n_depots + 0] = 4.0
    time_window[n_depots + 1] = 4.5
    time_window[n_depots + n_req + 0] = 18.0
    time_window[n_depots + n_req + 1] = 18.5

    return {
        "edge_attr_d": d_uav.reshape(-1, 1),
        "edge_attr_r": d_adr.reshape(-1, 1),
        "mask_adjacency_uav": mask.reshape(-1, 1),
        "mask_adjacency_adr": mask.reshape(-1, 1),
        "demand": demand.reshape(-1, 1),
        "time_window": time_window.reshape(-1, 1),
        "capacity": np.array([5.0, 10.0], dtype=np.float32),
        "battery": np.array([6500.0, 4500.0], dtype=np.float32),
        "wind": np.array([0.0, 0.0], dtype=np.float32),
        "n_depots": n_depots,
        "n_req": n_req,
    }

def demo_toy_alns() -> None:
    full_instance = build_toy_full_instance()
    config = SyntheticALNSConfig(
        n_uav=1,
        n_adr=1,
        n_depots_uav=1,
        n_depots_adr=1,
        depot_sharing=False,
        delta_minutes=2.0,
        shift_minutes=80.0,
    )

    solver = ALNSBaseline(
        seed=42,
        time_budget_small_s=1.0,
        time_budget_large_s=2.0,
    )

    log = solve_static_instance_with_alns(full_instance, config, solver=solver)
    print(summarize_episode_log(log))

if __name__ == "__main__":
    demo_toy_alns()
