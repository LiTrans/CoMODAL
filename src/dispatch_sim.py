import dataclasses
import math
from dataclasses import dataclass, field
from typing import Optional, Dict, List, Any, Tuple, Set
import time
import numpy as np

from reward import (
    request_penalty, operating_cost_per_minute, battery_penalty, ALPHA_U,
    ALPHA_E, ALPHA_P, ALPHA_D,
)
from reward import (objective_weights, objective_from_instance, pickup_penalty,
                    delivery_penalty, OBJECTIVE_VERSION, MANDATORY_OBJECTIVE_VERSION,
                    mandatory_objective_weights, delivery_event_time)


class MandatoryServiceError(RuntimeError):
    """An unsuccessful evaluator run, not proof that the instance is infeasible."""

    def __init__(self, reason, current_time, pending_ids, episode_log):
        self.reason = reason
        self.current_time = float(current_time)
        self.pending_ids = tuple(sorted(pending_ids))
        self.episode_log = episode_log
        super().__init__(f'Mandatory-service run failed ({reason}) at {current_time:g} min; '
                         f'pending requests: {self.pending_ids}. This is not an infeasibility certificate.')

@dataclass
class Request:
    req_id: int
    t_arrival: float
    t_pickup: float
    t_delivery: float
    pickup_node: int
    delivery_node: int
    demand: float
    status: str = 'unrevealed'
    assigned_vehicle: Optional[int] = None
    T_pickup_actual: Optional[float] = None
    T_delivery_actual: Optional[float] = None
    firm_owner: Optional[str] = None
    market_group: str = 'shared'
    allowed_modes: Tuple[str, ...] = ('uav', 'adr')

    VALID_TRANSITIONS = {
        'unrevealed': {'waiting_for_pickup'},
        'waiting_for_pickup': {'pickup_committed'},
        'pickup_committed': {'onboard'},
        'onboard': {'delivery_committed'},
        'delivery_committed': {'delivered'},
        'delivered': set(),
    }

    def transition(self, new_status: str) -> None:
        assert new_status in self.VALID_TRANSITIONS[self.status], (
            f'Invalid transition {self.status} -> {new_status} for req {self.req_id}'
        )
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
    onboard_requests: Set[int] = field(default_factory=set)
    battery_init: Optional[float] = None
    allowed_depots: Optional[tuple] = None
    recharge_count: int = 0

    def normalized_mode(self) -> str:
        mode = self.mode.lower()
        if mode not in {'uav', 'adr'}:
            raise ValueError(f'Unsupported vehicle mode: {self.mode}')
        return mode

    def onboard_set(self) -> Set[int]:
        return set(self.onboard_requests)

    def set_onboard_set(self, onboard) -> None:
        self.onboard_requests = set(int(x) for x in onboard)

def _get_distance_matrix(full_instance: Dict[str, Any], mode: str) -> np.ndarray:
    if mode == 'uav':
        if 'edge_attr_d' in full_instance:
            n_total = int(round(np.sqrt(full_instance['edge_attr_d'].numel())))
            return full_instance['edge_attr_d'].view(n_total, n_total).cpu().numpy()
        if 'd_uav' in full_instance:
            return full_instance['d_uav']
    else:
        if 'edge_attr_r' in full_instance:
            n_total = int(round(np.sqrt(full_instance['edge_attr_r'].numel())))
            return full_instance['edge_attr_r'].view(n_total, n_total).cpu().numpy()
        if 'd_adr' in full_instance:
            return full_instance['d_adr']
    raise KeyError(f'No distance matrix found for mode={mode}')

def _vehicle_speed(mode: str) -> float:
    from vrpUpdate import V_UAV_MAX, V_ADR_MAX
    return V_UAV_MAX if mode == 'uav' else V_ADR_MAX

from vrpUpdate import V_UAV_DEPOT, V_ADR_DEPOT

def _travel_time(from_node: int, to_node: int, mode: str, full_instance: Dict[str, Any]) -> float:
    dm = _get_distance_matrix(full_instance, mode)
    d = float(dm[from_node, to_node])
    if d >= 1e9:
        return float('inf')
    v = _vehicle_speed(mode)
    return d / max(v, 1e-9)

def _leg_operating_cost(vehicle: Vehicle, travel_time: float) -> float:
    return operating_cost_per_minute(vehicle.mode == 'uav') * travel_time

def _pickup_speed(distance: float, t_now: float, t_target: float, mode: str) -> float:
    from vrpUpdate import V_UAV_MAX, V_ADR_MAX, V_UAV_MIN_PICKUP, V_ADR_MIN_PICKUP
    if mode == 'uav':
        return V_UAV_MAX
    else:
        v_min, v_max = V_ADR_MIN_PICKUP, V_ADR_MAX

    slack = max(t_target - t_now, 0.0)
    if slack <= 0.0 or distance <= 1e-9:
        return v_max
    if distance / max(v_max, 1e-9) >= slack:
        return v_max
    return max(v_min, min(v_max, distance / max(slack, 1e-9)))

def _customer_service_time(vehicle: Vehicle) -> float:
    from vrpUpdate import UAV_LAND_TAKEOFF_MIN
    return UAV_LAND_TAKEOFF_MIN if vehicle.mode == 'uav' else 0.0

from model_config import (UAV_BATTERY_KJ as _NOMINAL_BATTERY_UAV,
                          ADR_BATTERY_KJ as _NOMINAL_BATTERY_ADR, ENERGY_MODEL_VERSION)


def _battery_threshold(vehicle: Vehicle) -> float:
    from ce_cpdptw_alns import _full_battery
    base = _full_battery(vehicle)
    return (0.25 if vehicle.normalized_mode() == "uav" else 0.20) * float(base)

def _recharge_minutes(vehicle: Vehicle) -> float:
    from model_config import UAV_SWAP_MIN, ADR_SWAP_MIN
    return UAV_SWAP_MIN if vehicle.mode == 'uav' else ADR_SWAP_MIN

def _count_undelivered(requests: Dict[int, Request]) -> int:
    return sum(1 for r in requests.values() if r.status not in ('delivered', 'unrevealed'))

def _count_revealed(requests: Dict[int, Request]) -> int:
    return sum(1 for r in requests.values() if r.status != 'unrevealed')

def _wind_vec(full_instance: Dict[str, Any]) -> np.ndarray:
    wind = full_instance.get('wind')
    if wind is None:
        return np.zeros(2)
    if hasattr(wind, 'cpu'):
        wind = wind.cpu().numpy()
    wind = np.asarray(wind, dtype=np.float64)
    wind_mag, wind_dir = float(wind[0]), float(wind[1])
    return np.array([wind_mag * math.cos(wind_dir), wind_mag * math.sin(wind_dir)])

def advance_time(
    t_start: float,
    t_end: float,
    vehicles: Dict[int, Vehicle],
    full_instance: Dict[str, Any],
    requests: Optional[Dict[int, 'Request']] = None,
) -> Tuple[List[Leg], Dict[int, Vehicle], float, float, float, float]:
    from ce_cpdptw_alns import _simulate_leg, _full_battery
    import copy

    if t_end < t_start:
        raise ValueError("Time cannot run backwards")
    completed = []
    operating = penalty = operating_uav = operating_adr = 0.0
    for veh in vehicles.values():
        leg = veh.committed_leg
        if leg is None:
            veh.current_time = max(veh.current_time, t_end)
            continue
        if veh.battery_init is None:
            veh.battery_init = _full_battery(veh, full_instance)
        if not getattr(leg, '_execution_validated', False):
            req = requests.get(leg.request_id) if requests is not None else None
            if leg.leg_type != "depot" and req is None:
                raise ValueError("Customer leg requires request metadata")
            if int(leg.from_node) != int(veh.current_node):
                raise ValueError("Committed leg origin does not match vehicle state")
            shadow = copy.deepcopy(veh)
            shadow.committed_leg = None
            shadow.current_time = max(float(veh.current_time), float(leg.t_depart))
            checked, _, ok = _simulate_leg(shadow, req, leg.leg_type, leg.to_node, full_instance)
            if not ok:
                raise ValueError(f"Infeasible committed leg for vehicle {veh.vehicle_id}")
            for field_name in (
                "t_depart", "t_arrive", "t_arrive_raw", "travel_time", "operating_cost",
                "energy_used", "travel_energy_used", "service_energy_used",
                "load_before", "load_after", "battery_before",
                "battery_after_travel", "battery_after",
            ):
                setattr(leg, field_name, getattr(checked, field_name))
            leg.t_arrive_committed = checked.t_arrive
            leg._execution_validated = True

        if leg.t_arrive_committed > t_end:
            veh.current_time = max(veh.current_time, t_end)
            continue
        # A locked leg cannot be replanned or recharged during intervening ticks.
        veh.current_node = int(leg.to_node)
        veh.current_time = float(leg.t_arrive_committed)
        veh.load = float(leg.load_after)
        veh.battery = float(leg.battery_after)
        onboard = veh.onboard_set()
        if leg.leg_type == "pickup":
            onboard.add(leg.request_id)
        elif leg.leg_type == "delivery":
            onboard.remove(leg.request_id)
        elif leg.leg_type == "depot":
            veh.recharge_count += 1
        veh.set_onboard_set(onboard)
        veh.committed_leg = None
        operating += leg.operating_cost
        if veh.normalized_mode() == "uav":
            operating_uav += leg.operating_cost
        else:
            operating_adr += leg.operating_cost
        penalty += objective_from_instance(full_instance).battery_cost(veh.battery, _full_battery(veh), veh.normalized_mode())
        completed.append(leg)

    return completed, vehicles, operating, penalty, operating_uav, operating_adr

def sample_arrival_stream(
    shift_minutes: float,
    peak_rate_per_hour: float,
    profile: str = 'uniform',
    seed: Optional[int] = None,
    max_requests: Optional[int] = None,
) -> List[Dict[str, Any]]:
    rng = np.random.default_rng(seed)
    rate_per_min = peak_rate_per_hour / 60.0
    arrivals: List[Dict[str, Any]] = []
    req_id = 0
    t = 0.0

    while t < shift_minutes:
        inter = rng.exponential(1.0 / max(rate_per_min, 1e-9))
        t += inter
        if t >= shift_minutes:
            break
        if max_requests is not None and req_id >= max_requests:
            break

        t_pickup = t + rng.uniform(10.0, 20.0)
        delivery_slack = float(np.clip(rng.normal(30.0, 5.0), 15.0, 60.0))
        arrivals.append({
            'req_id': req_id,
            't_arrival': float(t),
            't_pickup': float(t_pickup),
            't_delivery': float(t_pickup + delivery_slack),
            'demand': float(rng.uniform(0.5, 3.0)),
        })
        req_id += 1

    return arrivals

def build_residual_instance(
    requests: Dict[int, Request],
    vehicles: Dict[int, Vehicle],
    current_time: float,
    full_instance: Dict[str, Any],
) -> Dict[str, Any]:
    active_requests = {
        rid: req for rid, req in requests.items()
        if req.status not in ('unrevealed', 'delivered')
    }
    return {
        'full_instance': full_instance,
        'active_requests': active_requests,
        'vehicles': vehicles,
        'current_time': current_time,
    }

def pick_first_uncommitted_leg(route: List[Leg], requests: Dict[int, Request],
                               n_depots: Optional[int] = None) -> Optional[Leg]:
    min_customer_node = n_depots if n_depots is not None else min(
        [min(r.pickup_node, r.delivery_node) for r in requests.values()],
        default=10**9,
    )
    for leg in route:
        if leg.to_node < min_customer_node:
            return leg
        req = requests.get(leg.request_id)
        if req is None:
            continue
        if leg.leg_type == 'pickup' and req.status == 'waiting_for_pickup':
            return leg
        if leg.leg_type == 'delivery' and req.status in ('onboard', 'delivery_committed'):
            return leg
    return None


def _commit_route_prefixes(route_plan, requests, vehicles, full_instance, contract, rejected):
    from ce_cpdptw_alns import _simulate_planned_leg

    for vid, route in route_plan.items():
        vehicle = vehicles[vid]
        if vehicle.committed_leg is not None:
            continue
        first = pick_first_uncommitted_leg(route, requests, int(full_instance['n_depots']))
        if first is None:
            continue
        request = requests.get(first.request_id)
        checked, _, feasible = _simulate_planned_leg(vehicle, request, first, full_instance)
        if not feasible or checked is None or first.from_node != vehicle.current_node:
            rejected.append({'vehicle_id': vid, 'to_node': first.to_node})
            continue
        vehicle.committed_leg = checked
        if request is None:
            continue
        if contract is not None and first.leg_type == 'pickup':
            if not contract.reserve(request.req_id, vehicle.normalized_mode(), full_instance['sharing_assignments']):
                raise RuntimeError('Accepted plan violates sharing budget at commitment')
        request.assigned_vehicle = vid
        if first.leg_type == 'pickup' and request.status == 'waiting_for_pickup':
            request.transition('pickup_committed')
        elif first.leg_type == 'delivery' and request.status == 'onboard':
            request.transition('delivery_committed')


def _service_advancing_prefixes(plan, vehicles):
    """Keep replay-validated prefixes that clear the carrier's obligations."""
    result = {}
    for vid, route in plan.items():
        delivered = {leg.request_id for leg in route if leg.leg_type == 'delivery'}
        if not delivered:
            continue
        last_delivery = max(i for i, leg in enumerate(route) if leg.leg_type == 'delivery')
        prefix = route[:last_delivery + 1]
        picked = {leg.request_id for leg in prefix if leg.leg_type == 'pickup'}
        if (vehicles[vid].onboard_set() | picked) <= delivered:
            result[vid] = prefix
    return result


class RollingHorizonDispatcher:
    def __init__(self, solver, delta_minutes: float, shift_minutes: float = 120,
                 alpha_p_ratio: float = None, replan_gate=None,
                 compute_cost_per_replan: float = 0.0,
                 compute_cost_time_weight: float = 0.0,
                 compute_cost_node_weight: float = 0.0,
                 cost_weights=None, service_policy='mandatory', max_events=100000,
                 stalled_solver=None):
        if service_policy not in ('mandatory', 'legacy_optional'):
            raise ValueError('Unknown service policy')
        if not math.isfinite(delta_minutes) or delta_minutes <= 0:
            raise ValueError('Dispatch interval must be finite and positive')
        if not math.isfinite(shift_minutes) or shift_minutes <= 0:
            raise ValueError('Arrival-generation window must be finite and positive')
        if not isinstance(max_events, int) or isinstance(max_events, bool) or max_events <= 0:
            raise ValueError('Event guard must be a positive integer')
        self.solver = solver
        self.stalled_solver = stalled_solver
        self.delta = delta_minutes
        self.T = shift_minutes
        self.alpha_p_ratio = alpha_p_ratio
        self.cost_weights = cost_weights
        self.service_policy = service_policy
        self.max_events = max_events
        self.replan_gate = replan_gate
        self.compute_cost_per_replan = float(compute_cost_per_replan)
        self.compute_cost_time_weight = float(compute_cost_time_weight)
        self.compute_cost_node_weight = float(compute_cost_node_weight)

    def _gate_decision(self, gate, features: Dict[str, Any], residual: Dict[str, Any]):
        if hasattr(gate, 'decide'):
            decision = gate.decide(features, residual)
        elif callable(gate):
            decision = gate(features, residual)
        else:
            raise TypeError('replan_gate must expose decide(features, residual) or be callable')
        if hasattr(decision, 'replan'):
            return bool(decision.replan), decision
        if isinstance(decision, dict):
            return bool(decision.get('replan', False)), decision
        return bool(decision), decision

    def _compute_replan_cost(self, solver_time_s: float, residual: Dict[str, Any]) -> float:
        full_instance = residual.get('full_instance', {}) or {}
        n_depots = int(full_instance.get('n_depots', 0) or 0)
        n_active_requests = len(residual.get('active_requests', {}) or {})
        n_nodes = n_depots + 2 * n_active_requests
        return (
            self.compute_cost_per_replan
            + self.compute_cost_time_weight * float(solver_time_s)
            + self.compute_cost_node_weight * float(n_nodes)
        )

    def _build_trigger_log(
        self,
        replan_times: List[float],
        active_at_replan: List[int],
        solver_time_s_total: float,
        total_compute_cost: float,
        full_instance: Dict[str, Any],
    ) -> Dict[str, float]:
        """Aggregate per-instance ReplanGate trigger stats for the summary row."""
        n_replans = len(replan_times)
        if n_replans >= 2:
            gaps = [
                float(replan_times[i] - replan_times[i - 1])
                for i in range(1, n_replans)
            ]
            mean_gap = float(np.mean(gaps))
            min_gap = float(np.min(gaps))
        else:
            mean_gap = 0.0
            min_gap = 0.0

        mean_active = (
            float(np.mean(active_at_replan)) if active_at_replan else 0.0
        )
        cost_per_replan = (
            float(total_compute_cost) / n_replans if n_replans > 0 else 0.0
        )
        compute_overhead_s = float(solver_time_s_total)

        # Burst lead time: for each peak, the first replan in (peak-1, peak+8).
        peaks_arr = full_instance.get('arrival_peaks', None)
        if peaks_arr is None:
            peaks: List[float] = []
        else:
            try:
                peaks = [float(p) for p in peaks_arr.tolist()]
            except AttributeError:
                peaks = [float(p) for p in peaks_arr]
        lead_times: List[float] = []
        for peak in peaks:
            window_lo = peak - 1.0
            window_hi = peak + 8.0
            matches = [
                rt for rt in replan_times
                if window_lo <= rt <= window_hi
            ]
            if matches:
                lead_times.append(matches[0] - peak)
        if lead_times:
            mean_lead = float(np.mean(lead_times))
            n_covered = len(lead_times)
        else:
            mean_lead = 0.0
            n_covered = 0

        return {
            'mean_inter_replan_min': mean_gap,
            'min_inter_replan_min': min_gap,
            'mean_active_req_at_replan': mean_active,
            'compute_overhead_s': compute_overhead_s,
            'cost_per_replan': cost_per_replan,
            'n_arrival_peaks': len(peaks),
            'n_peaks_covered_by_replan': n_covered,
            'mean_burst_lead_time_min': mean_lead,
        }

    def run_shift(
        self,
        arrival_stream: List[Dict[str, Any]],
        fleet_init: Dict[int, Vehicle],
        full_instance: Dict[str, Any],
        sharing_contract=None,
    ) -> List[Dict[str, Any]]:
        from coalition_contract import planner_instance, constrain_plan
        import copy
        full_instance = dict(full_instance)
        mandatory = self.service_policy == 'mandatory'
        weight_spec = self.cost_weights if self.cost_weights is not None else full_instance.get('objective_weights')
        weights = (mandatory_objective_weights(weight_spec, self.alpha_p_ratio) if mandatory
                   else objective_weights(weight_spec, self.alpha_p_ratio))
        full_instance['service_policy'] = self.service_policy
        full_instance['objective_weights'] = dataclasses.asdict(weights)
        full_instance['objective_version'] = MANDATORY_OBJECTIVE_VERSION if mandatory else OBJECTIVE_VERSION
        stream_ids = [int(row['req_id']) for row in arrival_stream]
        if len(set(stream_ids)) != len(stream_ids):
            raise ValueError('Arrival stream contains duplicate request IDs')
        if mandatory and any(not math.isfinite(float(row['t_arrival']))
                             or not 0 <= row['t_arrival'] < self.T for row in arrival_stream):
            raise ValueError('Orders must arrive within the declared generation window [0, T)')
        stream_ids = set(stream_ids)
        if 'physical_config' in full_instance:
            from hardware_profiles import physical_configuration
            physical = physical_configuration(full_instance['physical_config'])
            full_instance['physical_config'] = physical.to_dict()
            full_instance['remaining_delivery_ids'] = set(stream_ids)
        if sharing_contract is not None:
            full_instance['sharing_contract'] = sharing_contract
            full_instance['sharing_assignments'] = {}
        contract = full_instance.get('sharing_contract')
        if contract is not None:
            full_instance['sharing_assignments'] = {}
            ids = [int(r['req_id']) for r in arrival_stream]
            if len(set(ids)) != len(ids) or set(ids) != set(contract.owners):
                raise ValueError('Arrival stream must match the contractual task book exactly')
            for row in arrival_stream:
                rid = int(row['req_id'])
                if (row.get('firm_owner') != contract.owners[rid]
                        or row.get('market_group', 'shared') != contract.groups[rid]
                        or tuple(row.get('allowed_modes', ('uav', 'adr'))) != contract.permissions[rid]):
                    raise ValueError('Arrival metadata differs from the sharing contract')
        t = 0.0
        requests: Dict[int, Request] = {}
        vehicles = copy.deepcopy(fleet_init)
        for veh in vehicles.values():
            if veh.battery_init is None:
                from ce_cpdptw_alns import _full_battery
                veh.battery_init = _full_battery(veh, full_instance)
        episode_log: List[Dict[str, Any]] = []

        total_operating_cost = 0.0
        total_operating_cost_uav = 0.0
        total_operating_cost_adr = 0.0
        total_penalty_cost = 0.0
        total_battery_penalty = 0.0
        total_compute_cost = 0.0
        replan_count = 0
        hold_count = 0
        replan_times: List[float] = []
        active_at_replan: List[int] = []
        solver_time_s_total = 0.0

        last_route_plan: Optional[Dict[int, List[Leg]]] = None
        last_replan_time: Optional[float] = None
        new_requests_since_replan = 0
        ticks_since_replan = 0
        last_solver_time_s = 0.0
        projection_count = 0
        if self.replan_gate is not None and hasattr(self.replan_gate, 'reset'):
            self.replan_gate.reset()

        event_count = 0
        while (mandatory and any(rid not in requests or requests[rid].status != 'delivered'
                                 for rid in stream_ids)) or (not mandatory and t < self.T):
            event_count += 1
            if mandatory and event_count > self.max_events:
                pending = stream_ids - {rid for rid, req in requests.items() if req.status == 'delivered'}
                raise MandatoryServiceError('event_guard_exceeded', t, pending, episode_log)
            for vehicle in vehicles.values():
                if vehicle.committed_leg is None:
                    vehicle.current_time = max(vehicle.current_time, t)
            new_this_tick = 0
            for r in arrival_stream:
                rid = int(r['req_id'])
                if r['t_arrival'] <= t and rid not in requests:
                    requests[rid] = Request(
                        req_id=rid,
                        t_arrival=float(r['t_arrival']),
                        t_pickup=float(r['t_pickup']),
                        t_delivery=float(r['t_delivery']),
                        pickup_node=int(r.get('pickup_node', full_instance['n_depots'] + rid)),
                        delivery_node=int(r.get('delivery_node', full_instance['n_depots'] + full_instance['n_req'] + rid)),
                        demand=float(r['demand']),
                        status='waiting_for_pickup',
                        firm_owner=r.get('firm_owner'),
                        market_group=r.get('market_group', 'shared'),
                        allowed_modes=tuple(r.get('allowed_modes', ('uav', 'adr'))),
                    )
                    new_this_tick += 1
            new_requests_since_replan += new_this_tick

            if 'physical_config' in full_instance:
                full_instance['remaining_delivery_ids'] = stream_ids - {
                    rid for rid, req in requests.items() if req.status == 'delivered'}

            residual = build_residual_instance(
                requests=requests,
                vehicles=vehicles,
                current_time=t,
                full_instance=full_instance,
            )
            residual['delta_minutes'] = self.delta
            residual['shift_minutes'] = self.T
            residual['service_policy'] = self.service_policy
            if contract is not None:
                residual['full_instance'] = planner_instance(full_instance, residual['active_requests'], vehicles)

            gate_features = None
            gate_decision = None
            gate_reason = 'fixed'
            gate_prob_replan = 1.0
            should_replan = True
            if self.replan_gate is not None:
                from replan_gate import build_replan_features
                gate_features = build_replan_features(
                    residual,
                    delta_minutes=self.delta,
                    shift_minutes=self.T,
                    last_replan_time=last_replan_time,
                    new_requests_since_replan=new_requests_since_replan,
                    ticks_since_replan=ticks_since_replan,
                    last_solver_time_s=last_solver_time_s,
                )
                residual['gate_features'] = gate_features
                residual['arrival_history'] = gate_features['arrival_history']
                if last_route_plan is None:
                    should_replan = True
                    gate_reason = 'initial'
                else:
                    should_replan, gate_decision = self._gate_decision(
                        self.replan_gate, gate_features, residual,
                    )
                    if hasattr(gate_decision, 'reason'):
                        gate_reason = str(gate_decision.reason)
                    elif isinstance(gate_decision, dict):
                        gate_reason = str(gate_decision.get('reason', 'gate'))
                    else:
                        gate_reason = 'gate'
                    if hasattr(gate_decision, 'prob_replan'):
                        gate_prob_replan = float(gate_decision.prob_replan)
                    elif isinstance(gate_decision, dict) and 'prob_replan' in gate_decision:
                        gate_prob_replan = float(gate_decision['prob_replan'])

            if mandatory and not residual['active_requests']:
                should_replan = False
                gate_reason = 'no_active_orders'
                last_route_plan = {}
            elif mandatory and not should_replan and not any(v.committed_leg for v in vehicles.values()):
                from ce_cpdptw_alns import _simulate_planned_leg
                cached_progress = False
                for vid, route in (last_route_plan or {}).items():
                    first = pick_first_uncommitted_leg(route, requests, int(full_instance['n_depots']))
                    if first is not None and first.from_node == vehicles[vid].current_node:
                        _, _, ok = _simulate_planned_leg(vehicles[vid], requests.get(first.request_id),
                                                         first, full_instance)
                        cached_progress = cached_progress or ok
                if not cached_progress:
                    should_replan = True
                    gate_reason = 'mandatory_progress'

            if should_replan:
                solve_start = time.perf_counter()
                route_plan = self.solver.solve(residual)
                solver_time_s = time.perf_counter() - solve_start
                last_solver_time_s = solver_time_s
                last_route_plan = route_plan
                last_replan_time = t
                new_requests_since_replan = 0
                ticks_since_replan = 0
                replan_count += 1
                compute_cost_step = self._compute_replan_cost(solver_time_s, residual)
                replan_times.append(float(t))
                n_active = sum(
                    1 for rq in requests.values()
                    if rq.status in ('waiting_for_pickup',
                                     'pickup_committed',
                                     'onboard',
                                     'delivery_committed')
                )
                active_at_replan.append(n_active)
                solver_time_s_total += solver_time_s
            else:
                route_plan = last_route_plan if last_route_plan is not None else {vid: [] for vid in vehicles}
                solver_time_s = 0.0
                compute_cost_step = 0.0
                hold_count += int(gate_reason != 'no_active_orders')

            rejected_legs = []
            if mandatory or contract is not None:
                route_plan, rejected_legs = constrain_plan(route_plan, requests, vehicles, full_instance)
                projection_count += len(rejected_legs)
            _commit_route_prefixes(route_plan, requests, vehicles, full_instance, contract, rejected_legs)

            recovered_stall = False
            if mandatory:
                future_arrivals = [float(row['t_arrival']) for row in arrival_stream
                                   if int(row['req_id']) not in requests]
                completions = [float(v.committed_leg.t_arrive) for v in vehicles.values()
                               if v.committed_leg is not None]
                if not future_arrivals and not completions and self.stalled_solver is not None:
                    started = time.perf_counter()
                    rescue = self.stalled_solver.solve(residual)
                    elapsed = time.perf_counter() - started
                    solver_time_s += elapsed
                    solver_time_s_total += elapsed
                    compute_cost_step += self._compute_replan_cost(elapsed, residual)
                    rescue, rescue_rejections = constrain_plan(rescue, requests, vehicles, full_instance)
                    rejected_legs.extend(rescue_rejections)
                    projection_count += len(rescue_rejections)
                    route_plan = _service_advancing_prefixes(rescue, vehicles)
                    _commit_route_prefixes(route_plan, requests, vehicles, full_instance, contract, rejected_legs)
                    completions = [float(v.committed_leg.t_arrive) for v in vehicles.values()
                                   if v.committed_leg is not None]
                    recovered_stall = bool(completions)
                    if recovered_stall:
                        last_route_plan = route_plan
                if not future_arrivals and not completions:
                    pending = stream_ids - {rid for rid, req in requests.items() if req.status == 'delivered'}
                    raise MandatoryServiceError('no_progress', t, pending, episode_log)
                events = future_arrivals + completions
                if completions:
                    events.append(t + self.delta)
                t_next = min(events)
                if t_next < t:
                    raise RuntimeError('Next execution event is in the past')
            else:
                t_next = min(t + self.delta, self.T)
            completed_legs, vehicles, step_operating_cost, step_battery_penalty, step_oc_uav, step_oc_adr = advance_time(
                t, t_next, vehicles, full_instance, requests
            )
            total_operating_cost += step_operating_cost
            total_operating_cost_uav += step_oc_uav
            total_operating_cost_adr += step_oc_adr
            total_battery_penalty += step_battery_penalty
            total_compute_cost += compute_cost_step

            completed_payload = []
            step_penalty_cost = 0.0

            for leg in completed_legs:
                req = requests.get(leg.request_id)
                veh = vehicles[leg.vehicle_id]

                if req is not None and leg.leg_type == 'pickup':
                    req.T_pickup_actual = leg.t_arrive_raw
                    step_penalty_cost += pickup_penalty(req.T_pickup_actual, req.t_pickup, weights)
                    if req.status == 'pickup_committed':
                        req.transition('onboard')

                elif req is not None and leg.leg_type == 'delivery':
                    req.T_delivery_actual = delivery_event_time(leg, full_instance)
                    if req.status == 'delivery_committed':
                        req.transition('delivered')
                    if 'physical_config' in full_instance:
                        full_instance['remaining_delivery_ids'].discard(req.req_id)

                    if req.T_pickup_actual is None:
                        raise ValueError('Delivered request lacks its actual pickup timestamp')
                    step_penalty_cost += delivery_penalty(req.T_delivery_actual, req.t_delivery, weights)

                completed_payload.append({
                    'request_id': leg.request_id,
                    'vehicle_id': leg.vehicle_id,
                    'leg_type': leg.leg_type,
                    'from_node': leg.from_node,
                    'to_node': leg.to_node,
                    't_depart': leg.t_depart,
                    't_arrive': leg.t_arrive,
                    't_arrive_raw': leg.t_arrive_raw,
                    'travel_time': leg.travel_time,
                    'operating_cost': leg.operating_cost,
                    'energy_used': leg.energy_used,
                    'travel_energy_used': leg.travel_energy_used,
                    'service_energy_used': leg.service_energy_used,
                    'load_before': leg.load_before,
                    'load_after': leg.load_after,
                    'battery_before': leg.battery_before,
                    'battery_after_travel': leg.battery_after_travel,
                    'battery_after': leg.battery_after,
                })

            total_penalty_cost += step_penalty_cost
            episode_log.append({
                't': t_next,
                'solver_time_s': solver_time_s,
                'replanned': bool(should_replan),
                'held': bool(not should_replan),
                'gate_reason': gate_reason,
                'recovered_stall': recovered_stall,
                'gate_prob_replan': gate_prob_replan,
                'new_requests_since_last_replan': int(new_requests_since_replan),
                'ticks_since_last_replan': int(ticks_since_replan),
                'compute_cost_step': compute_cost_step,
                'completed': completed_payload,
                'rejected_legs': rejected_legs,
                'waiting': sum(1 for r in requests.values() if r.status == 'waiting_for_pickup'),
                'onboard': sum(1 for r in requests.values() if r.status == 'onboard'),
                'delivered': sum(1 for r in requests.values() if r.status == 'delivered'),
                'operating_cost_step': step_operating_cost,
                'penalty_cost_step': step_penalty_cost,
                'battery_penalty_step': step_battery_penalty,
            })

            t = t_next
            ticks_since_replan += 1

        if mandatory and any(v.committed_leg is not None for v in vehicles.values()):
            # Drain committed-but-pending legs (e.g. post-last-delivery depot returns)
            # so their energy/operating cost show up in the totals.
            drain_t_end = max(
                float(v.committed_leg.t_arrive)
                for v in vehicles.values()
                if v.committed_leg is not None
            )
            (drain_completed, vehicles, drain_oc, drain_bp,
             drain_oc_uav, drain_oc_adr) = advance_time(
                t, drain_t_end, vehicles, full_instance, requests,
            )
            total_operating_cost += drain_oc
            total_operating_cost_uav += drain_oc_uav
            total_operating_cost_adr += drain_oc_adr
            total_battery_penalty += drain_bp
            drain_payload = []
            for leg in drain_completed:
                drain_payload.append({
                    'request_id': leg.request_id,
                    'vehicle_id': leg.vehicle_id,
                    'leg_type': leg.leg_type,
                    'from_node': leg.from_node,
                    'to_node': leg.to_node,
                    't_depart': leg.t_depart,
                    't_arrive': leg.t_arrive,
                    't_arrive_raw': leg.t_arrive_raw,
                    'travel_time': leg.travel_time,
                    'operating_cost': leg.operating_cost,
                    'energy_used': leg.energy_used,
                    'travel_energy_used': leg.travel_energy_used,
                    'service_energy_used': leg.service_energy_used,
                    'load_before': leg.load_before,
                    'load_after': leg.load_after,
                    'battery_before': leg.battery_before,
                    'battery_after_travel': leg.battery_after_travel,
                    'battery_after': leg.battery_after,
                })
            if drain_payload:
                episode_log.append({
                    't': drain_t_end,
                    'solver_time_s': 0.0,
                    'replanned': False,
                    'held': True,
                    'gate_reason': 'terminal_drain',
                    'gate_prob_replan': 0.0,
                    'new_requests_since_last_replan': 0,
                    'ticks_since_last_replan': 0,
                    'compute_cost_step': 0.0,
                    'completed': drain_payload,
                    'rejected_legs': [],
                    'waiting': sum(1 for r in requests.values() if r.status == 'waiting_for_pickup'),
                    'onboard': sum(1 for r in requests.values() if r.status == 'onboard'),
                    'delivered': sum(1 for r in requests.values() if r.status == 'delivered'),
                    'operating_cost_step': drain_oc,
                    'penalty_cost_step': 0.0,
                    'battery_penalty_step': drain_bp,
                })
                t = drain_t_end
            if any(v.committed_leg is not None for v in vehicles.values()):
                raise MandatoryServiceError('unfinished_terminal_legs', t, (), episode_log)

        delivered = [r for r in requests.values() if r.status == 'delivered']
        n_revealed = _count_revealed(requests)
        delivery_violations_min = [
            max(r.T_delivery_actual - r.t_delivery, 0.0)
            for r in delivered
            if r.T_delivery_actual is not None
        ]
        pickup_violations_min = [
            max(r.T_pickup_actual - r.t_pickup, 0.0)
            for r in requests.values()
            if r.T_pickup_actual is not None
        ]
        n_delivery_violated = sum(1 for v in delivery_violations_min if v > 0.0)
        n_pickup_violated = sum(1 for v in pickup_violations_min if v > 0.0)

        pickup_early_cost = float(weights.pickup_early * sum(
            max(r.t_pickup - r.T_pickup_actual, 0.0)
            for r in requests.values() if r.T_pickup_actual is not None
        ))
        pickup_late_cost = float(weights.pickup_late * sum(
            max(r.T_pickup_actual - r.t_pickup, 0.0)
            for r in requests.values() if r.T_pickup_actual is not None
        ))
        delivery_late_cost = float(weights.delivery_late * sum(
            max(r.T_delivery_actual - r.t_delivery, 0.0)
            for r in delivered if r.T_delivery_actual is not None
        ))

        episode_log.append({
            'summary': True,
            'physical_config': full_instance.get('physical_config'),
            'n_depots': int(full_instance['n_depots']),
            'remaining_delivery_ids': sorted(full_instance.get('remaining_delivery_ids', [])),
            'energy_model_version': ENERGY_MODEL_VERSION,
            'travel_energy_kj': sum(leg['travel_energy_used'] for row in episode_log
                                    for leg in row.get('completed', [])),
            'service_energy_kj': sum(leg['service_energy_used'] for row in episode_log
                                     for leg in row.get('completed', [])),
            'energy_used_kj': sum(leg['energy_used'] for row in episode_log
                                  for leg in row.get('completed', [])),
            'recharge_events': sum(1 for row in episode_log
                                   for leg in row.get('completed', [])
                                   if leg.get('leg_type') == 'depot'),
            'objective_version': full_instance['objective_version'],
            'service_policy': self.service_policy,
            'service_complete': len(delivered) == len(stream_ids),
            'total_orders': len(stream_ids),
            'arrival_window_min': self.T,
            'completion_time_min': max((leg['t_arrive'] for row in episode_log
                                        for leg in row.get('completed', [])
                                        if leg['leg_type'] == 'delivery'), default=0.),
            'delivery_penalty_clock': 'raw_arrival' if mandatory else 'service_completion',
            'objective_weights': dataclasses.asdict(weights),
            'sharing_budgets': contract.budgets() if contract is not None else None,
            'sharing_usage': contract.usage(full_instance['sharing_assignments']) if contract is not None else None,
            'sharing_assignments': dict(full_instance.get('sharing_assignments', {})),
            'plan_projection_count': projection_count,
            'total_revealed': n_revealed,
            'total_delivered': len(delivered),
            'undelivered': _count_undelivered(requests),
            'operating_cost': total_operating_cost,
            'operating_cost_uav': total_operating_cost_uav,
            'operating_cost_adr': total_operating_cost_adr,
            'penalty_cost': total_penalty_cost + total_battery_penalty,
            'pickup_early_cost': pickup_early_cost,
            'pickup_late_cost': pickup_late_cost,
            'delivery_late_cost': delivery_late_cost,
            'battery_penalty_cost': total_battery_penalty,
            'undelivered_penalty': weights.unserved * _count_undelivered(requests),
            'compute_cost': total_compute_cost,
            'replan_count': replan_count,
            'hold_count': hold_count,
            'pct_held': (
                float(hold_count) / float(hold_count + replan_count)
                if (hold_count + replan_count) > 0 else 0.0
            ),
            **self._build_trigger_log(
                replan_times=replan_times,
                active_at_replan=active_at_replan,
                solver_time_s_total=solver_time_s_total,
                total_compute_cost=total_compute_cost,
                full_instance=full_instance,
            ),
            'quality_cost': (total_operating_cost + total_penalty_cost
                             + total_battery_penalty
                             + weights.unserved * _count_undelivered(requests)),
            'total_cost': (total_operating_cost + total_penalty_cost
                           + total_battery_penalty
                           + weights.unserved * _count_undelivered(requests)
                           + total_compute_cost),
            'tw_delivery_violation_count': n_delivery_violated,
            'tw_delivery_violation_rate': n_delivery_violated / max(n_revealed, 1),
            'tw_delivery_violation_mean_min': float(np.mean(delivery_violations_min)) if delivery_violations_min else 0.0,
            'tw_delivery_violation_max_min': float(np.max(delivery_violations_min)) if delivery_violations_min else 0.0,
            'tw_pickup_violation_count': n_pickup_violated,
            'tw_pickup_violation_rate': n_pickup_violated / max(n_revealed, 1),
            'tw_pickup_violation_mean_min': float(np.mean(pickup_violations_min)) if pickup_violations_min else 0.0,
            'request_history': [
                {
                    'req_id': r.req_id,
                    'firm_owner': r.firm_owner,
                    'market_group': r.market_group,
                    'allowed_modes': r.allowed_modes,
                    'assigned_vehicle': r.assigned_vehicle,
                    'status': r.status,
                    't_pickup_target': r.t_pickup,
                    't_delivery_target': r.t_delivery,
                    'T_pickup_actual': r.T_pickup_actual,
                    'T_delivery_actual': r.T_delivery_actual,
                    'pickup_violation_min': max(r.T_pickup_actual - r.t_pickup, 0.0) if r.T_pickup_actual is not None else None,
                    'delivery_violation_min': max(r.T_delivery_actual - r.t_delivery, 0.0) if r.T_delivery_actual is not None else None,
                    'assigned_vehicle': r.assigned_vehicle,
                }
                for r in requests.values()
                if r.status != 'unrevealed'
            ],
        })

        return episode_log
