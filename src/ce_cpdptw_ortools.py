"""Common routing MILP: exact affine ADR, conservative interval UAV energy.

Optimality is for this inner approximation, NOT the nonlinear UAV problem.
The enclosure bounds the provisional active law, not real hardware error.
"""
from __future__ import annotations
from collections import defaultdict
from dataclasses import dataclass, field
import math
from typing import Any, Literal, Mapping, Sequence
import numpy as np
from ortools.linear_solver import pywraplp
from energy import UAV_PARAMETERS, _adr_power, _uav_power, coord_edge_energy_kj
from model_config import (SCALE_M_PER_COORD, UAV_CRUISE_MPS, ADR_CRUISE_MPS,
                          UAV_LANDING_MIN, UAV_TAKEOFF_MIN, UAV_SERVICE_MIN,
                          ADR_SERVICE_MIN, ENERGY_MODEL_VERSION)
from rotor_energy import induced_velocity

Mode = Literal["UAV", "ADR"]
Market = Literal["UAV", "ADR", "shared"]
V_UAV_MAX = UAV_CRUISE_MPS * 60. / SCALE_M_PER_COORD
V_ADR_MAX = ADR_CRUISE_MPS * 60. / SCALE_M_PER_COORD
V_UAV_DEPOT, V_ADR_DEPOT = V_UAV_MAX, V_ADR_MAX / 2.
V_UAV_MIN_PICKUP = V_UAV_MAX
V_ADR_MIN_PICKUP = min(2., ADR_CRUISE_MPS) * 60. / SCALE_M_PER_COORD
UAV_LAND_TAKEOFF_MIN, ADR_CUSTOMER_SERVICE_MIN = UAV_SERVICE_MIN, ADR_SERVICE_MIN
UAV_RECHARGE_MIN, ADR_RECHARGE_MIN = 10., 20.
UAV_LOW_BATTERY_FRAC, ADR_LOW_BATTERY_FRAC = .25, .20
EXACT_CONTRACT_VERSION = "common-load-envelope-1"

@dataclass(frozen=True)
class Request:
    id: str
    pickup: str
    delivery: str
    demand: float
    pickup_target: float
    pickup_latest: float
    delivery_target: float
    delivery_latest: float
    market: Market = "shared"  # Economic group, never physical access.
    owner_mode: Mode | None = None
    allowed_modes: tuple = ("UAV", "ADR")
    release_time: float = 0.
    picked_by: str | None = None
    assigned_to: str | None = None

@dataclass(frozen=True)
class Vehicle:
    id: str
    mode: Mode
    capacity: float
    battery_max: float
    battery_min: float
    depot_visit_limit: int
    speed_levels: tuple[str, ...] = ()
    initial_battery: float | None = None
    initial_time: float = 0.
    initial_load: float = 0.
    start_node: str | None = None
    allowed_depots: tuple | None = None

    @property
    def start_battery(self):
        return self.battery_max if self.initial_battery is None else self.initial_battery

@dataclass(frozen=True)
class ArcPhysics:
    distance: float  # Coordinate units.
    speed: float  # Coordinate units/minute.
    wind_xy: tuple = (0., 0.)
    heading: float = 0.

    def __post_init__(self):
        if (len(self.wind_xy) != 2 or not np.isfinite(
                [self.distance, self.speed, self.heading, *self.wind_xy]).all()
                or self.distance < 0 or self.speed <= 0):
            raise ValueError("Invalid arc physics")

    @property
    def minutes(self):
        return self.distance / self.speed

@dataclass(frozen=True)
class DepotCopyMeta:
    copy_id: str
    vehicle_id: str
    physical_depot_id: str
    role: str
    visit_index: int | None

@dataclass(frozen=True)
class CECPDPTWData:
    requests: Sequence[Request]
    vehicles: Sequence[Vehicle]
    physical_depots: Sequence[str]
    uav_depots: Sequence[str]
    adr_depots: Sequence[str]
    travel_time: Mapping[tuple, float]
    energy: Mapping[tuple, float]  # Historical features, not active coefficients.
    pickup_time_by_speed: Mapping[tuple, float] = field(default_factory=dict)
    pickup_energy_by_speed: Mapping[tuple, float] = field(default_factory=dict)
    depot_sharing: bool = True
    alpha_uav: float = .60
    alpha_adr: float = .10
    alpha_early_pickup: float = .02
    alpha_late_pickup: float = .10
    alpha_late_delivery: float = .15
    lambda_battery: float = 0.
    epsilon_shared: float = 1.
    epsilon_to_adr: float = 1.
    epsilon_to_uav: float = 1.
    uav_customer_service_time: float = UAV_SERVICE_MIN
    adr_customer_service_time: float = ADR_SERVICE_MIN
    uav_recharge_duration: float = 10.
    adr_recharge_duration: float = 20.
    big_m_time: float = 100_000.
    big_m_load: float = 100_000.
    big_m_battery: float = 1_000_000.
    undelivered_penalty: float = 0.
    arc_physics: Mapping[tuple, ArcPhysics] = field(default_factory=dict)
    service_policy: str = "mandatory"
    energy_scope: str = "actual_load_bounds"
    energy_load_intervals: int = 16
    sharing_remaining: Mapping[str, int] | None = None
    sharing_book_counts: Mapping[str, int] | None = None
    sharing_assignments: Mapping[str, str] = field(default_factory=dict)
    recharge_cost_includes_dwell: bool = True
    allow_delayed_departure: bool = False

@dataclass
class BuiltORToolsCECPDPTWModel:
    solver: Any
    status: int | None
    data: CECPDPTWData
    nodes_by_vehicle: dict
    depot_meta: dict
    depot_copies_by_vehicle: dict
    arcs: list
    adaptive_arcs: dict
    x: dict
    z: dict
    use_vehicle: dict
    depot_visit: dict
    arrival: dict
    completion: dict
    waiting: dict
    load: dict
    battery: dict
    early_pickup: dict
    late_pickup: dict
    late_delivery: dict
    # Battery parity: `battery_slack` is a vestigial field.
    # The MIP enforces reserve via the hard constraint
    #   battery[k, i] >= E + v.battery_min
    # on arcs into mid-copies. No soft slack G_i^k
    # variables are instantiated; constructor passes `{}` for this field. The
    # field is retained for pickled-model back-compat only.
    battery_slack: dict
    y_uav: dict
    y_adr: dict
    y_served: dict
    arc_energy: dict
    energy_certificate: dict
    scientific_objective: Any
    rank: dict
    deferral_cardinality_proven: bool = False
    cycle_departure: dict = field(default_factory=dict)

@dataclass(frozen=True)
class SyntheticAdapterConfig:
    n_uav: int
    n_adr: int
    n_depots_uav: int
    n_depots_adr: int
    depot_visit_limit: int | None = None
    alpha_uav: float = .60
    alpha_adr: float = .10
    alpha_early_pickup: float = .02
    alpha_late_pickup: float = .10
    alpha_late_delivery: float = .15
    lambda_battery: float = 0.
    latest_pickup_slack: float = 0.
    latest_delivery_slack: float = 0.
    depot_sharing: bool = True
    epsilon_shared: float = 1.
    epsilon_to_adr: float = 1.
    epsilon_to_uav: float = 1.
    use_discrete_pickup_speed: bool = False
    pickup_speed_grid_size: int = 5
    require_destination_access: bool = True
    strict_pair_access: bool = True
    depot_return_energy_payload_policy: str = "actual_load"
    big_m_time: float = 100_000.
    big_m_load: float = 100_000.
    big_m_battery: float = 1_000_000.
    energy_load_intervals: int = 16

def _power_bounds(lo, hi, speed, wind, heading):
    """Analytic interval enclosure in watts, without assumed total convexity.

    For fixed airspeed v/drag D, T=W+D increases with load, whereas
    b=v*D/hypot(W,D) decreases. In vi*sqrt(v^2+2*b*vi+vi^2)=c*T,
    the positive root increases with T and decreases with b. Hence vi is
    bounded by its endpoint roots. Enclose P=T*(b+vi)/eta using opposite
    endpoints for b. Numerical padding does not represent hardware error.
    """
    p = UAV_PARAMETERS
    v = math.hypot(speed * math.cos(heading) - wind[0], speed * math.sin(heading) - wind[1])
    drag = p.air_density_kg_m3 * p.drag_cd_area_m2 * v * v / 2.
    wl, wh = (p.gravity_m_s2 * (p.empty_mass_kg + q) for q in (lo, hi))
    vl = float(induced_velocity(wl + drag, v, math.atan2(drag, wl), parameters=p))
    vh = float(induced_velocity(wh + drag, v, math.atan2(drag, wh), parameters=p))
    bl, bh = v * drag / math.hypot(wh, drag), v * drag / math.hypot(wl, drag)
    lower, upper = (wl + drag) * (bl + vl) / p.efficiency, (wh + drag) * (bh + vh) / p.efficiency
    return max(0., lower * (1. - 1e-12)), upper * (1. + 1e-12)

def energy_interval_bounds(spec, mode, lo, hi, delta=0., customer=False):
    """kJ bounds for travel on origin post-service load and customer phases.

    Domain: 0<=lo<=hi, 0<=lo+delta<=hi+delta. Signed target demand changes
    load after travel/landing, before takeoff. Landed waiting has zero draw.
    """
    if not np.isfinite([lo, hi, delta]).all() or lo < 0 or hi < lo or lo + delta < -1e-10:
        raise ValueError("Invalid payload interval")
    if mode == "ADR":
        return tuple(coord_edge_energy_kj("adr", spec.distance, q, np.zeros(2), spec.speed) for q in (lo, hi))
    if mode != "UAV":
        raise ValueError("Invalid energy mode")
    low, high = _power_bounds(lo, hi, spec.speed * SCALE_M_PER_COORD / 60., spec.wind_xy, spec.heading)
    low, high = low * spec.minutes * .06, high * spec.minutes * .06
    if customer:
        land = _power_bounds(lo, hi, 0., spec.wind_xy, 0.)
        launch = _power_bounds(max(0., lo + delta), max(0., hi + delta), 0., spec.wind_xy, 0.)
        low += .06 * (UAV_LANDING_MIN * land[0] + UAV_TAKEOFF_MIN * launch[0])
        high += .06 * (UAV_LANDING_MIN * land[1] + UAV_TAKEOFF_MIN * launch[1])
    return low, high

def _admissible_depots(data, vehicle):
    depots = list(data.physical_depots if data.depot_sharing else
                  (data.uav_depots if vehicle.mode == "UAV" else data.adr_depots))
    return depots if vehicle.allowed_depots is None else [d for d in depots if d in vehicle.allowed_depots]

def _validate_data(data):
    if data.service_policy not in ("mandatory", "residual_defer", "legacy_optional"):
        raise ValueError("Unknown service policy")
    diagnostic = data.service_policy == "legacy_optional"
    # Battery parity: under mandatory service the MIP enforces reserve
    # as a hard arc-level constraint (battery[k,i] >= E + v.battery_min on
    # arcs into mid-copies); matches ALNS
    # (_simulate_leg rejects infeasible legs) and RL (CoopDecoder physics
    # battery_mask with pickup→delivery→depot chain lookahead). Any attempt
    # to re-introduce a soft slack via λ_B > 0 is rejected here to keep the
    # three paths identically-constrained.
    if not diagnostic and (data.lambda_battery or data.undelivered_penalty):
        raise ValueError("Common mandatory objective has no missed price or soft battery term")
    if data.energy_scope != "actual_load_bounds" and not diagnostic:
        raise ValueError("Fixed destination-demand energy is legacy diagnostic only")
    if data.energy_scope not in ("actual_load_bounds", "legacy_fixed"):
        raise ValueError("Unknown energy scope")
    if not isinstance(data.energy_load_intervals, int) or data.energy_load_intervals < 1:
        raise ValueError("Positive integer energy interval count required")
    if not diagnostic and (data.uav_customer_service_time != UAV_SERVICE_MIN or data.adr_customer_service_time != ADR_SERVICE_MIN):
        raise ValueError("Service time must match common phase proxy")
    if not diagnostic and (data.uav_recharge_duration != UAV_RECHARGE_MIN or data.adr_recharge_duration != ADR_RECHARGE_MIN):
        raise ValueError("Recharge durations must match common scalar transitions")
    for seq in ([v.id for v in data.vehicles], [r.id for r in data.requests],
                [n for r in data.requests for n in (r.pickup, r.delivery)], list(data.physical_depots)):
        if len(seq) != len(set(seq)):
            raise ValueError("Duplicate model IDs")
    if {n for r in data.requests for n in (r.pickup, r.delivery)} & set(data.physical_depots):
        raise ValueError("Customers and real depots must be distinct")
    for w in (data.alpha_uav, data.alpha_adr, data.alpha_early_pickup, data.alpha_late_pickup,
              data.alpha_late_delivery, data.lambda_battery, data.undelivered_penalty,
              data.uav_recharge_duration, data.adr_recharge_duration):
        if not math.isfinite(w) or w < 0:
            raise ValueError("Non-finite or negative model coefficient")
    for e in (data.epsilon_shared, data.epsilon_to_adr, data.epsilon_to_uav):
        if not math.isfinite(e) or not 0 <= e <= 1:
            raise ValueError("Sharing fractions must be in [0,1]")
    vehicles = {v.id: v for v in data.vehicles}
    for v in data.vehicles:
        if (not np.isfinite([v.capacity, v.battery_max, v.battery_min, v.start_battery, v.initial_time, v.initial_load]).all()
                or v.mode not in ("UAV", "ADR") or v.capacity <= 0 or v.battery_max <= 0
                or not 0 <= v.battery_min <= v.battery_max or not 0 <= v.start_battery <= v.battery_max
                or v.initial_time < 0 or not 0 <= v.initial_load <= v.capacity
                or not isinstance(v.depot_visit_limit, int) or v.depot_visit_limit < 0):
            raise ValueError(f"Invalid initial state/resources for {v.id}")
        if v.start_node is None and not _admissible_depots(data, v):
            raise ValueError(f"No known initial node for {v.id}")
        if v.allowed_depots is not None and not set(v.allowed_depots) <= set(data.physical_depots):
            raise ValueError("Recharge permissions refer to non-depots")
        if abs(sum(r.demand for r in data.requests if r.picked_by == v.id) - v.initial_load) > 1e-8:
            raise ValueError("Initial load must match known onboard deliveries")
    for r in data.requests:
        if (not np.isfinite([r.demand, r.pickup_target, r.delivery_target, r.release_time]).all()
                or r.demand <= 0 or min(r.pickup_target, r.delivery_target, r.release_time) < 0
                or r.owner_mode not in (None, "UAV", "ADR") or r.market not in ("shared", "UAV", "ADR")
                or not set(r.allowed_modes) <= {"UAV", "ADR"}):
            raise ValueError(f"Invalid request {r.id}")
        if r.market != "shared" and r.owner_mode != r.market:
            raise ValueError("Economic exclusive group must match owner, not permissions")
        if r.picked_by is not None and r.picked_by not in vehicles:
            raise ValueError("Unknown onboard carrier")
        if r.assigned_to is not None and r.assigned_to not in vehicles:
            raise ValueError("Unknown reserved carrier")
        if r.owner_mode is None and (data.sharing_remaining is not None or
                min(data.epsilon_shared, data.epsilon_to_adr, data.epsilon_to_uav) < 1):
            raise ValueError("Ownership required for sharing contract")
    for mapping in (data.sharing_remaining, data.sharing_book_counts):
        if mapping is not None and (set(mapping) != {"shared", "uav", "adr"} or
                any(not isinstance(n, (int, np.integer)) or n < 0 for n in mapping.values())):
            raise ValueError("Invalid persistent sharing counts/allowances")
    if data.sharing_remaining is not None and data.sharing_book_counts is None:
        raise ValueError("Persistent allowances need whole-book counts")
    if data.sharing_remaining is not None:
        fractions = dict(shared=data.epsilon_shared, uav=data.epsilon_to_adr, adr=data.epsilon_to_uav)
        if any(data.sharing_remaining[g] > math.floor(fractions[g] * data.sharing_book_counts[g] + 1e-12)
               for g in fractions):
            raise ValueError("Remaining sharing allowance exceeds whole-book budget")
    if any(mode not in ("uav", "adr") for mode in data.sharing_assignments.values()):
        raise ValueError("Unknown sharing-ledger mode")
    for key, tm in data.travel_time.items():
        if not math.isfinite(tm) or tm < 0:
            raise ValueError("Invalid travel time")
        if data.energy_scope == "actual_load_bounds":
            if key not in data.arc_physics or not math.isclose(tm, data.arc_physics[key].minutes, rel_tol=1e-9, abs_tol=1e-9):
                raise ValueError("Actual-load energy needs matching arc physics")
        elif key not in data.energy or not math.isfinite(data.energy[key]) or data.energy[key] < 0:
            raise ValueError("Invalid legacy energy")

def build_ce_cpdptw_ortools_model(data, *, solver_id="SCIP", use_adaptive_speed=False,
                                 tighten_battery_equalities=True, _solver=None):
    _validate_data(data)
    if use_adaptive_speed and data.pickup_time_by_speed:
        raise NotImplementedError("Speed-choice search not certified against common scalar timing")
    if not tighten_battery_equalities:
        raise ValueError("Common transitions require battery equalities")
    solver = _solver if _solver is not None else pywraplp.Solver.CreateSolver(solver_id)
    if solver is None:
        raise RuntimeError(f"OR-Tools solver unavailable: {solver_id}")
    S = lambda terms: solver.Sum(list(terms))
    vehicles = {v.id: v for v in data.vehicles}
    pickups = {r.pickup: r for r in data.requests if r.picked_by is None}
    deliveries = {r.delivery: r for r in data.requests}
    customer = set(pickups) | set(deliveries)
    delta = {n: pickups[n].demand if n in pickups else -deliveries[n].demand for n in customer}
    nodes, copies, meta, arcs = {}, {}, {}, []
    for k, v in vehicles.items():
        start = v.start_node or _admissible_depots(data, v)[0]
        start_id, end_id = f"__start__{k}", f"__terminal__{k}"
        copies[k] = [start_id, end_id]
        meta[start_id] = DepotCopyMeta(start_id, k, start, "start", 0)
        meta[end_id] = DepotCopyMeta(end_id, k, "__virtual_terminal__", "end", None)
        for h in range(v.depot_visit_limit):
            for depot in _admissible_depots(data, v):
                name = f"__recharge__{k}__{h}__{depot}"
                copies[k].append(name)
                meta[name] = DepotCopyMeta(name, k, depot, "mid", h)
        nodes[k] = sorted(customer) + copies[k]
        for i in nodes[k]:
            for j in nodes[k]:
                if i == j or i == end_id or j == start_id or (i == start_id and j == end_id):
                    continue
                if j == end_id and i in deliveries:
                    arcs.append((k, i, j))
                elif j == end_id:
                    continue
                elif (k, meta[i].physical_depot_id if i in meta else i,
                      meta[j].physical_depot_id if j in meta else j) in data.travel_time:
                    if j in meta and i in meta and meta[i].physical_depot_id == meta[j].physical_depot_id:
                        continue
                    arcs.append((k, i, j))
    physical = lambda n: meta[n].physical_depot_id if n in meta else n
    key = lambda k, i, j: (k, physical(i), physical(j))
    is_end = lambda n: n in meta and meta[n].role == "end"
    max_nodes = max((len(ns) for ns in nodes.values()), default=1)
    H = max([0., *[v.initial_time for v in data.vehicles],
             *[max(r.release_time, r.pickup_target, r.delivery_target) for r in data.requests]])
    H += (max_nodes + 2) * (max([0., *data.travel_time.values()])
        + max(data.uav_recharge_duration, data.adr_recharge_duration)
        + max(data.uav_customer_service_time, data.adr_customer_service_time) + 1.)
    M = 2 * H
    node_keys = [(k, n) for k in nodes for n in nodes[k]]
    request_keys = [(r.id, k) for r in data.requests for k in vehicles]
    binary = lambda name, keys: {q: solver.BoolVar(f"{name}{q}") for q in keys}
    continuous = lambda name, keys, ub: {q: solver.NumVar(0., ub, f"{name}{q}") for q in keys}
    cycle_departure = {
        k: solver.NumVar(v.initial_time, H, f"cycle_departure[{k}]")
        for k, v in vehicles.items()
    } if data.allow_delayed_departure else {}
    x, use = binary("x", arcs), binary("use", vehicles)
    depot_visit = binary("depot", [(k, n) for k in copies for n in copies[k]])
    arrival, completion, waiting = (continuous(n, node_keys, H) for n in ("arrival", "completion", "waiting"))
    load = {q: solver.NumVar(0., vehicles[q[0]].capacity, f"load{q}") for q in node_keys}
    battery = {q: solver.NumVar(0., vehicles[q[0]].battery_max, f"battery{q}") for q in node_keys}
    rank = continuous("rank", node_keys, max_nodes)
    early, late_p, late_d = (continuous(n, request_keys, H) for n in ("early", "late_p", "late_d"))
    yu, ya, ys = (binary(n, [r.id for r in data.requests]) for n in ("yu", "ya", "ys"))
    outgoing, incoming = defaultdict(list), defaultdict(list)
    for k, i, j in arcs:
        outgoing[k, i].append(x[k, i, j])
        incoming[k, j].append(x[k, i, j])
    out, inc = lambda k, n: S(outgoing[k, n]), lambda k, n: S(incoming[k, n])
    served = lambda r, k: inc(k, r.delivery)
    for k, v in vehicles.items():
        start, end = copies[k][:2]
        solver.Add(out(k, start) == use[k])
        solver.Add(inc(k, end) == use[k])
        solver.Add(S(served(r, k) for r in data.requests) >= use[k])
        solver.Add(S(served(r, k) for r in data.requests) <= len(data.requests) * use[k])
        # C2: per-vehicle total mid-recharge cap, matching the
        # simulator's max_recharges_per_vehicle. Without this the MIP could
        # activate one recharge copy per depot (up to n_depots total per
        # vehicle), exceeding the simulator budget and causing replay rejection
        # ("Common-model replay rejected exact route"). depot_visit_limit now
        # reads as per-vehicle total mid activations; the per-(vehicle, depot)
        # physical copies are still created but their activation sum is capped.
        _mid_copies_k = [n for n in copies[k] if n in meta and meta[n].role == "mid"]
        if _mid_copies_k and v.depot_visit_limit is not None:
            solver.Add(S(depot_visit[k, n] for n in _mid_copies_k) <= int(v.depot_visit_limit))
        for n in nodes[k]:
            if n not in (start, end):
                solver.Add(out(k, n) == inc(k, n))
                solver.Add(inc(k, n) <= 1)
            if n in meta:
                solver.Add(depot_visit[k, n] == (out(k, n) if n == start else inc(k, n)))
            if n == start:
                for var, val in ((arrival, v.initial_time), (completion, v.initial_time),
                                 (waiting, 0.), (load, v.initial_load), (battery, v.start_battery), (rank, 0.)):
                    solver.Add(var[k, n] == val)
                continue
            vis = inc(k, n)
            # Hard reserve: if this non-start node is
            # visited, SOC must be at or above the configured reserve. This is
            # the MILP equivalent of the RL decoder's battery mask and the
            # simulator's reachability check. Combined with arc-energy
            # continuity constraints (lines ~492-495), the implied bound on
            # outgoing-arc source SOC is battery_min + E, matching the hard-
            # mask rule the policy and simulator enforce. Start node is
            # excluded because start_battery can legitimately be below the
            # reserve for mid-shift rollover; the vehicle then must recharge
            # before continuing (depot nodes force battery = battery_max).
            solver.Add(battery[k, n] >= v.battery_min * vis)
            for times in (arrival, completion, waiting):
                solver.Add(times[k, n] <= H * vis)
            if n in pickups:
                r = pickups[n]
                service = data.uav_customer_service_time if v.mode == "UAV" else data.adr_customer_service_time
                ready = solver.BoolVar(f"ready[{k},{n}]")
                solver.Add(ready <= vis)
                solver.Add(completion[k, n] >= arrival[k, n] + service * vis)
                solver.Add(completion[k, n] >= (r.pickup_target + service) * vis)
                solver.Add(completion[k, n] <= arrival[k, n] + service * vis + M * ready)
                solver.Add(completion[k, n] <= (r.pickup_target + service) * vis + M * (1 - ready))
                solver.Add(waiting[k, n] == completion[k, n] - arrival[k, n] - service * vis)
            else:
                service = (data.uav_customer_service_time if v.mode == "UAV" else data.adr_customer_service_time) if n in customer else 0.
                if n in meta and meta[n].role == "mid":
                    service = data.uav_recharge_duration if v.mode == "UAV" else data.adr_recharge_duration
                    solver.Add(battery[k, n] >= v.battery_max * vis)
                solver.Add(completion[k, n] == arrival[k, n] + service * vis)
                solver.Add(waiting[k, n] == 0)
            if is_end(n):
                solver.Add(load[k, n] == 0)
    for r in data.requests:
        solver.Add(S(served(r, k) for k in vehicles) == ys[r.id])
        solver.Add(yu[r.id] == S(served(r, k) for k, v in vehicles.items() if v.mode == "UAV"))
        solver.Add(ya[r.id] == S(served(r, k) for k, v in vehicles.items() if v.mode == "ADR"))
        if data.service_policy == "mandatory" or r.picked_by is not None:
            solver.Add(ys[r.id] == 1)
        for k, v in vehicles.items():
            y = served(r, k)
            if (v.mode not in r.allowed_modes or (r.picked_by is not None and r.picked_by != k)
                    or (r.assigned_to is not None and r.assigned_to != k)):
                solver.Add(y == 0)
            if r.id in data.sharing_assignments and data.sharing_assignments[r.id] != v.mode.lower():
                solver.Add(y == 0)
            if r.picked_by is None:
                solver.Add(inc(k, r.pickup) == y)
                solver.Add(rank[k, r.delivery] >= rank[k, r.pickup] + 1 - (max_nodes + 1) * (1 - y))
                solver.Add(early[r.id, k] >= r.pickup_target - arrival[k, r.pickup] - M * (1 - y))
                solver.Add(late_p[r.id, k] >= arrival[k, r.pickup] - r.pickup_target - M * (1 - y))
            else:
                solver.Add(early[r.id, k] == 0)
                solver.Add(late_p[r.id, k] == 0)
            solver.Add(late_d[r.id, k] >= arrival[k, r.delivery] - r.delivery_target - M * (1 - y))
            for penalties in (early, late_p, late_d):
                solver.Add(penalties[r.id, k] <= H * y)
    energy_vars, certificates = {}, []
    for k, i, j in arcs:
        v, active = vehicles[k], x[k, i, j]
        tm = 0. if is_end(j) else data.travel_time[key(k, i, j)]
        departure = cycle_departure.get(k, completion[k, i]) if i == copies[k][0] else completion[k, i]
        solver.Add(arrival[k, j] >= departure + tm - M * (1 - active))
        solver.Add(arrival[k, j] <= departure + tm + M * (1 - active))
        solver.Add(rank[k, j] >= rank[k, i] + 1 - (max_nodes + 1) * (1 - active))
        d = delta.get(j, 0.)
        load_m = 2 * v.capacity + abs(d)
        solver.Add(load[k, j] >= load[k, i] + d - load_m * (1 - active))
        solver.Add(load[k, j] <= load[k, i] + d + load_m * (1 - active))
        if j in customer:
            solver.Add(departure >= pickups.get(j, deliveries.get(j)).release_time - M * (1 - active))
        if is_end(j):
            E, max_e = 0., 0.
        elif data.energy_scope == "legacy_fixed":
            E = max_e = data.energy[key(k, i, j)]
        else:
            spec = data.arc_physics[key(k, i, j)]
            lo, hi = max(0., -d), min(v.capacity, v.capacity - d)
            if hi < lo:
                solver.Add(active == 0)
                E, max_e = 0., 0.
            elif v.mode == "ADR":
                e0, e1 = energy_interval_bounds(spec, "ADR", 0., v.capacity)
                E, max_e = e0 + (e1 - e0) / v.capacity * load[k, i], e1
            else:
                bounds = np.linspace(lo, hi, data.energy_load_intervals + 1) if hi > lo else [lo, hi]
                bits, uppers = [], []
                for b, (left, right) in enumerate(zip(bounds[:-1], bounds[1:])):
                    bit = solver.BoolVar(f"payload_bin[{k},{i},{j},{b}]")
                    bits.append(bit)
                    solver.Add(load[k, i] >= float(left) - v.capacity * (1 - bit))
                    solver.Add(load[k, i] <= float(right) + v.capacity * (1 - bit))
                    lower, upper = energy_interval_bounds(spec, "UAV", float(left), float(right), d, j in customer)
                    uppers.append(upper)
                    certificates.append(dict(arc=(k, i, j), load_lo=float(left), load_hi=float(right), target_delta=d,
                        lower_kj=lower, upper_kj=upper, max_overestimate_kj=upper - lower,
                        relative_gap_bound=(upper - lower) / lower if lower else 0.))
                solver.Add(S(bits) == active)
                E, max_e = S(upper * bit for upper, bit in zip(uppers, bits)), max(uppers)
        energy_vars[k, i, j] = E
        BM = v.battery_max + max_e
        solver.Add(battery[k, i] >= E - max_e * (1 - active))
        # On arcs ending at a mid-recharge-copy,
        # the base `battery[k, i] >= E` only ensures the arc is energy-feasible,
        # NOT that the vehicle arrives with battery >= reserve. The common
        # simulator's _simulate_leg does enforce arrival >= reserve on depot
        # legs, so the MIP's chosen arc gets replay-rejected when it plans a
        # depot approach that lands below reserve. Strengthen: on arcs into
        # mid copies, require `battery[k, i] >= E + reserve` so pre-recharge
        # arrival is above the planning reserve, matching the simulator.
        if j in meta and meta[j].role == "mid":
            solver.Add(battery[k, i] >= E + v.battery_min - (v.battery_max + max_e) * (1 - active))
        if j not in meta or meta[j].role == "end":
            solver.Add(battery[k, j] >= battery[k, i] - E - BM * (1 - active))
            solver.Add(battery[k, j] <= battery[k, i] - E + BM * (1 - active))
    fractions = {"shared": data.epsilon_shared, "uav": data.epsilon_to_adr, "adr": data.epsilon_to_uav}
    counts = data.sharing_book_counts or {g: sum(r.market.lower() == g for r in data.requests) for g in fractions}
    remaining = data.sharing_remaining if data.sharing_remaining is not None else {
        g: math.floor(fractions[g] * counts[g] + 1e-12) for g in fractions}
    for group in fractions:
        cross = [ya[r.id] if r.owner_mode == "UAV" else yu[r.id] for r in data.requests
                 if r.market.lower() == group and r.owner_mode is not None and r.picked_by is None
                 and r.id not in data.sharing_assignments]
        solver.Add(S(cross) <= remaining[group])
    objective = S((data.alpha_uav if vehicles[k].mode == "UAV" else data.alpha_adr)
                  * (0. if is_end(j) else data.travel_time[key(k, i, j)]) * x[k, i, j] for k, i, j in arcs)
    # C1: price recharge dwell into operating cost. A recharge-copy activation
    # (depot_visit[k, n] with meta[n].role == "mid") is a stay-at-pad motion;
    # its travel_time is 0 by graph construction, but recharge_duration is a
    # paid wait. Without this term the MIP uses recharge arcs as free time
    # shifting at zero operating cost, inflating recharge counts. Mirrors the
    # simulator's _simulate_leg operating_cost += α · recharge_dwell.
    # gated on data.recharge_cost_includes_dwell so the MIP matches
    # the simulator's scoring when C1 is toggled off via --no-recharge-dwell-cost.
    if data.recharge_cost_includes_dwell:
        objective += S(
            (data.uav_recharge_duration if vehicles[k].mode == "UAV" else data.adr_recharge_duration)
            * (data.alpha_uav if vehicles[k].mode == "UAV" else data.alpha_adr)
            * depot_visit[k, n]
            for k in copies for n in copies[k] if n in meta and meta[n].role == "mid")
    # Objective parity: the α weights below come from the same
    # ObjectiveWeights fields (alpha_early_pickup, alpha_late_pickup,
    # alpha_late_delivery) used by ALNS (via reward.pickup_penalty /
    # delivery_penalty at ce_cpdptw_alns.py:697-699) and VNS (via
    # ce_cpdptw_vrp_heuristics._route_cost_energy:198-201). Under
    # service_policy='mandatory' (bench default) no battery or
    # undelivered term enters this objective — matches ALNS/VNS which
    # also zero out those terms via mandatory_objective_weights
    # (reward.py:222-228).
    objective += S(data.alpha_early_pickup * early[r, k] + data.alpha_late_pickup * late_p[r, k]
                   + data.alpha_late_delivery * late_d[r, k] for r, k in request_keys)
    if data.service_policy == "legacy_optional":
        # Legacy-optional path only: adds λ_U × (missed-count). The bench
        # NEVER takes this branch — mandatory is enforced in the runner.
        if data.lambda_battery:
            raise NotImplementedError("Legacy soft-shortfall optimization not certified")
        objective += data.undelivered_penalty * S(1 - ys[r.id] for r in data.requests)
    solver.Minimize(-S(ys.values()) if data.service_policy == "residual_defer" else objective)
    certificate = dict(version=EXACT_CONTRACT_VERSION, energy_model=ENERGY_MODEL_VERSION,
        scope=data.energy_scope, adr_energy="affine_actual_origin_load",
        uav_energy="analytic_conservative_interval_upper" if data.energy_scope == "actual_load_bounds" else "invalid_historical_fixed",
        intervals=data.energy_load_intervals, bins=certificates,
        max_overestimate_kj=max((c["max_overestimate_kj"] for c in certificates), default=0.),
        max_relative_gap_bound=max((c["relative_gap_bound"] for c in certificates), default=0.),
        physical_calibration_certified=False, nonlinear_optimality_certified=False,
        phase_scope="diagnostic_customer_phases_only; configured_depot_reserve_continuation_unsupported",
        payload_domains_kg={v.id: [0., v.capacity] for v in data.vehicles},
        battery_policy="hard_reserve_at_visited_non_start_nodes; parity_with_rl_hard_mask; start_may_be_below_reserve_for_mid_shift_rollover",
        objective_scope="five_raw_arrival_future_costs" if data.service_policy != "legacy_optional" else "legacy_diagnostic",
        service_policy=data.service_policy, sharing_book_counts=dict(counts), sharing_remaining=dict(remaining),
        final_depot_return=False, recharge_copy_limit={v.id: v.depot_visit_limit for v in data.vehicles})
    return BuiltORToolsCECPDPTWModel(solver, None, data, nodes, meta, copies, arcs, {}, x, {}, use,
        depot_visit, arrival, completion, waiting, load, battery, early, late_p, late_d, {}, yu, ya, ys,
        energy_vars, certificate, objective, rank, cycle_departure=cycle_departure)

def finish_residual_cardinality(built, value_fn):
    """Fix attained service count then minimize five costs, without missed price.

    Interrupted phase one gives an incumbent cardinality, not an optimum.
    """
    count = sum(int(value_fn(y) > .5) for y in built.y_served.values())
    built.solver.Add(built.solver.Sum(list(built.y_served.values())) == count)
    built.solver.Minimize(built.scientific_objective)
    return count

def solve_ce_cpdptw_ortools(data, *, solver_id="SCIP", use_adaptive_speed=False,
                           tighten_battery_equalities=True, time_limit_seconds=3600., mip_gap=.01,
                           log_to_console=False):
    import time
    built = build_ce_cpdptw_ortools_model(data, solver_id=solver_id,
        use_adaptive_speed=use_adaptive_speed, tighten_battery_equalities=tighten_battery_equalities)
    solver = built.solver
    (solver.EnableOutput if log_to_console else solver.SuppressOutput)()
    params = pywraplp.MPSolverParameters()
    if mip_gap is not None:
        params.SetDoubleParam(pywraplp.MPSolverParameters.RELATIVE_MIP_GAP,
                             0. if data.service_policy == "residual_defer" else float(mip_gap))
    if time_limit_seconds is not None:
        solver.SetTimeLimit(max(1, int(1000 * time_limit_seconds)))
    start = time.monotonic()
    built.status = solver.Solve(params)
    if data.service_policy == "residual_defer" and built.status in (pywraplp.Solver.OPTIMAL, pywraplp.Solver.FEASIBLE):
        built.deferral_cardinality_proven = built.status == pywraplp.Solver.OPTIMAL
        finish_residual_cardinality(built, lambda var: var.solution_value())
        if mip_gap is not None:
            params.SetDoubleParam(pywraplp.MPSolverParameters.RELATIVE_MIP_GAP, float(mip_gap))
        if time_limit_seconds is not None:
            remaining = time_limit_seconds - (time.monotonic() - start)
            if remaining <= 0:
                built.status = pywraplp.Solver.NOT_SOLVED
                return built
            solver.SetTimeLimit(max(1, int(1000 * remaining)))
        built.status = solver.Solve(params)
    return built

def to_numpy(value):
    return value.detach().cpu().numpy() if hasattr(value, "detach") else np.asarray(value)

def select_instance(instance_or_batch, batch_index=0):
    if to_numpy(instance_or_batch["x"]).ndim != 3:
        return dict(instance_or_batch)
    selected = {}
    for key, value in instance_or_batch.items():
        if isinstance(value, (str, dict)) or hasattr(value, "budgets"):
            selected[key] = value
            continue
        array = to_numpy(value)
        selected[key] = value if array.ndim == 0 else value[batch_index]
    return selected

def build_ce_cpdptw_data_from_synthetic(instance, config):
    """Static initial states through the same adapter; full book cannot defer."""
    if instance.get("physical_config") is not None:
        raise NotImplementedError("Configured reserve/depot/continuation physics are not encoded in this customer-phase MILP")
    if config.lambda_battery or config.latest_pickup_slack or config.latest_delivery_slack:
        raise ValueError("Common model uses raw targets and hard non-exhaustion")
    if config.use_discrete_pickup_speed:
        raise NotImplementedError("Adaptive speed grids not common-model certified")
    if config.depot_return_energy_payload_policy != "actual_load":
        raise ValueError("Fixed depot-return payload policy is no longer a scientific option")
    from ce_cpdptw_alns import build_initial_fleet_from_instance, SyntheticALNSConfig, Request as SimRequest
    from ce_cpdptw_rolling_baselines import ExactRollingConfig, _build_exact_data_from_residual
    ndep, nreq = int(to_numpy(instance["n_depots"]).item()), int(to_numpy(instance["n_req"]).item())
    fleet = build_initial_fleet_from_instance(instance, SyntheticALNSConfig(
        config.n_uav, config.n_adr, config.n_depots_uav, config.n_depots_adr, depot_sharing=config.depot_sharing))
    for key, attr in (("start_node", "current_node"), ("start_time", "current_time")):
        if key in instance:
            values = to_numpy(instance[key]).reshape(-1)
            if len(values) != len(fleet) or not np.isfinite(values).all():
                raise ValueError("Invalid static initial node/time arrays")
            for vid, v in fleet.items():
                if key == "start_node" and (values[vid] != int(values[vid]) or not 0 <= values[vid] < ndep + 2 * nreq):
                    raise ValueError("Invalid initial node")
                setattr(v, attr, int(values[vid]) if key == "start_node" else float(values[vid]))
    for key in ("start_load", "delivery_owner"):
        if key in instance:
            values = to_numpy(instance[key]).reshape(-1)
            if np.any(values > 0 if key == "start_load" else values >= 0):
                raise ValueError("Loaded/reserved state requires explicit residual request carriers")
    if "start_capacity" in instance and not np.allclose(to_numpy(instance["start_capacity"]).reshape(-1),
                                                        [v.capacity for v in fleet.values()]):
        raise ValueError("Loaded state requires the explicit residual adapter")
    demand, tw = to_numpy(instance["demand"]).reshape(-1), to_numpy(instance["time_window"]).reshape(-1)
    owners, groups, contract = instance.get("firm_owner"), instance.get("market_group"), instance.get("sharing_contract")
    releases = to_numpy(instance.get("t_arrival", np.zeros(nreq))).reshape(-1)
    if len(releases) != nreq or not np.isfinite(releases).all() or np.any(releases < 0):
        raise ValueError("Invalid static request revelation times")
    if fleet and any(np.any(releases > v.current_time) for v in fleet.values()):
        raise ValueError("Static immediate-departure reference requires already revealed orders; use the rolling adapter for an arrival stream")
    requests = {}
    for rid in range(nreq):
        owner = contract.owners[rid] if contract else (owners[rid] if owners is not None else None)
        group = contract.groups[rid] if contract else (groups[rid] if groups is not None else "shared")
        modes = contract.permissions[rid] if contract else ("uav", "adr")
        requests[rid] = SimRequest(rid, float(releases[rid]), float(tw[ndep + rid]), float(tw[ndep + nreq + rid]),
            ndep + rid, ndep + nreq + rid, float(demand[ndep + rid]), status="waiting_for_pickup",
            firm_owner=owner, market_group=group, allowed_modes=modes)
    revised = dict(instance, service_policy="mandatory")
    revised["objective_weights"] = dict(operating_uav=config.alpha_uav, operating_adr=config.alpha_adr,
        pickup_early=config.alpha_early_pickup, pickup_late=config.alpha_late_pickup,
        delivery_late=config.alpha_late_delivery, unserved=0., battery_shortfall=0.)
    cfg = ExactRollingConfig(config.n_uav, config.n_adr, config.n_depots_uav, config.n_depots_adr,
        depot_sharing=config.depot_sharing,
        depot_visit_limit=config.depot_visit_limit if config.depot_visit_limit is not None else max(1, nreq),
        energy_load_intervals=config.energy_load_intervals, epsilon_shared=config.epsilon_shared,
        epsilon_to_adr=config.epsilon_to_adr, epsilon_to_uav=config.epsilon_to_uav)
    import sys
    data, *_ = _build_exact_data_from_residual(module=sys.modules[__name__],
        residual=dict(full_instance=revised, active_requests=requests, vehicles=fleet), config=cfg,
        available_vehicle_ids=list(fleet), service_policy="mandatory")
    return data

def solve_synthetic_instance(instance, config, *, batch_index=0, **kwargs):
    data = build_ce_cpdptw_data_from_synthetic(select_instance(instance, batch_index), config)
    built = solve_ce_cpdptw_ortools(data, **kwargs)
    return built, extract_solution(built)

def _display_node(node, meta):
    if node not in meta:
        return node
    m = meta[node]
    return "terminal(no return)" if m.role == "end" else f"{m.role}({m.physical_depot_id},{m.visit_index})"

def extract_solution(built, *, tolerance=.5, value_fn=None, feasible=None):
    value = value_fn or (lambda variable: variable.solution_value())
    ok = built.status in (pywraplp.Solver.OPTIMAL, pywraplp.Solver.FEASIBLE) if feasible is None else feasible
    result = dict(status=built.status, objective=None, routes={}, timing={}, battery={}, load={}, served_requests={},
        energy_certificate=built.energy_certificate, deferral_cardinality_proven=built.deferral_cardinality_proven,
        nonlinear_optimality_certified=False, complete=False, solver_lower_bound=None,
        solver_absolute_gap=None, solver_relative_gap=None,
        solver_bound_scope='conservative_inner_model_only; not a nonlinear lower bound')
    if not ok:
        return result
    result["objective"] = float(built.solver.Objective().Value())
    bound = float(built.solver.Objective().BestBound())
    if math.isfinite(bound):
        result['solver_lower_bound'] = bound
        result['solver_absolute_gap'] = max(0., result['objective'] - bound)
        result['solver_relative_gap'] = result['solver_absolute_gap'] / max(abs(result['objective']), 1e-12)
    if built.data.service_policy == 'residual_defer':
        result['solver_bound_scope'] += '; conditional on achieved served cardinality'
    successors = {(k, i): j for k, i, j in built.arcs if value(built.x[k, i, j]) >= tolerance}
    for v in built.data.vehicles:
        current = built.depot_copies_by_vehicle[v.id][0]
        route, seen = [], set()
        if (v.id, current) not in successors:
            result["routes"][v.id] = []
            continue
        while current is not None:
            if current in seen:
                raise RuntimeError("Cycle in extracted route")
            seen.add(current)
            shown, q = _display_node(current, built.depot_meta), (v.id, current)
            route.append(shown)
            result["timing"][v.id, shown] = dict(arrival=value(built.arrival[q]), completion=value(built.completion[q]))
            result["battery"][v.id, shown], result["load"][v.id, shown] = value(built.battery[q]), value(built.load[q])
            current = successors.get(q)
        result["routes"][v.id] = route
    for r in built.data.requests:
        result["served_requests"][r.id] = "UAV" if value(built.y_uav[r.id]) >= tolerance else (
            "ADR" if value(built.y_adr[r.id]) >= tolerance else "deferred")
    result["complete"] = all(m != "deferred" for m in result["served_requests"].values())
    result["obligation_scope"] = ("supplied_free_active_requests" if built.data.service_policy == "residual_defer"
                                  else "supplied_static_book")
    result["objective_scope"] = ("legacy_diagnostic_not_coalition_value" if built.data.service_policy == "legacy_optional"
        else "residual_subset_cost_not_coalition_value" if built.data.service_policy == "residual_defer"
        else "complete_book_cost" if result["complete"] else "no_complete_book_value")
    return result

def export_model_lp(built, path="ce_cpdptw_ortools.lp"):
    with open(path, "w", encoding="utf-8") as stream:
        stream.write(built.solver.ExportModelAsLpFormat(False))

def pc_uav_numpy(payload_kg, v_ground_mps):
    return _uav_power(payload_kg, v_ground_mps, np.zeros(2)) / 1000.

def pc_adr_numpy(payload_kg, v_ground_mps):
    return _adr_power(payload_kg, v_ground_mps) / 1000.

def energy_joules_from_power(*, distance, speed_coord_per_min, payload_kg, power_fn):
    # Historical API name: power input is kW and returned energy is kJ.
    return distance / speed_coord_per_min * 60. * power_fn(payload_kg, speed_coord_per_min * SCALE_M_PER_COORD / 60.)
