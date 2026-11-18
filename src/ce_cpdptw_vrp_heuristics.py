"""
Pure-Python VRP heuristics for CE-PDPTW rolling-horizon dispatch.

Three solvers implementing  solve(residual) → Dict[int, List[Leg]]:
  ClarkeWrightSolver    — pairwise savings merging (Clarke & Wright 1964)
  RegretInsertionSolver — regret-2 insertion    (Ropke & Pisinger 2006)
  VNSSolver             — Variable Neighborhood Search (Mladenović & Hansen 1997)

All three are model-conformant:
  - Heterogeneous UAV / ADR fleet with energy and capacity constraints
  - Battery recharge legs injected via _simulate_leg / _nearest_feasible_depot
  - Same rolling-horizon RollingHorizonDispatcher interface as the other solvers
  - Both online (dynamic t_arrival) and offline (all-at-t0) wrappers supplied
"""
from __future__ import annotations

import copy
import random
import math
import time
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from ce_cpdptw_alns import (
    Leg,
    Request,
    RollingHorizonDispatcher,
    SyntheticALNSConfig,
    Vehicle,
    _battery_threshold,
    _cruise_speed,
    _depot_speed,
    _edge_energy_kj,
    _get_distance_matrix,
    _is_arc_feasible,
    _nearest_feasible_depot,
    _pickup_speed,
    _schedule_route_departure,
    _service_time,
    _simulate_leg,
    _to_numpy,
    _wind_vec,
    build_dynamic_arrival_stream_from_instance,
    build_initial_fleet_from_instance,
    build_static_arrival_stream_from_instance,
    summarize_episode_log,
)
from ce_cpdptw_rolling_baselines import _make_forced_delivery_or_depot_route


# ── shared utilities ──────────────────────────────────────────────────────────

def _can_serve_req(veh: Vehicle, req: "Request", full_instance: Dict[str, Any]) -> bool:
    """Return False if this vehicle's mode cannot access the request's pickup or delivery node.

    Uses _is_arc_feasible which checks both the distance matrix (Floyd-Warshall
    path exists) AND the adjacency mask (acc_uav / acc_adr node flags), so nodes
    inside NFZ circles or behind ADR barriers are correctly excluded even when a
    detour path exists in the distance matrix.
    """
    mode = veh.normalized_mode()
    ok_start = _is_arc_feasible(full_instance, mode, int(veh.current_node), req.pickup_node)
    ok_leg   = _is_arc_feasible(full_instance, mode, req.pickup_node, req.delivery_node)
    return ok_start and ok_leg


def _battery_penalty(
    veh: Vehicle,
    pickup_node: int,
    full_instance: Dict[str, Any],
) -> float:
    """Extra time cost (minutes) when the vehicle must recharge before reaching pickup_node.

    Returns 1e9 when recharging is geometrically infeasible (no reachable depot),
    so the caller can treat this assignment as prohibited.  Returns 0.0 if no
    recharge is needed.

    Used as an additive penalty in the assignment cost functions so that a vehicle
    with low battery appears more expensive than a fully-charged one that can serve
    the request directly.
    """
    if not _needs_recharge(veh, pickup_node, "pickup", None, full_instance):
        return 0.0
    depot = _nearest_feasible_depot(int(veh.current_node), veh, full_instance)
    if depot is None:
        return 1e9
    mode = veh.normalized_mode()
    dm   = _get_distance_matrix(full_instance, mode)
    if not _is_arc_feasible(full_instance, mode, int(veh.current_node), depot):
        return 1e9
    if not _is_arc_feasible(full_instance, mode, depot, pickup_node):
        return 1e9
    d_to_depot   = float(dm[int(veh.current_node), depot])
    d_depot_pick = float(dm[depot, pickup_node])
    d_direct     = float(dm[int(veh.current_node), pickup_node])
    spd = max(_cruise_speed(mode), 1e-9)
    # Extra time = detour(via depot) − direct trip, lower-bounded at 0
    return max(0.0, (d_to_depot + d_depot_pick - d_direct) / spd)


def _needs_recharge(
    vs: Vehicle,
    to_node: int,
    leg_type: str,
    req: Optional[Request],
    full_instance: Dict[str, Any],
) -> bool:
    n_dep = int(full_instance["n_depots"])
    if vs.current_node < n_dep:
        return False
    if not _is_arc_feasible(full_instance, vs.normalized_mode(), vs.current_node, to_node):
        return False
    dm = _get_distance_matrix(full_instance, vs.normalized_mode())
    dist = float(dm[vs.current_node, to_node])
    if leg_type == "pickup" and req is not None:
        spd = _pickup_speed(dist, vs.current_time, req.t_pickup, vs.normalized_mode())
    else:
        spd = _cruise_speed(vs.normalized_mode())
    energy = _edge_energy_kj(
        mode=vs.normalized_mode(), distance=dist, payload=float(vs.load),
        wind_vec=_wind_vec(full_instance), speed=spd,
    )
    return float(vs.battery) - energy < _battery_threshold(vs)


def _tmin(fp: int, tp: int, mode: str, dm: np.ndarray, n_dep: int) -> float:
    """Travel time in minutes including service time at destination."""
    if fp == tp:
        return 0.0
    n = dm.shape[0]
    if fp < 0 or tp < 0 or fp >= n or tp >= n:
        return 1e9
    d = float(dm[fp, tp])
    if not math.isfinite(d) or d >= 1e9:
        return 1e9
    to_dep = tp < n_dep
    spd = _depot_speed(mode) if (to_dep and fp >= n_dep) else _cruise_speed(mode)
    svc = _service_time(mode) if not to_dep else 0.0
    return d / max(spd, 1e-9) + svc


def _route_cost(
    stops: List[Tuple[int, str]],
    start: int,
    mode: str,
    dm: np.ndarray,
    n_dep: int,
    req_list: List[Request],
) -> float:
    pos, cost = start, 0.0
    for ri, lt in stops:
        to = req_list[ri].pickup_node if lt == "pickup" else req_list[ri].delivery_node
        cost += _tmin(pos, to, mode, dm, n_dep)
        pos = to
    return cost


def _route_cost_energy(
    stops: List[Tuple[int, str]],
    veh: Vehicle,
    req_list: List[Request],
    full_instance: Dict[str, Any],
) -> float:
    """Physics-accurate route cost including recharge detour overhead.

    Uses _materialize (same execution path as the real dispatch) so the
    VNS objective accounts for energy detours rather than ignoring them.
    Returns 1e9 when materialization fails (energy infeasible route).

    Pickup/delivery time-window penalties are included to align VNS's
    optimisation objective with ALNS and the paper spec (operating + TW
    deviations). Battery is treated as a hard feasibility constraint only
    (materialisation fails on infeasible energy); no soft battery_cost term,
    matching Gurobi/OR-Tools. The earlier travel-time-only objective made
    VNS systematically trade off TW compliance for shorter travel, producing
    costs that were hard to compare against ALNS/MILP numbers.

    Objective parity: the per-leg term pickup_penalty + delivery_penalty
    (pulled via reward.objective_from_instance at line 189) matches the
    α_pe·early + α_pl·late_pickup + α_dl·late_delivery terms Gurobi/OR-Tools
    add in ce_cpdptw_ortools.py:549-550 and the pickup_penalty/delivery_penalty
    calls ALNS makes in ce_cpdptw_alns.py:697-699. No battery_cost, no
    undelivered-penalty — the 5-cost structure under mandatory service.
    """
    if not stops:
        return 0.0
    vs = copy.deepcopy(veh)
    if vs.battery_init is None:
        vs.battery_init = float(vs.battery)
    legs, _ = _materialize(stops, vs, req_list, full_instance)
    if not legs:
        return 1e9
    # Travel time is the operating-cost proxy; TW deviations come from the
    # paper's pickup/delivery penalty functions (reward.py).
    from reward import (pickup_penalty, delivery_penalty,
                        objective_from_instance, delivery_event_time)
    weights = objective_from_instance(full_instance)
    req_by_id = {r.req_id: r for r in req_list}
    total = 0.0
    for leg in legs:
        total += leg.travel_time
        req = req_by_id.get(leg.request_id)
        if req is None:
            continue  # recharge or repositioning leg — no TW term
        if leg.leg_type == 'pickup':
            total += pickup_penalty(leg.t_arrive_raw, req.t_pickup, weights)
        elif leg.leg_type == 'delivery':
            total += delivery_penalty(delivery_event_time(leg, full_instance),
                                      req.t_delivery, weights)
    return total


def _check_cap(
    stops: List[Tuple[int, str]],
    capacity: float,
    init_load: float,
    req_list: List[Request],
) -> bool:
    load = init_load
    for ri, lt in stops:
        load += req_list[ri].demand if lt == "pickup" else -req_list[ri].demand
        if load < -1e-6 or load > capacity + 1e-6:
            return False
    return True


def _check_prec(stops: List[Tuple[int, str]]) -> bool:
    seen: set = set()
    for ri, lt in stops:
        if lt == "pickup":
            seen.add(ri)
        elif ri not in seen:
            return False
    return True


def _insertion_delta(
    stops: List[Tuple[int, str]],
    p: int,   # insert pickup before position p  (0 = first)
    d: int,   # insert delivery before position d (d >= p+1)
    ri: int,
    req_list: List[Request],
    start: int,
    mode: str,
    dm: np.ndarray,
    n_dep: int,
) -> float:
    """Cost increase from inserting request ri's pickup at index p and delivery at index d."""
    def phys(idx: int, after_insert: bool = False) -> int:
        """Physical node of stop at index idx (in the stops list, before/after pickup insert)."""
        return req_list[stops[idx][0]].pickup_node if stops[idx][1] == "pickup" \
               else req_list[stops[idx][0]].delivery_node

    pick_phys = req_list[ri].pickup_node
    dliv_phys = req_list[ri].delivery_node
    n = len(stops)

    def node_at(idx: int) -> int:
        if idx < 0:
            return start
        if idx >= n:
            return -1  # no successor
        r, lt = stops[idx]
        return req_list[r].pickup_node if lt == "pickup" else req_list[r].delivery_node

    # Cost before pickup insertion
    a = node_at(p - 1)
    b = node_at(p)
    cost_before_p = _tmin(a, b, mode, dm, n_dep) if b >= 0 else 0.0
    cost_after_p = (_tmin(a, pick_phys, mode, dm, n_dep) +
                    (_tmin(pick_phys, b, mode, dm, n_dep) if b >= 0 else 0.0))
    delta_p = cost_after_p - cost_before_p

    # After pickup inserted at p, the sequence is [stops[0..p-1], pick, stops[p..n-1]].
    # exp_node(idx) gives the physical node at position idx in this expanded sequence.
    # d is the delivery insertion position in this expanded sequence (d >= p+1).
    def exp_node(idx: int) -> int:
        if idx == p:
            return pick_phys      # the pickup we just inserted
        if idx < p:
            return node_at(idx)   # unchanged prefix
        return node_at(idx - 1)   # shifted by 1 due to pickup insertion

    c_before_d = exp_node(d - 1)
    c_after_d  = exp_node(d)

    cost_before_d = _tmin(c_before_d, c_after_d, mode, dm, n_dep) if c_after_d >= 0 else 0.0
    cost_after_d = (_tmin(c_before_d, dliv_phys, mode, dm, n_dep) +
                    (_tmin(dliv_phys, c_after_d, mode, dm, n_dep) if c_after_d >= 0 else 0.0))
    delta_d = cost_after_d - cost_before_d

    return delta_p + delta_d


def _best_insertion(
    ri: int,
    stops: List[Tuple[int, str]],
    veh: Vehicle,
    req_list: List[Request],
    dm: np.ndarray,
    n_dep: int,
    full_instance: Optional[Dict[str, Any]] = None,
    ablate_nfz: bool = False,
    ablate_battery: bool = False,
) -> Tuple[float, int, int]:
    """Best (pickup_pos, delivery_pos) insertion for request ri into stops.

    When full_instance is provided, up to two additional checks are made:
      1. Accessibility gate  (skipped when ablate_nfz=True): _can_serve_req
         checks the distance matrix AND adjacency mask, rejecting NFZ/ADR-blocked
         nodes even if Floyd-Warshall gives a finite distance.
      2. Battery penalty     (skipped when ablate_battery=True): extra detour
         time for recharge before pickup; infeasible if no depot reachable.
    Setting ablate_nfz=True isolates the value of battery-aware planning.
    Setting ablate_battery=True isolates the value of NFZ-aware planning.
    Both False = full urban-constraint-aware planning (default).
    Both True  = constraint-blind planning (equivalent to passing full_instance=None).
    """
    mode = veh.normalized_mode()
    start = int(veh.current_node)
    n = len(stops)
    best_cost, best_p, best_d = 1e18, 0, 1

    # ── urban constraint checks (O(1), applied once per vehicle-request pair) ──
    req = req_list[ri]
    if full_instance is not None:
        if not ablate_nfz and not _can_serve_req(veh, req, full_instance):
            return 1e18, 0, 1
        if not ablate_battery:
            batt_pen = _battery_penalty(veh, req.pickup_node, full_instance)
            if batt_pen >= 1e9:
                return 1e18, 0, 1
        else:
            batt_pen = 0.0
    else:
        batt_pen = 0.0

    for p in range(n + 1):
        for d in range(p + 1, n + 2):
            candidate = list(stops)
            candidate.insert(p, (ri, "pickup"))
            candidate.insert(d, (ri, "delivery"))
            if not _check_cap(candidate, veh.capacity, veh.load, req_list):
                continue
            if not _check_prec(candidate):
                continue
            delta = _insertion_delta(stops, p, d, ri, req_list, start, mode, dm, n_dep) + batt_pen
            if delta < best_cost:
                best_cost, best_p, best_d = delta, p, d
    return best_cost, best_p, best_d


def _materialize(
    stops: List[Tuple[int, str]],
    veh_state: Vehicle,
    req_list: List[Request],
    full_instance: Dict[str, Any],
) -> Tuple[List[Leg], Vehicle]:
    legs: List[Leg] = []
    for ri, lt in stops:
        req = req_list[ri]
        to_node = req.pickup_node if lt == "pickup" else req.delivery_node
        # Pre-check: the travel-only estimate in _needs_recharge may underestimate
        # the actual (travel + service) energy the simulator will charge, so this
        # branch is the optimistic path. We also do a retry below.
        tried_recharge = _needs_recharge(veh_state, to_node, lt, req, full_instance)
        if tried_recharge:
            depot = _nearest_feasible_depot(veh_state.current_node, veh_state, full_instance)
            if depot is not None:
                leg, veh_state, ok = _simulate_leg(veh_state, None, "depot", depot, full_instance)
                if ok and leg is not None:
                    legs.append(leg)
                elif not ok:
                    break  # recharge leg itself infeasible — give up on this plan
        leg, veh_state_next, ok = _simulate_leg(veh_state, req, lt, to_node, full_instance)
        # Retry-with-recharge: the simulator uses full energy (travel + service);
        # if it rejects a leg that _needs_recharge thought was fine, insert a
        # depot visit and retry once before giving up. This keeps VNS honest
        # with the common simulator without duplicating its energy logic.
        if (not ok) and (not tried_recharge):
            depot = _nearest_feasible_depot(veh_state.current_node, veh_state, full_instance)
            if depot is not None:
                recharge_leg, veh_state_rc, ok_rc = _simulate_leg(
                    veh_state, None, "depot", depot, full_instance)
                if ok_rc and recharge_leg is not None:
                    legs.append(recharge_leg)
                    veh_state = veh_state_rc
                    leg, veh_state_next, ok = _simulate_leg(
                        veh_state, req, lt, to_node, full_instance)
        if ok and leg is not None:
            legs.append(leg)
            veh_state = veh_state_next
        else:
            break
    if legs and full_instance.get("allow_delayed_departure", False):
        legs, _ = _schedule_route_departure(legs, {r.req_id: r for r in req_list}, full_instance)
        veh_state.current_time = legs[-1].t_arrive
    return legs, veh_state


def _setup(
    residual: Dict[str, Any],
) -> Tuple[
    Dict[int, List[Leg]],   # routes (pre-filled forced)
    List[int],              # available_vehicle_ids
    List[Vehicle],          # veh_list
    List[Request],          # req_list (waiting_for_pickup only)
    Any,                    # full_instance
    float,                  # current_time
    int,                    # n_depots
]:
    vehicles       = residual["vehicles"]
    active_req     = residual["active_requests"]
    full_instance  = residual["full_instance"]
    current_time   = float(residual.get("current_time", 0.0))

    routes: Dict[int, List[Leg]] = {vid: [] for vid in vehicles}
    available_vids: List[int] = []

    for vid, veh in vehicles.items():
        forced = _make_forced_delivery_or_depot_route(veh, active_req, full_instance)
        if forced is not None:
            routes[vid] = forced
        else:
            available_vids.append(vid)

    req_list = [r for r in active_req.values() if r.status == "waiting_for_pickup"]
    veh_list = [vehicles[vid] for vid in available_vids]
    n_depots = int(full_instance["n_depots"])

    return routes, available_vids, veh_list, req_list, full_instance, current_time, n_depots


# ── 1. Clarke-Wright Savings ──────────────────────────────────────────────────

class ClarkeWrightSolver:
    """
    Savings-based constructive heuristic (Clarke & Wright 1964) adapted for PDPTW.

    Savings: s(i,j,k) = cost(depot→j) − cost(delivery_i→pickup_j)
    where "depot" is vehicle k's current position.  Routes are chained by
    decreasing savings, respecting capacity and pickup-before-delivery precedence.
    Unassigned requests fall back to cheapest-insertion.
    """

    def __init__(
        self,
        latest_pickup_slack: float = 30.0,
        latest_delivery_slack: float = 30.0,
        ablate_urban: bool = False,
        ablate_nfz: bool = False,
        ablate_battery: bool = False,
    ) -> None:
        self.latest_pickup_slack = latest_pickup_slack
        self.latest_delivery_slack = latest_delivery_slack
        self.ablate_urban = ablate_urban
        self.ablate_nfz = ablate_nfz
        self.ablate_battery = ablate_battery

    def solve(self, residual: Dict[str, Any]) -> Dict[int, List[Leg]]:
        routes, available_vids, veh_list, req_list, full_instance, current_time, n_depots = _setup(residual)
        vehicles = residual["vehicles"]

        if not req_list or not available_vids:
            return routes

        _ablate_nfz  = self.ablate_urban or self.ablate_nfz
        _ablate_batt = self.ablate_urban or self.ablate_battery

        n_veh = len(available_vids)
        n_req = len(req_list)

        dms = {
            "uav": _get_distance_matrix(full_instance, "uav"),
            "adr": _get_distance_matrix(full_instance, "adr"),
        }

        # Per-vehicle route stops
        veh_stops: List[List[Tuple[int, str]]] = [[] for _ in range(n_veh)]
        assigned = set()

        savings = []
        for k, veh in enumerate(veh_list):
            mode = veh.normalized_mode()
            dm = dms[mode]
            start = int(veh.current_node)
            batt_pen_j: Dict[int, float] = {}
            for i in range(n_req):
                ri = req_list[i]
                if not _ablate_nfz and not _is_arc_feasible(full_instance, mode, start, ri.pickup_node):
                    continue
                if not _ablate_nfz and not _is_arc_feasible(full_instance, mode, ri.pickup_node, ri.delivery_node):
                    continue
                for j in range(n_req):
                    if i == j:
                        continue
                    rj = req_list[j]
                    if not _ablate_nfz and not _is_arc_feasible(full_instance, mode, start, rj.pickup_node):
                        continue
                    if not _ablate_nfz and not _is_arc_feasible(full_instance, mode, rj.pickup_node, rj.delivery_node):
                        continue
                    cost_to_j = _tmin(start, rj.pickup_node, mode, dm, n_depots)
                    cost_chain = _tmin(ri.delivery_node, rj.pickup_node, mode, dm, n_depots)
                    if j not in batt_pen_j:
                        batt_pen_j[j] = 0.0 if _ablate_batt else _battery_penalty(veh, rj.pickup_node, full_instance)
                    pen = batt_pen_j[j]
                    if pen >= 1e9:
                        continue
                    s = cost_to_j - cost_chain - pen
                    savings.append((s, k, i, j))

        savings.sort(key=lambda x: -x[0])

        # Step 2: Greedily chain routes by savings.
        # Standard CW: chain unassigned request j onto the tail of vehicle k's route,
        # where the tail is request i (already on k, or k is empty and i is unassigned).
        # j must always be unassigned; i must be at k's tail or k must be empty+i unassigned.
        route_tail: Dict[int, int] = {}  # k -> last req index

        for s, k, i, j in savings:
            if s <= 0:
                break
            if j in assigned:
                continue
            if veh_stops[k]:
                # Route exists: i must be the tail (already assigned to k)
                if route_tail.get(k) != i:
                    continue
            else:
                # Empty route: both i and j must be unassigned
                if i in assigned:
                    continue
            candidate = list(veh_stops[k])
            if not veh_stops[k]:
                candidate.extend([(i, "pickup"), (i, "delivery")])
            candidate.extend([(j, "pickup"), (j, "delivery")])
            if not _check_cap(candidate, veh_list[k].capacity, veh_list[k].load, req_list):
                continue
            if not _check_prec(candidate):
                continue
            if not veh_stops[k]:
                veh_stops[k].extend([(i, "pickup"), (i, "delivery")])
                assigned.add(i)
            veh_stops[k].extend([(j, "pickup"), (j, "delivery")])
            assigned.add(j)
            route_tail[k] = j

        # Step 3: Assign remaining unassigned via cheapest insertion
        for i in range(n_req):
            if i in assigned:
                continue
            best_k, best_p, best_d, best_delta = -1, 0, 1, 1e18
            for k, veh in enumerate(veh_list):
                dm = dms[veh.normalized_mode()]
                delta, p, d = _best_insertion(i, veh_stops[k], veh, req_list, dm, n_depots,
                                              full_instance, _ablate_nfz, _ablate_batt)
                if delta < best_delta:
                    best_delta, best_k, best_p, best_d = delta, k, p, d
            if best_k < 0 or best_delta >= 1e17:
                continue  # no feasible insertion; request dropped
            veh_stops[best_k].insert(best_p, (i, "pickup"))
            veh_stops[best_k].insert(best_d, (i, "delivery"))
            assigned.add(i)

        # Step 4: Materialize
        for k, vid in enumerate(available_vids):
            vs = copy.deepcopy(vehicles[vid])
            if vs.battery_init is None:
                vs.battery_init = float(vs.battery)
            legs, _ = _materialize(veh_stops[k], vs, req_list, full_instance)
            routes[vid] = legs

        return routes


# ── 2. Regret-2 Insertion ─────────────────────────────────────────────────────

class RegretInsertionSolver:
    """
    Regret-2 insertion heuristic (Ropke & Pisinger 2006).

    At each step, for every unassigned request i:
      regret(i) = 2nd-best insertion cost − best insertion cost
    The request with the highest regret is inserted at its best position.
    High regret = large opportunity cost of not inserting now.
    """

    def __init__(
        self,
        latest_pickup_slack: float = 30.0,
        latest_delivery_slack: float = 30.0,
        ablate_urban: bool = False,
        ablate_nfz: bool = False,
        ablate_battery: bool = False,
    ) -> None:
        self.latest_pickup_slack = latest_pickup_slack
        self.latest_delivery_slack = latest_delivery_slack
        self.ablate_urban = ablate_urban
        self.ablate_nfz = ablate_nfz
        self.ablate_battery = ablate_battery

    def solve(self, residual: Dict[str, Any]) -> Dict[int, List[Leg]]:
        routes, available_vids, veh_list, req_list, full_instance, current_time, n_depots = _setup(residual)
        vehicles = residual["vehicles"]

        if not req_list or not available_vids:
            return routes

        _ablate_nfz  = self.ablate_urban or self.ablate_nfz
        _ablate_batt = self.ablate_urban or self.ablate_battery

        n_veh = len(available_vids)
        dms = {
            "uav": _get_distance_matrix(full_instance, "uav"),
            "adr": _get_distance_matrix(full_instance, "adr"),
        }

        veh_stops: List[List[Tuple[int, str]]] = [[] for _ in range(n_veh)]
        unassigned = set(range(len(req_list)))

        while unassigned:
            best_choice = None  # (regret, ri, best_k, best_p, best_d)

            for ri in unassigned:
                # Find best and second-best insertion across all vehicles
                insertions = []
                for k, veh in enumerate(veh_list):
                    dm = dms[veh.normalized_mode()]
                    delta, p, d = _best_insertion(ri, veh_stops[k], veh, req_list, dm, n_depots,
                                                  full_instance, _ablate_nfz, _ablate_batt)
                    if delta < 1e17:
                        insertions.append((delta, k, p, d))
                insertions.sort()

                if not insertions:
                    # No feasible insertion — skip for now (will be inserted last)
                    continue

                c1 = insertions[0][0]
                c2 = insertions[1][0] if len(insertions) >= 2 else c1 + 1e9
                regret = c2 - c1

                if best_choice is None or regret > best_choice[0]:
                    best_choice = (regret, ri, insertions[0][1], insertions[0][2], insertions[0][3])

            if best_choice is None:
                # Infeasible for all: force into least-loaded vehicle at tail
                ri = next(iter(unassigned))
                loads = [
                    sum(req_list[s[0]].demand for s in veh_stops[k] if s[1] == "pickup")
                    for k in range(n_veh)
                ]
                best_k = int(np.argmin(loads))
                veh_stops[best_k].extend([(ri, "pickup"), (ri, "delivery")])
                unassigned.discard(ri)
            else:
                _, ri, k, p, d = best_choice
                veh_stops[k].insert(p, (ri, "pickup"))
                veh_stops[k].insert(d, (ri, "delivery"))
                unassigned.discard(ri)

        for k, vid in enumerate(available_vids):
            vs = copy.deepcopy(vehicles[vid])
            if vs.battery_init is None:
                vs.battery_init = float(vs.battery)
            legs, _ = _materialize(veh_stops[k], vs, req_list, full_instance)
            routes[vid] = legs

        return routes


# ── 3. Variable Neighborhood Search ──────────────────────────────────────────

class VNSSolver:
    """
    Variable Neighborhood Search (Mladenović & Hansen 1997) for PDPTW.

    Initial solution: greedy cheapest insertion.
    Neighborhoods (applied in order, reset to N1 on improvement):
      N1 — Relocate: move one (pickup, delivery) pair to a cheaper position
           in any vehicle (same or different).
      N2 — Swap: exchange two (pickup, delivery) pairs between any two vehicles.
    Terminates when both neighborhoods fail to improve or time budget exhausted.

    NOTE on offline_vns < online_vns: In rolling-horizon dispatch, offline VNS
    sees all n_req requests per planning step (greedy init degrades with many
    simultaneous requests), while online VNS sees only revealed requests (1-5
    per step, easy to optimize). This makes offline_vns naturally score lower
    than online_vns — it is an expected artifact of VNS + rolling-horizon, not
    a bug.  A true "offline" baseline would solve the full shift once (no
    re-planning); the current offline_vns re-solves at every step with full
    information, which is a different (and harder) problem per step.
    """

    def __init__(
        self,
        time_limit_seconds: float = 8.0,
        latest_pickup_slack: float = 30.0,
        latest_delivery_slack: float = 30.0,
        ablate_urban: bool = False,
        ablate_nfz: bool = False,
        ablate_battery: bool = False,
        robust: bool = False,
        robust_restarts: int = 3,
    ) -> None:
        self.time_limit_seconds = time_limit_seconds
        self.latest_pickup_slack = latest_pickup_slack
        self.latest_delivery_slack = latest_delivery_slack
        self.ablate_urban = ablate_urban
        self.ablate_nfz = ablate_nfz
        self.ablate_battery = ablate_battery
        # VNS default fails on 3/10 hard instances with
        # dispatch_sim 'no_progress' errors. Root cause: single-pass greedy
        # init + VND on 2 neighborhoods can get stuck producing an infeasible
        # route the dispatcher can't execute, but VNS reports it as "solved".
        # robust=True tries N_RESTARTS random insertion orderings and keeps the
        # best feasible result by _total_cost. Fallback: if all restarts yield
        # infeasible, keep the default (lowest-cost) and let the dispatcher
        # decide. Default off preserves existing bench CSVs.
        self.robust = bool(robust)
        self.robust_restarts = int(robust_restarts)

    # ── internal helpers ──────────────────────────────────────────────────

    def _total_cost(self, veh_stops, veh_list, req_list, full_instance):
        return sum(
            _route_cost_energy(veh_stops[k], veh_list[k], req_list, full_instance)
            for k in range(len(veh_list))
        )

    def _feasible(self, stops, veh, req_list):
        return _check_cap(stops, veh.capacity, veh.load, req_list) and _check_prec(stops)

    def _greedy_init(self, req_list, veh_list, dms, n_depots, full_instance=None,
                     ablate_nfz=False, ablate_battery=False):
        n_veh = len(veh_list)
        veh_stops: List[List[Tuple[int, str]]] = [[] for _ in range(n_veh)]
        for ri in range(len(req_list)):
            best_k, best_p, best_d, best_delta = -1, 0, 1, 1e18
            for k, veh in enumerate(veh_list):
                dm = dms[veh.normalized_mode()]
                delta, p, d = _best_insertion(ri, veh_stops[k], veh, req_list, dm, n_depots,
                                              full_instance, ablate_nfz, ablate_battery)
                if delta < best_delta:
                    best_delta, best_k, best_p, best_d = delta, k, p, d
            if best_k < 0 or best_delta >= 1e17:
                continue  # no feasible vehicle for this request; skip
            veh_stops[best_k].insert(best_p, (ri, "pickup"))
            veh_stops[best_k].insert(best_d, (ri, "delivery"))
        return veh_stops

    def _relocate(self, veh_stops, veh_list, req_list, dms, n_depots, full_instance,
                  ablate_nfz=False, ablate_battery=False):
        """N1: try moving each request to a better position anywhere.

        full_instance is always the real instance.  ablate_nfz / ablate_battery
        control which planning constraints are visible to _best_insertion.
        _total_cost always uses full_instance for physics-accurate evaluation.
        """
        n_veh = len(veh_list)
        improved = False
        current_cost = self._total_cost(veh_stops, veh_list, req_list, full_instance)

        for src_k in range(n_veh):
            req_indices = sorted({ri for ri, _ in veh_stops[src_k]})
            for ri in req_indices:
                new_src = [(r, lt) for r, lt in veh_stops[src_k] if r != ri]
                for dst_k in range(n_veh):
                    dm = dms[veh_list[dst_k].normalized_mode()]
                    base = list(new_src) if dst_k == src_k else list(veh_stops[dst_k])
                    delta, p, d = _best_insertion(ri, base, veh_list[dst_k], req_list, dm, n_depots,
                                                  full_instance, ablate_nfz, ablate_battery)
                    if delta >= 1e17:
                        continue
                    candidate_dst = list(base)
                    candidate_dst.insert(p, (ri, "pickup"))
                    candidate_dst.insert(d, (ri, "delivery"))
                    if not self._feasible(candidate_dst, veh_list[dst_k], req_list):
                        continue

                    new_stops = [list(s) for s in veh_stops]
                    new_stops[src_k] = new_src
                    new_stops[dst_k] = candidate_dst
                    new_cost = self._total_cost(new_stops, veh_list, req_list, full_instance)
                    if new_cost < current_cost - 1e-6:
                        for k in range(n_veh):
                            veh_stops[k] = new_stops[k]
                        current_cost = new_cost
                        improved = True
                        break
                if improved:
                    break
            if improved:
                break

        return improved

    def _swap(self, veh_stops, veh_list, req_list, dms, n_depots, full_instance,
              ablate_nfz=False, ablate_battery=False):
        """N2: try swapping requests between two DIFFERENT vehicles.

        Same-vehicle swaps are skipped (relocate covers intra-route moves).
        full_instance is always the real instance.  ablate_nfz / ablate_battery
        control which planning constraints are visible to _best_insertion.
        _total_cost always uses full_instance for physics-accurate evaluation.
        """
        n_veh = len(veh_list)
        improved = False
        current_cost = self._total_cost(veh_stops, veh_list, req_list, full_instance)

        req_by_veh = [{ri for ri, _ in veh_stops[k]} for k in range(n_veh)]

        for k1 in range(n_veh):
            for ri in sorted(req_by_veh[k1]):
                for k2 in range(k1 + 1, n_veh):  # k2 > k1: inter-route only, avoids double-counting
                    for rj in sorted(req_by_veh[k2]):
                        s1 = [(r, lt) for r, lt in veh_stops[k1] if r != ri]
                        s2 = [(r, lt) for r, lt in veh_stops[k2] if r != rj]

                        dm1 = dms[veh_list[k1].normalized_mode()]
                        dm2 = dms[veh_list[k2].normalized_mode()]
                        d1, p1, dd1 = _best_insertion(rj, s1, veh_list[k1], req_list, dm1, n_depots,
                                                       full_instance, ablate_nfz, ablate_battery)
                        d2, p2, dd2 = _best_insertion(ri, s2, veh_list[k2], req_list, dm2, n_depots,
                                                       full_instance, ablate_nfz, ablate_battery)

                        if d1 >= 1e17 or d2 >= 1e17:
                            continue

                        new_s1 = list(s1)
                        new_s1.insert(p1, (rj, "pickup"))
                        new_s1.insert(dd1, (rj, "delivery"))
                        new_s2 = list(s2)
                        new_s2.insert(p2, (ri, "pickup"))
                        new_s2.insert(dd2, (ri, "delivery"))

                        if not self._feasible(new_s1, veh_list[k1], req_list):
                            continue
                        if not self._feasible(new_s2, veh_list[k2], req_list):
                            continue

                        new_stops = [list(s) for s in veh_stops]
                        new_stops[k1] = new_s1
                        new_stops[k2] = new_s2
                        new_cost = self._total_cost(new_stops, veh_list, req_list, full_instance)

                        if new_cost < current_cost - 1e-6:
                            for k in range(n_veh):
                                veh_stops[k] = new_stops[k]
                            current_cost = new_cost
                            improved = True
                            break
                    if improved:
                        break
                if improved:
                    break
            if improved:
                break

        return improved

    def solve(self, residual: Dict[str, Any]) -> Dict[int, List[Leg]]:
        routes, available_vids, veh_list, req_list, full_instance, current_time, n_depots = _setup(residual)
        vehicles = residual["vehicles"]

        if not req_list or not available_vids:
            return routes

        _ablate_nfz  = self.ablate_urban or self.ablate_nfz
        _ablate_batt = self.ablate_urban or self.ablate_battery

        dms = {
            "uav": _get_distance_matrix(full_instance, "uav"),
            "adr": _get_distance_matrix(full_instance, "adr"),
        }

        # robust=True wraps the usual greedy_init+VND in a
        # restart loop with perturbed insertion ordering. Keeps the best
        # feasible-materialized result by _total_cost. Non-robust path is
        # bit-identical to the baseline implementation.
        if self.robust:
            n_restarts = max(1, self.robust_restarts)
            per_restart_budget = max(0.5, self.time_limit_seconds / n_restarts)
            rng = random.Random(0xC0A1)   # deterministic across restarts
            best_total = float('inf')
            best_stops = None
            for restart_idx in range(n_restarts):
                # Perturbation: shuffle insertion order on restarts > 0. The
                # first restart preserves the default t_delivery-sorted order
                # so robust mode is strictly >= default on cost (never worse).
                if restart_idx == 0:
                    req_list_perm = req_list
                else:
                    perm = list(range(len(req_list)))
                    rng.shuffle(perm)
                    req_list_perm = [req_list[i] for i in perm]

                cand_stops = self._greedy_init(
                    req_list_perm, veh_list, dms, n_depots, full_instance,
                    _ablate_nfz, _ablate_batt,
                )
                # If we permuted, remap stops' indices back to original req_list
                # order so downstream _total_cost/_materialize indexing is sane.
                if restart_idx > 0:
                    inv = [0] * len(perm)
                    for new_i, old_i in enumerate(perm):
                        inv[old_i] = new_i
                    # cand_stops uses indices into req_list_perm; remap to req_list
                    # indices by inverting perm.
                    remap = {new_i: old_i for new_i, old_i in enumerate(perm)}
                    cand_stops = [[(remap[ri], lt) for ri, lt in route] for route in cand_stops]

                r_deadline = time.perf_counter() + per_restart_budget
                k_max = 2
                k = 0
                while k < k_max and time.perf_counter() < r_deadline:
                    if k == 0:
                        improved = self._relocate(cand_stops, veh_list, req_list, dms, n_depots,
                                                  full_instance, _ablate_nfz, _ablate_batt)
                    else:
                        improved = self._swap(cand_stops, veh_list, req_list, dms, n_depots,
                                              full_instance, _ablate_nfz, _ablate_batt)
                    k = 0 if improved else k + 1

                cand_total = self._total_cost(cand_stops, veh_list, req_list, full_instance)
                if cand_total < best_total:
                    best_total = cand_total
                    best_stops = cand_stops
            veh_stops = best_stops if best_stops is not None else self._greedy_init(
                req_list, veh_list, dms, n_depots, full_instance, _ablate_nfz, _ablate_batt)
        else:
            veh_stops = self._greedy_init(req_list, veh_list, dms, n_depots, full_instance,
                                          _ablate_nfz, _ablate_batt)

            deadline = time.perf_counter() + self.time_limit_seconds
            k_max = 2
            k = 0
            while k < k_max and time.perf_counter() < deadline:
                if k == 0:
                    improved = self._relocate(veh_stops, veh_list, req_list, dms, n_depots,
                                              full_instance, _ablate_nfz, _ablate_batt)
                else:
                    improved = self._swap(veh_stops, veh_list, req_list, dms, n_depots,
                                          full_instance, _ablate_nfz, _ablate_batt)
                k = 0 if improved else k + 1

        # Materialize always uses full_instance (execution ground truth)
        for k_idx, vid in enumerate(available_vids):
            vs = copy.deepcopy(vehicles[vid])
            if vs.battery_init is None:
                vs.battery_init = float(vs.battery)
            legs, _ = _materialize(veh_stops[k_idx], vs, req_list, full_instance)
            routes[vid] = legs

        return routes


# ── 4. FIFO Solver ───────────────────────────────────────────────────────────

class FIFOSolver:
    """
    FIFO dispatch: assign each waiting request (sorted by t_pickup) to the nearest
    available vehicle by distance from its current route end-node to the pickup.
    Appends pickup+delivery to the end of the chosen vehicle's route — no position
    search.  Uses ce_cpdptw_alns infrastructure so arc masks and energy are enforced.
    """

    def __init__(self, ablate_urban: bool = False, ablate_nfz: bool = False,
                 ablate_battery: bool = False, **kwargs) -> None:
        self.ablate_urban = ablate_urban
        self.ablate_nfz = ablate_nfz
        self.ablate_battery = ablate_battery

    def solve(self, residual: Dict[str, Any]) -> Dict[int, List[Leg]]:
        routes, available_vids, veh_list, req_list, full_instance, current_time, n_depots = _setup(residual)
        if not req_list or not available_vids:
            return routes

        _ablate_nfz  = self.ablate_urban or self.ablate_nfz
        _ablate_batt = self.ablate_urban or self.ablate_battery

        n_veh = len(available_vids)
        dms = {
            "uav": _get_distance_matrix(full_instance, "uav"),
            "adr": _get_distance_matrix(full_instance, "adr"),
        }

        veh_stops: List[List[Tuple[int, str]]] = [[] for _ in range(n_veh)]
        end_node: List[int] = [int(veh_list[k].current_node) for k in range(n_veh)]

        sorted_pairs = sorted(enumerate(req_list), key=lambda x: (x[1].t_pickup, x[1].req_id))

        for ri, req in sorted_pairs:
            best_k, best_cost = -1, 1e18
            for k, veh in enumerate(veh_list):
                mode = veh.normalized_mode()
                dm = dms[mode]

                if not _ablate_nfz and not _is_arc_feasible(full_instance, mode, end_node[k], req.pickup_node):
                    continue
                if not _ablate_nfz and not _is_arc_feasible(full_instance, mode, req.pickup_node, req.delivery_node):
                    continue

                candidate = veh_stops[k] + [(ri, "pickup"), (ri, "delivery")]
                if not _check_cap(candidate, veh.capacity, veh.load, req_list):
                    continue

                en = end_node[k]
                d = float(dm[en, req.pickup_node])
                if d >= 1e9:
                    continue

                batt_pen = 0.0 if _ablate_batt else _battery_penalty(veh, req.pickup_node, full_instance)
                if batt_pen >= 1e9:
                    continue

                effective_cost = d + batt_pen
                if effective_cost < best_cost:
                    best_cost, best_k = effective_cost, k

            if best_k < 0:
                continue
            veh_stops[best_k].extend([(ri, "pickup"), (ri, "delivery")])
            end_node[best_k] = req.delivery_node

        vehicles = residual["vehicles"]
        for k, vid in enumerate(available_vids):
            vs = copy.deepcopy(vehicles[vid])
            if vs.battery_init is None:
                vs.battery_init = float(vs.battery)
            legs, _ = _materialize(veh_stops[k], vs, req_list, full_instance)
            routes[vid] = legs

        return routes


# ── 5. Greedy Insertion Solver ────────────────────────────────────────────────

class GreedyInsertionSolver:
    """
    Cheapest-insertion heuristic: for each waiting request (sorted by t_delivery),
    find the (vehicle, position) pair with the smallest insertion cost delta and
    insert there.  Uses ce_cpdptw_alns infrastructure so arc masks and energy are
    enforced.  This replaces the dispatch_sim-based GreedyInsertion used in the RL
    training pipeline, which skips arc mask checks.
    """

    def __init__(self, ablate_urban: bool = False, ablate_nfz: bool = False,
                 ablate_battery: bool = False, **kwargs) -> None:
        self.ablate_urban = ablate_urban
        self.ablate_nfz = ablate_nfz
        self.ablate_battery = ablate_battery

    def solve(self, residual: Dict[str, Any]) -> Dict[int, List[Leg]]:
        routes, available_vids, veh_list, req_list, full_instance, current_time, n_depots = _setup(residual)
        if not req_list or not available_vids:
            return routes

        _ablate_nfz  = self.ablate_urban or self.ablate_nfz
        _ablate_batt = self.ablate_urban or self.ablate_battery

        n_veh = len(available_vids)
        dms = {
            "uav": _get_distance_matrix(full_instance, "uav"),
            "adr": _get_distance_matrix(full_instance, "adr"),
        }

        veh_stops: List[List[Tuple[int, str]]] = [[] for _ in range(n_veh)]

        sorted_pairs = sorted(enumerate(req_list), key=lambda x: (x[1].t_delivery, x[1].t_pickup, x[1].req_id))

        for ri, _req in sorted_pairs:
            best_cost, best_k, best_p, best_d_pos = 1e18, -1, 0, 1
            for k, veh in enumerate(veh_list):
                mode = veh.normalized_mode()
                dm = dms[mode]
                cost, p, d_pos = _best_insertion(ri, veh_stops[k], veh, req_list, dm, n_depots,
                                                 full_instance, _ablate_nfz, _ablate_batt)
                if cost < best_cost:
                    best_cost, best_k, best_p, best_d_pos = cost, k, p, d_pos

            if best_k < 0 or best_cost >= 1e17:
                continue
            veh_stops[best_k].insert(best_p, (ri, "pickup"))
            veh_stops[best_k].insert(best_d_pos, (ri, "delivery"))

        vehicles = residual["vehicles"]
        for k, vid in enumerate(available_vids):
            vs = copy.deepcopy(vehicles[vid])
            if vs.battery_init is None:
                vs.battery_init = float(vs.battery)
            legs, _ = _materialize(veh_stops[k], vs, req_list, full_instance)
            routes[vid] = legs

        return routes


# ── wrapper functions (online + offline for each solver) ──────────────────────

def _make_wrapper(solver_cls, solver_kwargs=None):
    """Return online and offline solve functions for the given solver class."""

    def _solve_rolling(
        full_instance: Dict[str, Any],
        *,
        n_uav: int,
        n_adr: int,
        n_depots_uav: int,
        n_depots_adr: int,
        depot_sharing: bool = True,
        delta_minutes: float = 5.0,
        shift_minutes: Optional[float] = None,
        **kwargs,
    ) -> List[Dict[str, Any]]:
        config = SyntheticALNSConfig(
            n_uav=n_uav, n_adr=n_adr,
            n_depots_uav=n_depots_uav, n_depots_adr=n_depots_adr,
            depot_sharing=depot_sharing,
            delta_minutes=delta_minutes,
            shift_minutes=shift_minutes,
        )
        fleet = build_initial_fleet_from_instance(full_instance, config)
        arrival_stream = build_dynamic_arrival_stream_from_instance(full_instance)
        if shift_minutes is None:
            tw = _to_numpy(full_instance["time_window"]).reshape(-1).astype(float)
            shift_minutes = float(np.nanmax(tw) + 120.0)
        kw = (solver_kwargs or {}).copy()
        kw.update({k: v for k, v in kwargs.items()
                   if k in ("time_limit_seconds", "latest_pickup_slack", "latest_delivery_slack",
                            "ablate_urban", "ablate_nfz", "ablate_battery",
                            "robust", "robust_restarts")})
        solver = solver_cls(**kw)
        dispatcher = RollingHorizonDispatcher(
            solver=solver, delta_minutes=delta_minutes, shift_minutes=shift_minutes,
        )
        return dispatcher.run_shift(arrival_stream, fleet, full_instance)

    def _solve_offline(
        full_instance: Dict[str, Any],
        *,
        n_uav: int,
        n_adr: int,
        n_depots_uav: int,
        n_depots_adr: int,
        depot_sharing: bool = True,
        delta_minutes: float = 5.0,
        shift_minutes: Optional[float] = None,
        **kwargs,
    ) -> List[Dict[str, Any]]:
        config = SyntheticALNSConfig(
            n_uav=n_uav, n_adr=n_adr,
            n_depots_uav=n_depots_uav, n_depots_adr=n_depots_adr,
            depot_sharing=depot_sharing,
            delta_minutes=delta_minutes,
            shift_minutes=shift_minutes,
        )
        fleet = build_initial_fleet_from_instance(full_instance, config)
        # Static stream: all requests at t=0 (oracle full information)
        arrival_stream = build_static_arrival_stream_from_instance(full_instance)
        if shift_minutes is None:
            tw = _to_numpy(full_instance["time_window"]).reshape(-1).astype(float)
            shift_minutes = float(np.nanmax(tw) + 120.0)
        kw = (solver_kwargs or {}).copy()
        kw.update({k: v for k, v in kwargs.items()
                   if k in ("time_limit_seconds", "latest_pickup_slack", "latest_delivery_slack",
                            "ablate_urban", "ablate_nfz", "ablate_battery",
                            "robust", "robust_restarts")})
        solver = solver_cls(**kw)
        dispatcher = RollingHorizonDispatcher(
            solver=solver, delta_minutes=delta_minutes, shift_minutes=shift_minutes,
        )
        return dispatcher.run_shift(arrival_stream, fleet, full_instance)

    return _solve_rolling, _solve_offline


(
    solve_static_instance_with_cw_rolling,
    solve_static_instance_with_cw_offline,
) = _make_wrapper(ClarkeWrightSolver)

(
    solve_static_instance_with_regret_rolling,
    solve_static_instance_with_regret_offline,
) = _make_wrapper(RegretInsertionSolver)

(
    solve_static_instance_with_vns_rolling,
    solve_static_instance_with_vns_offline,
) = _make_wrapper(VNSSolver)

(
    solve_static_instance_with_fifo_rolling,
    solve_static_instance_with_fifo_offline,
) = _make_wrapper(FIFOSolver)

(
    solve_static_instance_with_greedy_rolling,
    solve_static_instance_with_greedy_offline,
) = _make_wrapper(GreedyInsertionSolver)


# ── no-urban-constraints variants (ablation: planners ignore NFZ / ADR barrier /
#    battery-penalty; execution still enforces all constraints via _simulate_leg) ──

(
    solve_static_instance_with_fifo_nouc_rolling,
    solve_static_instance_with_fifo_nouc_offline,
) = _make_wrapper(FIFOSolver, {"ablate_urban": True})

(
    solve_static_instance_with_cw_nouc_rolling,
    solve_static_instance_with_cw_nouc_offline,
) = _make_wrapper(ClarkeWrightSolver, {"ablate_urban": True})

(
    solve_static_instance_with_greedy_nouc_rolling,
    solve_static_instance_with_greedy_nouc_offline,
) = _make_wrapper(GreedyInsertionSolver, {"ablate_urban": True})

(
    solve_static_instance_with_regret_nouc_rolling,
    solve_static_instance_with_regret_nouc_offline,
) = _make_wrapper(RegretInsertionSolver, {"ablate_urban": True})

(
    solve_static_instance_with_vns_nouc_rolling,
    solve_static_instance_with_vns_nouc_offline,
) = _make_wrapper(VNSSolver, {"ablate_urban": True})


# ── NFZ-blind / battery-aware (_nofc): isolates value of NFZ-aware routing ───
# Planner skips arc-feasibility checks but still accounts for battery detours.

(
    solve_static_instance_with_fifo_nofc_rolling,
    solve_static_instance_with_fifo_nofc_offline,
) = _make_wrapper(FIFOSolver, {"ablate_nfz": True})

(
    solve_static_instance_with_cw_nofc_rolling,
    solve_static_instance_with_cw_nofc_offline,
) = _make_wrapper(ClarkeWrightSolver, {"ablate_nfz": True})

(
    solve_static_instance_with_greedy_nofc_rolling,
    solve_static_instance_with_greedy_nofc_offline,
) = _make_wrapper(GreedyInsertionSolver, {"ablate_nfz": True})

(
    solve_static_instance_with_regret_nofc_rolling,
    solve_static_instance_with_regret_nofc_offline,
) = _make_wrapper(RegretInsertionSolver, {"ablate_nfz": True})

(
    solve_static_instance_with_vns_nofc_rolling,
    solve_static_instance_with_vns_nofc_offline,
) = _make_wrapper(VNSSolver, {"ablate_nfz": True})


# ── NFZ-aware / battery-blind (_nob): isolates value of battery-aware routing ─
# Planner enforces arc-feasibility but ignores recharge-detour penalties.

(
    solve_static_instance_with_fifo_nob_rolling,
    solve_static_instance_with_fifo_nob_offline,
) = _make_wrapper(FIFOSolver, {"ablate_battery": True})

(
    solve_static_instance_with_cw_nob_rolling,
    solve_static_instance_with_cw_nob_offline,
) = _make_wrapper(ClarkeWrightSolver, {"ablate_battery": True})

(
    solve_static_instance_with_greedy_nob_rolling,
    solve_static_instance_with_greedy_nob_offline,
) = _make_wrapper(GreedyInsertionSolver, {"ablate_battery": True})

(
    solve_static_instance_with_regret_nob_rolling,
    solve_static_instance_with_regret_nob_offline,
) = _make_wrapper(RegretInsertionSolver, {"ablate_battery": True})

(
    solve_static_instance_with_vns_nob_rolling,
    solve_static_instance_with_vns_nob_offline,
) = _make_wrapper(VNSSolver, {"ablate_battery": True})
