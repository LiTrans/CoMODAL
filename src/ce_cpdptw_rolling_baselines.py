"""Rolling adapters for the common bounded-load formulation.

Residual deferral is a lexicographic planner rule, not optional episode
service and not a finite coalition value for an incomplete book.
"""
from __future__ import annotations
import copy
import importlib
import math
from dataclasses import dataclass
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple
import numpy as np
from ce_cpdptw_alns import (
    ALNSBaseline, Leg, Request, RollingHorizonDispatcher, SyntheticALNSConfig, Vehicle,
    _battery_threshold, _full_battery as _shared_full_battery, _cruise_speed,
    _depot_speed, _edge_energy_kj, _get_distance_matrix, _is_arc_feasible,
    _nearest_feasible_depot, _simulate_leg, _to_numpy, _wind_vec,
    build_initial_fleet_from_instance, build_dynamic_arrival_stream_from_instance,
    remaining_recharges, solve_static_instance_with_alns, summarize_episode_log,
)
from energy import arc_heading
from reward import mandatory_objective_weights, pickup_penalty, delivery_penalty

@dataclass(frozen=True)
class ExactRollingConfig:
    n_uav: int
    n_adr: int
    n_depots_uav: int
    n_depots_adr: int
    depot_sharing: bool = True
    depot_visit_limit: int = 4
    delta_minutes: float = 5.
    shift_minutes: Optional[float] = None
    latest_pickup_slack: float = 0.
    latest_delivery_slack: float = 0.
    use_discrete_pickup_speed: bool = False
    pickup_speed_grid_size: int = 5
    time_limit_seconds: float = 3600.
    mip_gap: float = .01
    log_to_console: bool = False
    n_threads: int = 0
    disable_battery: bool = False
    gurobi_env: object = None
    undelivered_penalty: float = 0.
    energy_load_intervals: int = 16
    epsilon_shared: float = 1.
    epsilon_to_adr: float = 1.
    epsilon_to_uav: float = 1.
    # Planning-reserve fractions (UAV 25% / ADR 20% per Section 5).
    reserve_frac_uav: float = 0.25
    reserve_frac_adr: float = 0.20
    # Capacity sweep knobs (kJ); None = model_config defaults.
    capacity_kj_uav: Optional[float] = None
    capacity_kj_adr: Optional[float] = None
    max_recharges_per_vehicle: Optional[int] = None
    recharge_cost_includes_dwell: bool = True
    # Dead-code: pickup-TW lookahead filter retained for CLI compatibility only.
    pickup_tw_lookahead_buffer: float = 10.

def _node_name(index):
    return f"n{int(index)}"

def _node_index(name, origin_node_by_name):
    return int(name[1:]) if name.startswith("n") else int(origin_node_by_name[name])

def _vehicle_mode_for_exact(vehicle):
    return vehicle.normalized_mode().upper()

def _full_battery(vehicle):
    return _shared_full_battery(vehicle)

def _make_forced_delivery_or_depot_route(vehicle, active_requests, full_instance):
    """Legacy heuristic helper, never substituted for native MILP loaded state."""
    if vehicle.committed_leg is not None:
        return [vehicle.committed_leg]
    if vehicle.onboard_set():
        route, state = [], copy.deepcopy(vehicle)
        for rid in sorted(vehicle.onboard_set()):
            req = active_requests.get(rid)
            if req is None:
                raise ValueError("Onboard cargo missing from active requests")
            leg, following, ok = _simulate_leg(state, req, "delivery", req.delivery_node, full_instance)
            if not ok:
                depot = _nearest_feasible_depot(state.current_node, state, full_instance)
                if depot is None:
                    return route
                recharge, state, good = _simulate_leg(state, None, "depot", depot, full_instance)
                if not good or recharge is None:
                    return route
                route.append(recharge)
                leg, following, ok = _simulate_leg(state, req, "delivery", req.delivery_node, full_instance)
            if not ok or leg is None:
                return route
            route.append(leg)
            state = following
        return route
    if vehicle.battery < _battery_threshold(vehicle):
        depot = _nearest_feasible_depot(vehicle.current_node, vehicle, full_instance)
        if depot is not None:
            leg, _, ok = _simulate_leg(vehicle, None, "depot", depot, full_instance)
            return [leg] if ok and leg is not None else []
        return []
    return None

def _build_exact_data_from_residual(*, module, residual, config, available_vehicle_ids,
                                    service_policy="residual_defer"):
    full = residual["full_instance"]
    active, vehicles = residual["active_requests"], residual["vehicles"]
    if full.get("physical_config") is not None:
        raise NotImplementedError(
            "Configured reserve, depot phases and terminal/safe-continuation rules are not yet "
            "encoded in this customer-phase MILP; no scientific comparator is returned")
    ndep = int(full["n_depots"])
    if ndep != config.n_depots_uav + config.n_depots_adr:
        raise ValueError("Depot split mismatch")
    if config.disable_battery:
        raise NotImplementedError("Battery-disabled planner is an uncertified diagnostic, not this common baseline")
    if config.undelivered_penalty or config.latest_pickup_slack or config.latest_delivery_slack:
        raise ValueError("Common objective uses raw targets, no missed price")
    if config.use_discrete_pickup_speed:
        raise NotImplementedError("Discrete speed search not common scalar-policy certified")
    weights = mandatory_objective_weights(full.get("objective_weights"))
    contract, ledger = full.get("sharing_contract"), dict(full.get("sharing_assignments", {}))
    exact_to_real = {f"v{vid}": vid for vid in available_vehicle_ids}
    if any(vehicles[vid].committed_leg is not None for vid in available_vehicle_ids):
        raise ValueError("Busy/locked vehicles must not enter residual replanning")
    waiting = [r for r in active.values() if r.status == "waiting_for_pickup"
               and (r.assigned_vehicle is None or r.assigned_vehicle in available_vehicle_ids)]
    onboard = [r for r in active.values() if r.status == "onboard" and r.assigned_vehicle in available_vehicle_ids]
    # pickup_tw_lookahead_buffer is retained for CLI compat but currently unused.
    requests, actions = [], {}
    for r in [*waiting, *onboard]:
        owner, group, permissions = r.firm_owner, r.market_group, tuple(r.allowed_modes)
        if contract is not None:
            if (r.req_id not in contract.owners or owner != contract.owners[r.req_id]
                    or group != contract.groups[r.req_id] or permissions != tuple(contract.permissions[r.req_id])):
                raise ValueError("Request metadata differs from persistent sharing contract")
            permissions = tuple(m for m in permissions if m in contract.members)
        requests.append(module.Request(id=f"r{r.req_id}", pickup=_node_name(r.pickup_node),
            delivery=_node_name(r.delivery_node), demand=float(r.demand), pickup_target=float(r.t_pickup),
            pickup_latest=float(r.t_pickup), delivery_target=float(r.t_delivery), delivery_latest=float(r.t_delivery),
            market=group.upper() if group != "shared" else "shared", owner_mode=owner.upper() if owner else None,
            allowed_modes=tuple(m.upper() for m in permissions), release_time=float(r.t_arrival),
            picked_by=f"v{r.assigned_vehicle}" if r.status == "onboard" else None,
            assigned_to=f"v{r.assigned_vehicle}" if r.assigned_vehicle is not None else None))
        if r.status == "waiting_for_pickup":
            actions[_node_name(r.pickup_node)] = (r.req_id, "pickup")
        actions[_node_name(r.delivery_node)] = (r.req_id, "delivery")
    exact_vehicles, origins, origin_nodes = [], {}, {}
    recharge_context = _augment_full_instance(full, config)
    for k, vid in exact_to_real.items():
        v = vehicles[vid]
        known = {r.req_id for r in onboard if r.assigned_vehicle == vid}
        if known != v.onboard_set():
            raise ValueError("Onboard set and active delivery obligations disagree")
        origin = f"origin_v{vid}"
        origins[k], origin_nodes[origin] = origin, int(v.current_node)
        _mode = v.normalized_mode()
        _cap_override = getattr(config, 'capacity_kj_uav' if _mode == 'uav' else 'capacity_kj_adr', None)
        full_pack = float(_cap_override) if _cap_override is not None else _full_battery(v)
        _reserve_frac = float(getattr(config, 'reserve_frac_uav' if _mode == 'uav' else 'reserve_frac_adr'))
        copies = int(config.depot_visit_limit)
        remaining = remaining_recharges(v, recharge_context)
        if remaining is not None:
            copies = min(copies, remaining)
        exact_vehicles.append(module.Vehicle(k, _vehicle_mode_for_exact(v), float(v.capacity), full_pack,
            _reserve_frac * full_pack, copies,
            initial_battery=float(v.battery), initial_time=float(v.current_time), initial_load=float(v.load),
            start_node=origin, allowed_depots=tuple(_node_name(d) for d in (
                v.allowed_depots if v.allowed_depots is not None else range(ndep)))))
    depots = [_node_name(i) for i in range(ndep)]
    names = list(dict.fromkeys([*depots, *origin_nodes, *actions]))
    times, features, physics = {}, {}, {}
    for ev in exact_vehicles:
        mode = ev.mode.lower()
        dm = _get_distance_matrix(full, mode)
        for source in names:
            for target in names:
                if source == target or target in origin_nodes:
                    continue
                i, j = _node_index(source, origin_nodes), _node_index(target, origin_nodes)
                if not _is_arc_feasible(full, mode, i, j):
                    continue
                if j < ndep and target not in ev.allowed_depots:
                    continue
                speed = _depot_speed(mode) if j < ndep else _cruise_speed(mode)
                distance, heading = float(dm[i, j]), arc_heading(full, i, j)
                q = (ev.id, source, target)
                physics[q] = module.ArcPhysics(distance, speed, tuple(_wind_vec(full)), heading)
                times[q] = distance / speed
                # Empty-load graph features are retained for old inspectors only.
                # Model coefficients are derived from ArcPhysics and actual load.
                features[q] = _edge_energy_kj(mode=mode, distance=distance, payload=0.,
                    wind_vec=_wind_vec(full), speed=speed, heading_rad=heading)
    remaining = counts = None
    if contract is not None:
        usage, budgets = contract.usage(ledger), contract.budgets()
        remaining = {g: budgets[g] - usage[g] for g in budgets}
        if any(n < 0 for n in remaining.values()):
            raise ValueError("Existing sharing ledger exceeds whole-book budget")
        for rid, mode in ledger.items():
            if mode not in contract.members or mode not in contract.permissions[rid]:
                raise ValueError("Existing ledger violates physical/coalition permission")
        counts = {g: sum(group == g for group in contract.groups.values()) for g in budgets}
        fractions = (contract.eps_shared, contract.eps_uav_to_adr, contract.eps_adr_to_uav)
    else:
        fractions = (config.epsilon_shared, config.epsilon_to_adr, config.epsilon_to_uav)
        if service_policy == "residual_defer" and min(fractions) < 1:
            raise ValueError("Residual sharing quotas require a persistent whole-book SharingContract")
    data = module.CECPDPTWData(requests=requests, vehicles=exact_vehicles,
        physical_depots=depots, uav_depots=depots[:config.n_depots_uav], adr_depots=depots[config.n_depots_uav:],
        travel_time=times, energy=features, arc_physics=physics, depot_sharing=config.depot_sharing,
        alpha_uav=weights.operating_uav, alpha_adr=weights.operating_adr,
        alpha_early_pickup=weights.pickup_early, alpha_late_pickup=weights.pickup_late,
        alpha_late_delivery=weights.delivery_late, epsilon_shared=fractions[0],
        epsilon_to_adr=fractions[1], epsilon_to_uav=fractions[2],
        sharing_remaining=remaining, sharing_book_counts=counts,
        sharing_assignments={f"r{rid}": mode for rid, mode in ledger.items()},
        service_policy=service_policy, energy_load_intervals=config.energy_load_intervals,
        recharge_cost_includes_dwell=bool(getattr(config, 'recharge_cost_includes_dwell', True)),
        allow_delayed_departure=bool(full.get("allow_delayed_departure", False)))
    return data, exact_to_real, origins, origin_nodes, actions

def _dump_replay_rejection(*, state, req, role, target_name, target_idx, full,
                            vehicle_key, current_copy, target_copy, built, value_fn):
    """Emit a detailed breakdown of why _simulate_leg rejected the MIP's arc.
    Called only when env var CE_DEBUG_REPLAY=1."""
    import sys as _sys
    from ce_cpdptw_alns import (_physical_config, _full_battery as _fb,
        _is_arc_feasible, _get_distance_matrix)
    mode = state.normalized_mode()
    n_depots = int(full['n_depots'])
    cfg = _physical_config(full)
    full_battery = _fb(state, full)
    if cfg is not None:
        reserve = cfg.battery_for(mode).reserve_energy_kj
    else:
        _frac = full.get('reserve_frac_by_mode', {}).get(mode, 0.)
        reserve = float(_frac) * full_battery
    print('\n' + '=' * 72, file=_sys.stderr)
    print(f'[replay-reject] vehicle={vehicle_key} mode={mode.upper()}', file=_sys.stderr)
    print(f'  role={role}  target_name={target_name}  target_idx={target_idx}', file=_sys.stderr)
    print(f'  from current_copy={current_copy}  to target_copy={target_copy}', file=_sys.stderr)
    print(f'  current_node={state.current_node}  t={state.current_time:.3f}  '
          f'battery={state.battery:.3f}  load={state.load:.3f}', file=_sys.stderr)
    print(f'  full_battery={full_battery:.3f}  reserve={reserve:.3f}  '
          f'capacity={state.capacity:.3f}', file=_sys.stderr)
    print(f'  reserve_margin (battery - reserve) = {state.battery - reserve:+.3f} kJ', file=_sys.stderr)
    # Per-check replay
    is_depot_leg = target_idx < n_depots
    print(f'  arc_feasible(mode={mode}, {state.current_node}->{target_idx}): '
          f'{_is_arc_feasible(full, mode, int(state.current_node), target_idx)}', file=_sys.stderr)
    dm = _get_distance_matrix(full, mode)
    try:
        dist = float(dm[int(state.current_node), target_idx])
        print(f'  distance={dist:.3f}', file=_sys.stderr)
    except Exception as _e:
        print(f'  distance= ERROR {_e}', file=_sys.stderr)
    print(f'  is_depot_leg={is_depot_leg}  req is None: {req is None}', file=_sys.stderr)
    if is_depot_leg:
        print(f'  depot cap? max_recharges_per_vehicle='
              f'{full.get("max_recharges_per_vehicle")}  '
              f'state.recharge_count={getattr(state, "recharge_count", "?")}', file=_sys.stderr)
        print(f'  allowed_depots={state.allowed_depots}  in_allowed={state.allowed_depots is None or target_idx in state.allowed_depots}', file=_sys.stderr)
    # MIP's own model values for this arc
    try:
        model_arrival = value_fn(built.arrival[vehicle_key, target_copy])
        model_complete = value_fn(built.completion[vehicle_key, target_copy])
        print(f'  MIP expected arrival={model_arrival:.3f}  completion={model_complete:.3f}', file=_sys.stderr)
    except Exception:
        pass
    try:
        _e = built.arc_energy[vehicle_key, current_copy, target_copy]
        model_energy = float(_e) if isinstance(_e, (int, float)) else (
            _e.solution_value() if hasattr(_e, 'solution_value') else _e.getValue())
        print(f'  MIP expected arc energy={model_energy:.3f} kJ   '
              f'battery_after_arc_mip={state.battery - model_energy:.3f}  '
              f'vs reserve={reserve:.3f}', file=_sys.stderr)
    except Exception:
        pass
    print('=' * 72, file=_sys.stderr)


def _patch_ortools_origin_starts(built, origins):
    for k, origin in origins.items():
        if built.depot_meta[built.depot_copies_by_vehicle[k][0]].physical_depot_id != origin:
            raise ValueError("Common builder did not preserve actual origin")

_patch_gurobi_origin_starts = _patch_ortools_origin_starts

def _extract_exact_routes_to_legs(*, built, residual, exact_to_real_vid, origin_node_by_name,
                                 node_to_action, value_fn):
    successors = {(k, i): j for k, i, j in built.arcs if value_fn(built.x[k, i, j]) >= .5}
    full = dict(residual["full_instance"], service_policy="mandatory")
    assignments = dict(full.get("sharing_assignments", {}))
    full["sharing_assignments"] = assignments
    contract, active = full.get("sharing_contract"), residual["active_requests"]
    weights = mandatory_objective_weights(full.get("objective_weights"))
    full["objective_weights"] = weights
    routes, replay_cost, delivered = {}, 0., set()
    for k, vid in exact_to_real_vid.items():
        state = copy.deepcopy(residual["vehicles"][vid])
        current, seen, legs = built.depot_copies_by_vehicle[k][0], set(), []
        if k in built.cycle_departure:
            state.current_time = max(state.current_time, value_fn(built.cycle_departure[k]))
        while (k, current) in successors:
            if current in seen:
                raise RuntimeError("Cycle in exact route extraction")
            seen.add(current)
            target = successors[k, current]
            if target in built.depot_meta and built.depot_meta[target].role == "end":
                if state.onboard_set():
                    raise RuntimeError("Exact route terminates with undelivered cargo")
                break
            name = built.depot_meta[target].physical_depot_id if target in built.depot_meta else target
            if name in node_to_action:
                rid, role = node_to_action[name]
                req = active[rid]
            else:
                rid, role, req = -1, "depot", None
            target_idx = _node_index(name, origin_node_by_name)
            leg, following, ok = _simulate_leg(state, req, role, target_idx, full)
            if not ok or leg is None:
                import os as _os
                if _os.environ.get('CE_DEBUG_REPLAY'):
                    _dump_replay_rejection(state=state, req=req, role=role,
                        target_name=name, target_idx=target_idx, full=full,
                        vehicle_key=k, current_copy=current, target_copy=target,
                        built=built, value_fn=value_fn)
                raise RuntimeError("Common-model replay rejected exact route; no silent prefix truncation")
            model_arrival, model_complete = value_fn(built.arrival[k, target]), value_fn(built.completion[k, target])
            # abs_tol=1e-3 min covers Gurobi LP drift through the arrival-chain constraints.
            if not (math.isclose(leg.t_arrive_raw, model_arrival, rel_tol=1e-6, abs_tol=1e-3)
                    and math.isclose(leg.t_arrive, model_complete, rel_tol=1e-6, abs_tol=1e-3)):
                import os as _os
                if _os.environ.get('CE_DEBUG_REPLAY'):
                    import sys as _sys
                    _sys.stderr.write(
                        f'\n[CE_DEBUG_REPLAY timing-mismatch] vid={k} rid={rid} role={role} '
                        f'target={name}\n'
                        f'  sim.t_arrive_raw = {leg.t_arrive_raw!r}\n'
                        f'  mip.arrival      = {model_arrival!r}\n'
                        f'  delta_raw        = {leg.t_arrive_raw - model_arrival!r}\n'
                        f'  sim.t_arrive     = {leg.t_arrive!r}\n'
                        f'  mip.completion   = {model_complete!r}\n'
                        f'  delta_complete   = {leg.t_arrive - model_complete!r}\n'
                        f'  sim.travel_time  = {getattr(leg, "travel_time", None)!r}\n'
                        f'  sim.service_time = {getattr(leg, "service_time", None)!r}\n'
                        f'  sim.operating    = {leg.operating_cost!r}\n'
                    )
                    _sys.stderr.flush()
                raise RuntimeError("Exact/scalar timing mismatch")
            energy_expr = built.arc_energy[k, current, target]
            model_energy = float(energy_expr) if isinstance(energy_expr, (int, float)) else (
                energy_expr.solution_value() if hasattr(energy_expr, "solution_value") else energy_expr.getValue())
            if leg.energy_used > model_energy + 1e-5:
                raise RuntimeError("Analytic energy enclosure violated by scalar replay")
            if role == "pickup":
                if contract is not None and not contract.reserve(rid, state.normalized_mode(), assignments):
                    raise RuntimeError("Native sharing constraints and replay ledger disagree")
                replay_cost += pickup_penalty(leg.t_arrive_raw, req.t_pickup, weights)
            elif role == "delivery":
                if rid in delivered:
                    raise RuntimeError("Duplicated exact delivery")
                delivered.add(rid)
                replay_cost += delivery_penalty(leg.t_arrive_raw, req.t_delivery, weights)
            replay_cost += leg.operating_cost
            legs.append(leg)
            state, current = following, target
        routes[vid] = legs
    selected = {int(r.id[1:]) for r in built.data.requests if value_fn(built.y_served[r.id]) >= .5}
    if selected != delivered:
        raise RuntimeError("Selected request set was not completely replayed")
    objective = built.scientific_objective
    model_cost = objective.solution_value() if hasattr(objective, "solution_value") else objective.getValue()
    # abs_tol=1e-3 matches the timing-check tolerance for accumulated LP drift.
    if not math.isclose(replay_cost, model_cost, rel_tol=1e-6, abs_tol=1e-3):
        import os as _os
        if _os.environ.get('CE_DEBUG_REPLAY'):
            import sys as _sys
            # Summarize per-vehicle replayed legs.
            _sys.stderr.write(
                f'\n[CE_DEBUG_REPLAY five-cost-disagree]\n'
                f'  replay_cost = {replay_cost!r}\n'
                f'  model_cost  = {model_cost!r}\n'
                f'  delta       = {replay_cost - model_cost!r}\n'
                f'  rel_tol=1e-6  abs_tol=1e-4\n'
            )
            for _vid, _legs in routes.items():
                _op = sum(float(_lg.operating_cost) for _lg in _legs)
                _sys.stderr.write(f'  vehicle {_vid}: n_legs={len(_legs)}  sum(operating_cost)={_op!r}\n')
                for _i, _lg in enumerate(_legs):
                    _sys.stderr.write(
                        f'    leg[{_i}] role={getattr(_lg, "leg_type", "?")} '
                        f'rid={getattr(_lg, "request_id", "?")} '
                        f'operating={_lg.operating_cost!r} '
                        f't_arrive_raw={getattr(_lg, "t_arrive_raw", None)!r}\n'
                    )
            # MIP's per-term objective breakdown is harder to fish out
            # generically; dump its value and leave term decomposition for
            # a follow-up targeted probe.
            _sys.stderr.flush()
        raise RuntimeError("Five-cost objective and scalar replay disagree")
    return routes

class ORToolsRollingHorizonSolver:
    def __init__(self, config, solver_id="SCIP"):
        self.config, self.solver_id = config, solver_id
        self.last_report = None

    def solve(self, residual):
        module = importlib.import_module("ce_cpdptw_ortools")
        available = [vid for vid, v in residual["vehicles"].items() if v.committed_leg is None]
        routes = {vid: [] for vid in residual["vehicles"]}
        if not available:
            return routes
        data, ids, origins, origin_nodes, actions = _build_exact_data_from_residual(
            module=module, residual=residual, config=self.config, available_vehicle_ids=available)
        if not data.requests:
            return routes
        built = module.solve_ce_cpdptw_ortools(data, solver_id=self.solver_id,
            time_limit_seconds=self.config.time_limit_seconds, mip_gap=self.config.mip_gap,
            log_to_console=self.config.log_to_console)
        self.last_report = module.extract_solution(built)
        if self.last_report["objective"] is None:
            # Distinguish true infeasibility from solver time-out. The author's
            # original message intentionally reads "not proof of physical
            # infeasibility" because objective=None can mean either. For bench
            # use we need to not kill the whole dispatcher episode when a
            # single cycle times out without a feasible incumbent — on that
            # path we skip this replan cycle (empty routes) so the rolling
            # dispatcher continues; the empty cycle shows up as unserved
            # requests in the final metrics, which is the honest result.
            from ortools.linear_solver import pywraplp as _pwl
            _status = self.last_report.get("status")
            if _status == _pwl.Solver.INFEASIBLE:
                raise RuntimeError("Common bounded MILP certified INFEASIBLE; physical infeasibility certificate")
            # Time-out / NOT_SOLVED / ABNORMAL: no plan this cycle.
            return routes
        routes.update(_extract_exact_routes_to_legs(built=built, residual=residual,
            exact_to_real_vid=ids, origin_node_by_name=origin_nodes, node_to_action=actions,
            value_fn=lambda var: float(var.solution_value())))
        self.last_report["scalar_replay_passed"] = True
        return routes

class GurobiRollingHorizonSolver(ORToolsRollingHorizonSolver):
    def solve(self, residual):
        module = importlib.import_module("ce_cpdptw_merged_gurobi")
        available = [vid for vid, v in residual["vehicles"].items() if v.committed_leg is None]
        routes = {vid: [] for vid in residual["vehicles"]}
        if not available:
            return routes
        data, ids, origins, origin_nodes, actions = _build_exact_data_from_residual(
            module=module, residual=residual, config=self.config, available_vehicle_ids=available)
        if not data.requests:
            return routes
        built = module.solve_ce_cpdptw(data, time_limit_seconds=self.config.time_limit_seconds,
            mip_gap=self.config.mip_gap, log_to_console=self.config.log_to_console,
            n_threads=self.config.n_threads, gurobi_env=self.config.gurobi_env)
        self.last_report = module.extract_solution(built)
        if self.last_report["objective"] is None:
            # Parallel of the ortools path above: raise on certified INFEASIBLE,
            # degrade to empty routes on time-limit / no-solution-found so a
            # single hard cycle does not nuke the full rolling episode.
            from gurobipy import GRB as _GRB
            _status = self.last_report.get("status")
            if _status == _GRB.INFEASIBLE:
                raise RuntimeError("Common Gurobi MILP certified INFEASIBLE; physical infeasibility certificate")
            # TIME_LIMIT / INTERRUPTED / etc. with SolCount=0: no plan this cycle.
            return routes
        routes.update(_extract_exact_routes_to_legs(built=built, residual=residual,
            exact_to_real_vid=ids, origin_node_by_name=origin_nodes, node_to_action=actions,
            value_fn=lambda var: float(var.X)))
        self.last_report["scalar_replay_passed"] = True
        return routes

def _infer_shift_minutes(full_instance, extra_minutes=0.):
    # This is an arrival generation window, never a terminal service deadline.
    return float(full_instance.get("generation_window_minutes", 120.))

def _augment_full_instance(full_instance, config):
    """Thread Variant-3 + C1 + C2 knobs onto full_instance for the simulator.
    _simulate_leg reads reserve_frac_by_mode, max_recharges_per_vehicle, and
    recharge_cost_includes_dwell from this dict so ALNS/VNS/heuristics/RL
    share the same budget as the MIP."""
    out = dict(full_instance)
    out['reserve_frac_by_mode'] = {
        'uav': float(getattr(config, 'reserve_frac_uav', 0.25)),
        'adr': float(getattr(config, 'reserve_frac_adr', 0.20)),
    }
    _cap = getattr(config, 'max_recharges_per_vehicle', None)
    if _cap is None:
        _cap = out.get('max_recharges_per_vehicle')
    if _cap is None:
        _cap = getattr(config, 'depot_visit_limit', None)
    if _cap is not None:
        out['max_recharges_per_vehicle'] = int(_cap)
    out['recharge_cost_includes_dwell'] = bool(getattr(config, 'recharge_cost_includes_dwell', True))
    return out


def _run_rolling(full_instance, config, solver):
    full_instance = _augment_full_instance(full_instance, config)
    fleet = build_initial_fleet_from_instance(full_instance, SyntheticALNSConfig(
        config.n_uav, config.n_adr, config.n_depots_uav, config.n_depots_adr,
        depot_sharing=config.depot_sharing, delta_minutes=config.delta_minutes,
        shift_minutes=config.shift_minutes))
    stream = build_dynamic_arrival_stream_from_instance(full_instance)
    dispatcher = RollingHorizonDispatcher(solver, config.delta_minutes,
        config.shift_minutes if config.shift_minutes is not None else _infer_shift_minutes(full_instance),
        service_policy="mandatory")
    return dispatcher.run_shift(stream, fleet, full_instance)

def solve_static_instance_with_gurobi_rolling(full_instance, config):
    return _run_rolling(full_instance, config, GurobiRollingHorizonSolver(config))

def solve_static_instance_with_ortools_rolling(full_instance, config, *, solver_id="SCIP"):
    return _run_rolling(full_instance, config, ORToolsRollingHorizonSolver(config, solver_id))

def solve_static_instance_with_gurobi_no_battery_rolling(full_instance, config):
    raise NotImplementedError("Historical no-battery planner is not a common-model exact comparator")

def solve_static_instance_with_all_rolling_baselines(full_instance, *, n_uav, n_adr,
        n_depots_uav, n_depots_adr, depot_sharing=True, delta_minutes=5., shift_minutes=None,
        exact_time_limit_seconds=3600., exact_mip_gap=.01, ortools_solver_id="SCIP",
        alns_seed=42, alns_time_budget_small_s=3., alns_time_budget_large_s=8.):
    cfg = ExactRollingConfig(n_uav, n_adr, n_depots_uav, n_depots_adr, depot_sharing=depot_sharing,
        delta_minutes=delta_minutes, shift_minutes=shift_minutes, time_limit_seconds=exact_time_limit_seconds,
        mip_gap=exact_mip_gap)
    alns_cfg = SyntheticALNSConfig(n_uav, n_adr, n_depots_uav, n_depots_adr,
        depot_sharing=depot_sharing, delta_minutes=delta_minutes, shift_minutes=shift_minutes)
    calls = dict(alns=lambda: solve_static_instance_with_alns(full_instance, alns_cfg,
        solver=ALNSBaseline(seed=alns_seed, time_budget_small_s=alns_time_budget_small_s,
                            time_budget_large_s=alns_time_budget_large_s)),
        gurobi=lambda: solve_static_instance_with_gurobi_rolling(full_instance, cfg),
        ortools=lambda: solve_static_instance_with_ortools_rolling(full_instance, cfg, solver_id=ortools_solver_id))
    report = {}
    for name, call in calls.items():
        try:
            report[name] = dict(available=True, summary=summarize_episode_log(call()))
        except Exception as exc:
            report[name] = dict(available=False, error=f"{type(exc).__name__}: {exc}")
    return report
