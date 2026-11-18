from __future__ import annotations
import math
import copy
from typing import Dict, List, Tuple, Optional, Any

from dispatch_sim import Request, Vehicle, Leg
from reward import request_penalty, operating_cost_per_minute, battery_penalty
from reward import objective_from_instance, accept_service_insertion
from energy import coord_edge_energy_kj
from vrpUpdate import (
    V_UAV_MAX,
    V_ADR_MAX,
    V_UAV_DEPOT,
    V_ADR_DEPOT,
    V_UAV_MIN_PICKUP,
    V_ADR_MIN_PICKUP,
    UAV_LAND_TAKEOFF_MIN,
)

def _get_dist(full_inst: dict, mode: str):
    from ce_cpdptw_alns import _get_distance_matrix
    try:
        return _get_distance_matrix(full_inst, mode)
    except KeyError:
        return None

def _wind_vec(full_inst: Dict[str, Any]):
    import numpy as np
    wind = full_inst.get('wind')
    if wind is None:
        return np.zeros(2)
    if hasattr(wind, 'cpu'):
        wind = wind.cpu().numpy()
    wind_mag, wind_dir = float(wind[0]), float(wind[1])
    return np.array([wind_mag * math.cos(wind_dir), wind_mag * math.sin(wind_dir)])

def _service_time(mode: str) -> float:
    return UAV_LAND_TAKEOFF_MIN if mode == 'uav' else 0.0

def _pickup_speed(distance: float, t_now: float, t_target: float, mode: str) -> float:
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

def _depot_speed(mode: str) -> float:
    return V_UAV_DEPOT if mode == 'uav' else V_ADR_DEPOT

def _cruise_speed(mode: str) -> float:
    return V_UAV_MAX if mode == 'uav' else V_ADR_MAX

def _battery_threshold(vehicle: Vehicle) -> float:
    base = vehicle.battery_init if vehicle.battery_init is not None else vehicle.battery
    frac = 0.25 if vehicle.mode == 'uav' else 0.20
    return frac * float(base)

def _nearest_feasible_depot(from_node: int, dm, n_depots: int) -> Optional[int]:
    best_node = None
    best_d = float('inf')
    for dep in range(n_depots):
        d = float(dm[from_node][dep])
        if d < best_d and d < 1e9:
            best_d = d
            best_node = dep
    return best_node

def _energy_kj(mode: str, distance: float, payload: float, wind_vec, speed: float,
               heading_rad: float = 0.0) -> float:
    return coord_edge_energy_kj(mode, distance, payload, wind_vec, speed, heading_rad)

def _simulate_leg(
    vehicle_state: Vehicle,
    req: Optional[Request],
    leg_type: str,
    to_node: int,
    full_inst: dict,
) -> Tuple[Optional[Leg], Vehicle, bool]:
    from ce_cpdptw_alns import _simulate_leg as simulate
    return simulate(vehicle_state, req, leg_type, to_node, full_inst)


def _materialize_ops(
    vehicle: Vehicle,
    ops: List[Tuple[int, str]],
    requests: Dict[int, Request],
    full_inst: dict,
) -> Tuple[List[Leg], float, bool]:
    from ce_cpdptw_alns import _materialize_ops as materialize
    return materialize(vehicle, ops, requests, full_inst)

def _route_ops_from_vehicle(vehicle: Vehicle, active_requests: Dict[int, Request]) -> List[Tuple[int, str]]:
    from ce_cpdptw_alns import _route_ops_from_vehicle as route_ops
    return route_ops(vehicle, active_requests)

def _insertion_cost(
    vehicle: Vehicle,
    ops: List[Tuple[int, str]],
    req: Request,
    insert_pos: int,
    active_requests: Dict[int, Request],
    full_inst: dict,
) -> float:
    candidate_ops = list(ops)
    candidate_ops[insert_pos:insert_pos] = [(req.req_id, 'pickup'), (req.req_id, 'delivery')]

    _, candidate_cost, ok = _materialize_ops(vehicle, candidate_ops, active_requests, full_inst)
    if not ok:
        return float('inf')

    _, base_cost, ok_base = _materialize_ops(vehicle, ops, active_requests, full_inst)
    if not ok_base:
        base_cost = 0.0

    return candidate_cost - base_cost

class GreedyInsertion:

    def solve_ops(self, residual: dict) -> Dict[int, List[Tuple[int, str]]]:
        vehicles = residual['vehicles']
        active_requests = residual['active_requests']

        ops = {vid: _route_ops_from_vehicle(veh, active_requests) for vid, veh in vehicles.items()}

        to_insert = [
            r for r in active_requests.values()
            if r.status == 'waiting_for_pickup'
        ]
        to_insert.sort(key=lambda r: (r.t_delivery, r.t_pickup, r.req_id))

        for req in to_insert:
            best_cost = float('inf')
            best_vid = None
            best_pos = None

            for vid, veh in vehicles.items():
                if veh.committed_leg is not None:
                    continue
                vehicle_ops = ops[vid]
                for pos in range(len(vehicle_ops) + 1):
                    from coalition_contract import ops_permitted
                    candidate = list(vehicle_ops)
                    candidate[pos:pos] = [(req.req_id, 'pickup'), (req.req_id, 'delivery')]
                    fleet_ops = dict(ops)
                    fleet_ops[vid] = candidate
                    if not ops_permitted(fleet_ops, vehicles, active_requests, residual['full_instance']):
                        continue
                    cost = _insertion_cost(
                        veh, vehicle_ops, req, pos, active_requests, residual['full_instance']
                    )
                    if cost < best_cost:
                        best_cost = cost
                        best_vid = vid
                        best_pos = pos

            if best_vid is not None and accept_service_insertion(best_cost, residual['full_instance']):
                ops[best_vid][best_pos:best_pos] = [(req.req_id, 'pickup'), (req.req_id, 'delivery')]

        return ops

    def solve(self, residual: dict) -> Dict[int, List[Leg]]:
        vehicles = residual['vehicles']
        active_requests = residual['active_requests']
        full_inst = residual['full_instance']

        ops = self.solve_ops(residual)
        routes: Dict[int, List[Leg]] = {}
        for vid, veh in vehicles.items():
            legs, _, ok = _materialize_ops(veh, ops.get(vid, []), active_requests, full_inst)
            routes[vid] = legs if ok else []
        return routes
