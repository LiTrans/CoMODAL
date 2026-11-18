"""Owned task books, mode permissions and persistent cross-service budgets.

Budgets are fixed contractual whole-book counts, not redrawn at each epoch.
Only a committed pickup reserves one assignment; routing candidates use copies.
This is an executable feasibility contract, not an optimal coalition valuation.
"""
from dataclasses import dataclass
import copy
import math
from typing import Mapping

import numpy as np

COALITION_VERSION = "owned-books-sharing-1"
MODES = ("uav", "adr")


@dataclass(frozen=True)
class SharingContract:
    owners: Mapping[int, str]
    groups: Mapping[int, str]
    permissions: Mapping[int, tuple]
    members: tuple = MODES
    eps_shared: float = 0.0
    eps_uav_to_adr: float = 0.0
    eps_adr_to_uav: float = 0.0

    def __post_init__(self):
        if set(self.owners) != set(self.groups) or set(self.owners) != set(self.permissions):
            raise ValueError("Ownership, groups and permissions must describe the same book")
        if not self.members or len(set(self.members)) != len(self.members) or not set(self.members) <= set(MODES):
            raise ValueError("Invalid coalition members")
        for fraction in (self.eps_shared, self.eps_uav_to_adr, self.eps_adr_to_uav):
            if not math.isfinite(fraction) or not 0 <= fraction <= 1:
                raise ValueError("Sharing fractions must be in [0, 1]")
        for rid, owner in self.owners.items():
            group = self.groups[rid]
            if owner not in MODES or group not in ("shared", *MODES):
                raise ValueError("Invalid firm owner or market group")
            if group != "shared" and group != owner:
                raise ValueError("Exclusive-book group must match its owner")
            if not set(self.permissions[rid]) <= set(MODES):
                raise ValueError("Invalid mode permissions")

    @classmethod
    def from_arrivals(cls, arrivals, members=MODES, **fractions):
        ids = [int(row['req_id']) for row in arrivals]
        if len(set(ids)) != len(ids):
            raise ValueError("A task book cannot contain duplicate request IDs")
        return cls(
            owners={int(row['req_id']): row['firm_owner'] for row in arrivals},
            groups={int(row['req_id']): row.get('market_group', 'shared') for row in arrivals},
            permissions={int(row['req_id']): tuple(row.get('allowed_modes', MODES)) for row in arrivals},
            members=tuple(members), **fractions,
        )

    def budgets(self):
        fractions = dict(shared=self.eps_shared, uav=self.eps_uav_to_adr, adr=self.eps_adr_to_uav)
        return {group: math.floor(fractions[group] * sum(g == group for g in self.groups.values()) + 1e-12)
                for group in fractions}

    def usage(self, assignments):
        used = dict(shared=0, uav=0, adr=0)
        for rid, mode in assignments.items():
            if rid not in self.owners or mode not in MODES:
                raise ValueError("Assignment not in the contract")
            if mode != self.owners[rid]:
                used[self.groups[rid]] += 1
        return used

    def allows(self, rid, mode, assignments):
        if rid not in self.owners or mode not in self.members or mode not in self.permissions[rid]:
            return False
        if rid in assignments:
            return assignments[rid] == mode
        if mode == self.owners[rid]:
            return True
        group = self.groups[rid]
        return self.usage(assignments)[group] < self.budgets()[group]

    def reserve(self, rid, mode, assignments):
        if not self.allows(rid, mode, assignments):
            return False
        assignments[rid] = mode
        return True


def request_mode_allowed(request, mode, full_instance):
    if mode not in getattr(request, 'allowed_modes', MODES):
        return False
    contract = full_instance.get('sharing_contract')
    return contract is None or contract.allows(
        request.req_id, mode, full_instance.get('sharing_assignments', {}),
    )


def ops_permitted(ops, vehicles, requests, full_instance):
    """Check fleet-wide assignment coupling before accepting an insertion."""
    contract = full_instance.get('sharing_contract')
    ledger = dict(full_instance.get('sharing_assignments', {}))
    carriers = {r.req_id: r.assigned_vehicle for r in requests.values() if r.assigned_vehicle is not None}
    for vid, route in ops.items():
        if vid not in vehicles:
            return False
        mode = vehicles[vid].normalized_mode()
        seen = set()
        for rid, role in route:
            if rid not in requests or (rid, role) in seen:
                return False
            seen.add((rid, role))
            if rid in carriers and carriers[rid] != vid:
                return False
            if mode not in getattr(requests[rid], 'allowed_modes', MODES):
                return False
            if contract is not None and not contract.reserve(rid, mode, ledger):
                return False
            carriers[rid] = vid
    return True


def _numpy(value):
    return value.detach().cpu().numpy() if hasattr(value, 'detach') else np.asarray(value)


def _like(value, array):
    return value.new_tensor(array) if hasattr(value, 'new_tensor') else array


def planner_instance(full_instance, active_requests, vehicles):
    """Intersect physical accessibility with contractual restrictions for all solvers."""
    result = dict(full_instance)
    n = int(full_instance['n_depots']) + 2 * int(full_instance['n_req'])
    x_original = full_instance.get('x')
    x = _numpy(x_original).copy() if x_original is not None else None
    for mode, column in (('uav', 8), ('adr', 9)):
        key = 'mask_adjacency_' + mode
        original = full_instance.get(key)
        mask = _numpy(original).reshape(n, n).copy() if original is not None else np.ones((n, n))
        blocked = []
        for request in active_requests.values():
            if not request_mode_allowed(request, mode, full_instance):
                blocked.extend((request.pickup_node, request.delivery_node))
        # Coalition fleets have common mode-specific access sets. Never union
        # heterogeneous vehicle permissions into a more permissive mode mask.
        fleet = [v for v in vehicles.values() if v.normalized_mode() == mode]
        sets = [set(v.allowed_depots) for v in fleet if v.allowed_depots is not None]
        if sets:
            allowed = set.intersection(*sets)
            blocked.extend(d for d in range(int(full_instance['n_depots'])) if d not in allowed)
        if blocked:
            mask[:, blocked] = 0
            # An occupied node remains a valid origin for leaving it.
            if x is not None:
                x[blocked, column] = 0
        shape = original.shape if original is not None else (n * n, 1)
        result[key] = _like(original, mask.reshape(shape))
    if x is not None:
        result['x'] = _like(x_original, x)
    return result


def constrain_plan(route_plan, requests, vehicles, full_instance):
    """Re-simulate feasible prefixes with coupled quotas and single carriers.

    Vehicle-ID order is an explicit deterministic fallback projection, not a
    quota optimizer. Rejections are counted and make model-conformance visible.
    Internal joint quota-aware search still needs backend-specific validation.
    """
    from ce_cpdptw_alns import _simulate_planned_leg
    contract = full_instance.get('sharing_contract')
    assignments = dict(full_instance.get('sharing_assignments', {}))
    carriers = {r.req_id: r.assigned_vehicle for r in requests.values() if r.assigned_vehicle is not None}
    context = dict(full_instance, sharing_assignments=assignments)
    result, rejected = {}, []
    for vid in sorted(route_plan):
        if vid not in vehicles:
            rejected.append(dict(vehicle_id=vid, reason='unknown_vehicle'))
            continue
        state = copy.deepcopy(vehicles[vid])
        if state.committed_leg is not None:
            result[vid] = []
            continue
        result[vid] = []
        seen_ops = set()
        for leg in route_plan[vid]:
            req = requests.get(leg.request_id)
            if req is not None:
                if req.status == 'delivered' or (leg.leg_type == 'pickup' and req.status != 'waiting_for_pickup'):
                    continue
            elif leg.leg_type == 'depot' and leg.t_arrive <= state.current_time and leg.to_node == state.current_node:
                continue
            if (leg.request_id, leg.leg_type) in seen_ops and leg.leg_type != 'depot':
                rejected.append(dict(vehicle_id=vid, to_node=leg.to_node, reason='duplicate_operation'))
                break
            seen_ops.add((leg.request_id, leg.leg_type))
            if (leg.vehicle_id != vid or leg.from_node != state.current_node
                    or (req is not None and req.req_id in carriers and carriers[req.req_id] != vid)):
                rejected.append(dict(vehicle_id=vid, to_node=leg.to_node, reason='origin_or_carrier'))
                break
            checked, following, ok = _simulate_planned_leg(state, req, leg, context)
            if not ok or checked is None:
                rejected.append(dict(vehicle_id=vid, to_node=leg.to_node, reason='physical_or_contract'))
                break
            if req is not None and leg.leg_type == 'pickup':
                if contract is not None and not contract.reserve(req.req_id, state.normalized_mode(), assignments):
                    raise RuntimeError("Simulation and sharing contract disagree")
                carriers[req.req_id] = vid
            result[vid].append(checked)
            state = following
    return result, rejected
