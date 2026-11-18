"""Replay rD with depot witnesses and pickup opportunity checks at commitment."""

import argparse
from collections import Counter, defaultdict
from copy import copy
from hashlib import file_digest
import json
from pathlib import Path
from time import perf_counter
from unittest.mock import patch

import torch

from service_first_decode import (
    ServiceFirstDecoder, delivery_witness, first_legs, leg_record, score_candidate,
)


def delivered_by(route):
    return {leg.request_id for leg in route if leg.leg_type == 'delivery'}


def closed_delivery_prefix(route, vehicle):
    delivered = delivered_by(route)
    if not delivered:
        return []
    last = max(index for index, leg in enumerate(route) if leg.leg_type == 'delivery')
    prefix = route[:last + 1]
    required = vehicle.onboard_set() | {leg.request_id for leg in prefix if leg.leg_type == 'pickup'}
    return prefix if required <= delivered else []


def replay_stop_order(vehicle, route, residual):
    from ce_cpdptw_alns import _schedule_route_departure, _simulate_leg

    context, active = residual['full_instance'], residual['active_requests']
    rebased = []
    for prior in route:
        order = active.get(prior.request_id)
        leg, vehicle, ok = _simulate_leg(vehicle, order, prior.leg_type, prior.to_node, context)
        if not ok:
            return None
        rebased.append(leg)
    return _schedule_route_departure(rebased, active, context)[0]


def validate_depot_prefixes(plan, residual):
    """Drop leading depot visits that serve no cargo or can be bypassed."""
    from coalition_contract import constrain_plan

    fleet, active, context = residual['vehicles'], residual['active_requests'], residual['full_instance']
    checked, _ = constrain_plan(plan, active, fleet, context)
    reasons = []
    for vid, route in checked.items():
        while route and route[0].leg_type == 'depot':
            current = route[0]
            witness = closed_delivery_prefix(route, fleet[vid])
            if not witness:
                reasons.append(dict(vehicle=vid, depot=current.to_node, reason='no_delivery_suffix'))
                route = []
                break
            bypass = replay_stop_order(fleet[vid], witness[1:], residual)
            if bypass is None:
                route = witness
                break
            reasons.append(dict(vehicle=vid, depot=current.to_node, reason='same_stops_without_visit'))
            route = bypass
        checked[vid] = route

    checked, rejected = constrain_plan(checked, active, fleet, context)
    certificates = {}
    for vid, leg in first_legs(checked, residual).items():
        if leg.leg_type != 'depot':
            continue
        witness = closed_delivery_prefix(checked[vid], fleet[vid])
        if not witness:
            checked[vid] = []
            reasons.append(dict(vehicle=vid, depot=leg.to_node, reason='projected_suffix_incomplete'))
            continue
        certificates[vid] = dict(visit=leg_record(leg), enables=sorted(delivered_by(witness)),
                                 suffix=[leg_record(action) for action in witness])
    return checked, certificates, reasons, rejected


def projected_commitment(residual, first):
    fleet = dict(residual['vehicles'])
    active = dict(residual['active_requests'])
    for vid, leg in first.items():
        vehicle = copy(fleet[vid])
        vehicle.committed_leg = leg
        fleet[vid] = vehicle
        if leg.request_id in active:
            request = copy(active[leg.request_id])
            request.assigned_vehicle = vid
            request.status = 'pickup_committed' if leg.leg_type == 'pickup' else 'delivery_committed'
            active[leg.request_id] = request
    return dict(residual, vehicles=fleet, active_requests=active)


class GuardedCommitment:
    def __init__(self, solver, sample_count, width):
        self.model = ServiceFirstDecoder(solver, 0, 0, 0, [], False, False, None, 0)
        self.sample_count = sample_count
        self.width = width
        self.decisions = []
        self.depot_plans = []
        self.replan_count = 0
        self.certified_depots_by_carrier = Counter()

    def solve(self, residual):
        self.replan_count += 1
        greedy = self.model.greedy(residual)
        active, fleet, context = residual['active_requests'], residual['vehicles'], residual['full_instance']
        pending = context['remaining_delivery_ids']
        clock_can_advance = pending != set(active) or any(v.committed_leg for v in fleet.values())
        waiting = sorted(rid for rid, req in active.items() if req.status == 'waiting_for_pickup')
        cached_witnesses = {}

        def witness(rid, free_only=False):
            key = (rid, free_only)
            if key not in cached_witnesses:
                cached_witnesses[key] = delivery_witness(residual, rid, typed=True, free_only=free_only)
            return cached_witnesses[key]

        prepared = {}

        def assess(label, raw):
            checked, depots, suppressed, rejected = validate_depot_prefixes(raw, residual)
            first = first_legs(checked, residual)
            key = tuple((vid, tuple((leg.leg_type, leg.request_id, leg.to_node, leg.t_depart) for leg in route))
                        for vid, route in checked.items() if route)
            if key in prepared:
                return prepared[key]
            picked = {leg.request_id for leg in first.values() if leg.leg_type == 'pickup'}
            advanced = projected_commitment(residual, first) if picked else None
            lost = []
            if advanced is not None:
                for rid in waiting:
                    if rid not in picked and witness(rid) is not None and delivery_witness(advanced, rid) is None:
                        lost.append(rid)
            checked, record = score_candidate(checked, residual, label)
            record.update(lost=lost, suppressed_depots=suppressed, depot_witnesses=depots,
                          projection_rejections=rejected, clock_can_advance=bool(first) or clock_can_advance)
            candidate = checked, record
            prepared[key] = candidate
            return candidate

        best = primary = assess('guarded_greedy', greedy)
        block = primary[1]
        candidates = [primary]
        threatened = block['lost']
        depot_first = bool(block['depot_witnesses'] or block['suppressed_depots'])
        stalled = not block['clock_can_advance'] and bool(active)
        if depot_first or threatened or stalled:
            hold_vehicles = set(block['depot_witnesses'])
            hold_vehicles.update(witness(rid)['vehicle'] for rid in threatened)
            for vid in sorted(hold_vehicles)[:self.width]:
                if primary[0].get(vid) and not fleet[vid].onboard_set():
                    candidates.append(assess(f'hold_v{vid}', primary[0] | {vid: []}))
            if clock_can_advance:
                candidates.append(assess('hold_all', {}))

            targets = list(threatened)
            if depot_first or stalled:
                targets.extend(rid for rid in sorted(active) if rid not in targets
                               and active[rid].status in ('onboard', 'waiting_for_pickup'))
            proofs = [(rid, witness(rid, free_only=True)) for rid in targets]
            valid = [(rid, proof) for rid, proof in proofs if proof is not None]
            for rid, proof in valid[:self.width]:
                carrier = proof['vehicle']
                suffix = proof['suffix']
                reassigned = {leg.request_id for leg in suffix if leg.leg_type == 'pickup'}
                combined = {vid: route for vid, route in primary[0].items()
                            if not (reassigned & delivered_by(route))}
                candidates.append(assess(f'witness_r{rid}', combined | {carrier: suffix}))
            for k in range(self.sample_count):
                candidates.append(assess(f'sample_{k + 1}', self.model.decode(residual, sampled=True)))

            feasible = [candidate for candidate in candidates if candidate[1]['clock_can_advance']]
            if feasible:
                best = min(feasible, key=lambda pair: (
                    len(pair[1]['lost']), pair[1]['unassigned'], pair[1]['prefix_cost'],
                ))
            self.decisions.append(dict(
                time=residual['current_time'], primary_lost=threatened, primary_depot=depot_first,
                selected=best[1]['label'], candidates=[candidate[1] for candidate in candidates],
            ))
        chosen = best[1]['depot_witnesses']
        self.certified_depots_by_carrier.update(chosen.keys())
        for vid, cert in chosen.items():
            self.depot_plans.append(dict(vehicle=vid, **cert))
        return best[0]

