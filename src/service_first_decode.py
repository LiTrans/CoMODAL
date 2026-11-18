"""Frozen-weight, replay-scored decoding and per-order reachability trace."""

import argparse
from collections import Counter
from copy import copy, deepcopy
from hashlib import file_digest
import json
from pathlib import Path
from time import perf_counter
from unittest.mock import patch

import torch



def first_legs(plan, residual):
    from dispatch_sim import pick_first_uncommitted_leg

    active = residual['active_requests']
    depot_count = int(residual['full_instance']['n_depots'])
    fleet = residual['vehicles']
    return {vid: leg for vid, route in plan.items() if fleet[vid].committed_leg is None
            and (leg := pick_first_uncommitted_leg(route, active, depot_count)) is not None}


def leg_record(leg):
    return dict(kind=leg.leg_type, request=leg.request_id, source=leg.from_node,
                target=leg.to_node, operating=leg.operating_cost,
                start=leg.t_depart, finish=leg.t_arrive)


def score_candidate(plan, residual, label):
    from coalition_contract import constrain_plan
    from reward import score_route_plan

    fleet, active = residual['vehicles'], residual['active_requests']
    context = residual['full_instance']
    checked, rejected = constrain_plan(plan, active, fleet, context)
    score = score_route_plan(checked, fleet, active, context)
    first = first_legs(checked, residual)
    delivered = sorted({leg.request_id for route in checked.values() for leg in route
                        if leg.leg_type == 'delivery'})
    pickups = {leg.request_id for route in checked.values() for leg in route
               if leg.leg_type == 'pickup'}
    # A sampled plan is useful only if its accepted prefixes close its new pickups.
    record = dict(label=label, unassigned=score['unassigned_count'], delivered=delivered,
                  prefix_cost=score['prefix_cost'], operating=score['operating'],
                  visits=sum(leg.leg_type == 'depot' for route in checked.values() for leg in route),
                  rejected=len(rejected), unclosed_new_pickups=sorted(pickups - set(delivered)),
                  first={vid: leg_record(leg) for vid, leg in first.items()})
    return checked, record


def first_action_choices(trace, vids, event, depot_vehicles, depot_count):
    proposals = []
    for entry in trace:
        if entry['path'] != 'sequential' or entry['episode_step'] != 0:
            continue
        if bool(entry['all_masked'].item()):
            continue
        agent = int(entry['agent'])
        vid = vids[agent]
        if event == 'depot' and vid not in depot_vehicles:
            continue
        if event == 'stop' and not bool(entry['stop_selected'].item()):
            continue
        count = depot_count + 2 * entry['num_pickup']
        active = entry['legal_mask'].flatten()[depot_count:count].nonzero().flatten() + depot_count
        if active.numel() == 0:
            continue
        log_probs = entry['log_probs'].flatten()[active]
        for pos in log_probs.argsort(descending=True)[:2]:
            proposals.append((float(log_probs[pos]), agent, vid, int(active[pos])))
    return sorted(proposals, reverse=True)


def lost_on_commit(residual, plan):
    """Find waiting orders with a suffix now that the first legs would remove."""
    from coalition_contract import constrain_plan

    active = residual['active_requests']
    checked, _ = constrain_plan(plan, active, residual['vehicles'], residual['full_instance'])
    covered = {leg.request_id for route in checked.values() for leg in route
               if leg.leg_type == 'delivery'}
    omitted = [rid for rid, req in active.items() if req.status == 'waiting_for_pickup'
               and rid not in covered]
    if not omitted:
        return []
    successor = dict(residual, vehicles=dict(residual['vehicles']), active_requests=dict(active))
    for vid, leg in first_legs(checked, residual).items():
        carrier = copy(successor['vehicles'][vid])
        carrier.committed_leg = leg
        successor['vehicles'][vid] = carrier
        if leg.request_id in active:
            request = copy(active[leg.request_id])
            request.assigned_vehicle = vid
            request.status = 'pickup_committed' if leg.leg_type == 'pickup' else 'delivery_committed'
            successor['active_requests'][leg.request_id] = request
    lost = []
    for rid in omitted:
        proof = delivery_witness(residual, rid)
        if proof is not None and delivery_witness(successor, rid) is None:
            lost.append((rid, proof))
    return lost


class ServiceFirstDecoder:
    def __init__(self, solver, samples, depot_limit, force_limit, diagnostics, witness_alternatives,
                 offer_depot_stop, protect, opportunity_limit):
        self.solver = solver
        self.greedy = solver.solve
        self.samples = samples
        self.depot_limit = depot_limit
        self.force_limit = force_limit
        self.diagnostics = diagnostics
        self.witness_alternatives = witness_alternatives
        self.offer_depot_stop = offer_depot_stop
        self.protect = protect
        self.opportunity_limit = opportunity_limit
        self.depot_events = 0
        self.opportunity_events = 0
        self.history = []

    def decode(self, residual, *, sampled=False, forced=None, trace=None):
        import VRP_Actor

        actor = self.solver.actor
        forward = actor.forward

        def decode_control(data, *args, **kwargs):
            if forced is not None:
                data = dict(data, diagnostic_actions=forced)
            kwargs['greedy'] = not sampled
            return forward(data, *args, **kwargs)

        with patch.object(actor, 'forward', decode_control), patch.object(
            VRP_Actor, '_DECODER_TRACE_SINK', trace
        ):
            return self.greedy(residual)

    def __call__(self, residual):
        from ce_cpdptw_alns import _full_battery, _recharge_minutes, remaining_recharges
        from reward import objective_from_instance

        plan = self.greedy(residual)
        stopped = getattr(self.solver, 'last_decode_status', {}).get('stopped')
        fleet, active = residual['vehicles'], residual['active_requests']
        context = residual['full_instance']
        pending = context['remaining_delivery_ids']
        first = first_legs(plan, residual)
        depots = [vid for vid, leg in first.items() if leg.leg_type == 'depot']
        has_next_event = pending != set(active) or any(vehicle.committed_leg for vehicle in fleet.values())
        last_replan = not first and pending and not has_next_event
        final_stop = last_replan and stopped is not None and bool(stopped.any())
        protect_request = None
        if self.protect is not None:
            rid, at = self.protect
            if abs(residual['current_time'] - at) < 1e-5 and rid in active:
                protect_request = rid
        event = ('protect' if protect_request is not None
                 else 'stop' if final_stop else 'stall' if last_replan and self.witness_alternatives
                 else 'depot' if depots and self.depot_events < self.depot_limit else None)
        lost = []
        if event is None and first and self.opportunity_events < self.opportunity_limit:
            lost = lost_on_commit(residual, plan)
            if lost:
                event = 'opportunity'
                self.opportunity_events += 1
        selected = plan
        if event:
            self.depot_events += event == 'depot'
            trace = []
            greedy = self.decode(residual, trace=trace)
            candidates = [score_candidate(greedy, residual, 'greedy')]
            order = self.solver._vehicle_ids(fleet)
            options = first_action_choices(trace, order, event, depots, int(context['n_depots']))
            for _, agent, vid, node in options[:self.force_limit]:
                alternative = self.decode(residual, forced={(0, agent): node})
                candidates.append(score_candidate(alternative, residual, f'force_v{vid}_n{node}'))
            if event == 'depot' and self.offer_depot_stop:
                for entry in trace:
                    if entry['path'] != 'sequential' or entry['episode_step'] != 0:
                        continue
                    agent = int(entry['agent'])
                    vid = order[agent]
                    if vid not in depots:
                        continue
                    stop = int(context['n_depots']) + 2 * int(entry['num_pickup'])
                    legal = entry['legal_mask'].flatten()
                    if stop < len(legal) and legal[stop]:
                        alternative = self.decode(residual, forced={(0, agent): stop})
                        candidates.append(score_candidate(alternative, residual, f'hold_v{vid}'))
            for index in range(self.samples):
                alternative = self.decode(residual, sampled=True)
                candidates.append(score_candidate(alternative, residual, f'sample_{index + 1}'))
            witness_rejections = []
            if (self.witness_alternatives and last_replan) or protect_request is not None:
                targets = [protect_request] if protect_request is not None else sorted(active)
                proofs = [(rid, delivery_witness(residual, rid)) for rid in targets]
            else:
                proofs = lost
            usable = [(rid, proof) for rid, proof in proofs if proof is not None]
            usable.sort(key=lambda item: sum(leg['operating'] for leg in item[1]['suffix']))
            for rid, proof in usable[:self.force_limit]:
                agent = order.index(proof['vehicle'])
                forced = {(step, agent): move['target'] for step, move in enumerate(proof['suffix'])}
                try:
                    alternative = self.decode(residual, forced=forced)
                except ValueError as error:
                    if str(error) != 'Requested diagnostic alternative is masked':
                        raise
                    witness_rejections.append(rid)
                    continue
                candidates.append(score_candidate(alternative, residual, f'witness_r{rid}'))
            best_plan, best = candidates[0]

            def key(record):
                protection = int(protect_request is not None and protect_request not in record['delivered'])
                return protection, record['unassigned'], record['prefix_cost']

            for candidate, record in candidates[1:]:
                if record['rejected'] or record['unclosed_new_pickups']:
                    continue
                if not (record['first'] or has_next_event):
                    continue
                if key(record) < key(best):
                    best_plan, best = candidate, record
            selected = best_plan
            tariff = objective_from_instance(context)
            depot_detail = [{
                'vehicle': vid, 'mode': fleet[vid].normalized_mode(),
                'battery_fraction': fleet[vid].battery / _full_battery(fleet[vid], context),
                'visits_left': remaining_recharges(fleet[vid], context),
                'dwell_cost': _recharge_minutes(fleet[vid]) * tariff.operating(fleet[vid].normalized_mode()),
            } for vid in depots]
            self.diagnostics.append(dict(
                event=event, time=residual['current_time'], depot=depot_detail, selected=best['label'],
                alternatives=[record for _, record in candidates], targeted_customer_options=len(options),
                witness_mask_rejections=witness_rejections, protect_request=protect_request,
                lost_on_greedy_commit=[rid for rid, _ in lost],
            ))

        frozen_context = dict(context, remaining_delivery_ids=set(pending))
        self.history.append(dict(
            current_time=residual['current_time'], full_instance=frozen_context,
            active_requests=deepcopy(active), vehicles=deepcopy(fleet),
            first={vid: leg_record(leg) for vid, leg in first_legs(selected, residual).items()},
        ))
        return selected


def delivery_witness(state, target, *, typed=False, free_only=False, depart_at_least=None):
    """Search one carrier's existing onboard work plus one revealed order."""
    from ce_cpdptw_alns import _simulate_leg, _simulate_planned_leg, remaining_recharges

    requests = state['active_requests']
    if target not in requests:
        return None
    for vid, initial in state['vehicles'].items():
        vehicle = initial
        instance = state['full_instance']
        leg = initial.committed_leg
        if free_only and leg is not None:
            continue
        if leg is not None:
            checked, vehicle, ok = _simulate_planned_leg(initial, requests.get(leg.request_id), leg, instance)
            if not ok:
                continue
            vehicle.committed_leg = None
            if checked.leg_type == 'delivery':
                if checked.request_id == target:
                    return dict(vehicle=vid, suffix=[checked if typed else leg_record(checked)])
                instance = dict(instance, remaining_delivery_ids=instance['remaining_delivery_ids'] - {checked.request_id})
        if depart_at_least is not None and vehicle.current_time < depart_at_least:
            vehicle = copy(vehicle)
            vehicle.current_time = depart_at_least
        target_req = requests[target]
        if target_req.assigned_vehicle is not None and target_req.assigned_vehicle != vid:
            continue
        to_pick = target_req.status == 'waiting_for_pickup'
        if not to_pick and target not in vehicle.onboard_set():
            continue

        seen = {}

        def suffix(carrier, context, pickup_needed):
            onboard = carrier.onboard_set()
            if not onboard and not pickup_needed:
                return []
            cap = remaining_recharges(carrier, context)
            key = (carrier.current_node, tuple(sorted(onboard)), pickup_needed, cap)
            if seen.get(key, -1.) >= carrier.battery - 1e-8:
                return None
            seen[key] = carrier.battery
            moves = [(requests[rid], 'delivery', requests[rid].delivery_node) for rid in sorted(onboard)]
            if pickup_needed:
                moves.append((target_req, 'pickup', target_req.pickup_node))
            if cap:
                moves.extend((None, 'depot', depot) for depot in carrier.allowed_depots or ()
                             if depot != carrier.current_node)
            for request, kind, node in moves:
                next_leg, next_carrier, feasible = _simulate_leg(carrier, request, kind, node, context)
                if not feasible:
                    continue
                following = context
                if kind == 'delivery':
                    following = dict(context, remaining_delivery_ids=context['remaining_delivery_ids'] - {request.req_id})
                tail = suffix(next_carrier, following, pickup_needed and kind != 'pickup')
                if tail is not None:
                    return [next_leg, *tail]
            return None

        path = suffix(vehicle, instance, to_pick)
        if path:
            return dict(vehicle=vid, suffix=path if typed else [leg_record(leg) for leg in path])
    return None


def last_service_opportunities(history, pending):
    result = []
    for rid in sorted(pending):
        for index in range(len(history) - 1, -1, -1):
            state = history[index]
            if rid not in state['active_requests']:
                break
            witness = delivery_witness(state, rid)
            if witness is None:
                continue
            vid = witness['vehicle']
            v = state['vehicles'][vid]
            following = history[index + 1] if index + 1 < len(history) else None
            later_v = following['vehicles'][vid] if following is not None else None
            result.append(dict(
                request=rid, last_witness_time=state['current_time'], first_unwitnessed_time=(
                    following['current_time'] if following is not None else None
                ), witness=witness, committed=state['first'].get(vid), before=dict(
                    node=v.current_node, battery=v.battery, visits_used=v.recharge_count,
                ), after=None if later_v is None else dict(
                    node=later_v.current_node, battery=later_v.battery,
                    visits_used=later_v.recharge_count,
                    in_flight=leg_record(later_v.committed_leg) if later_v.committed_leg is not None else None,
                ),
            ))
            break
        else:
            result.append(dict(request=rid, witness=None))
            continue
        if not result or result[-1]['request'] != rid:
            result.append(dict(request=rid, witness=None))
    return result

