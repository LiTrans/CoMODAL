from dataclasses import asdict, dataclass, replace
import math

ALPHA_1 = 0.60
ALPHA_2 = 0.10
ALPHA_E = 0.02
ALPHA_P = 0.10
ALPHA_D = 0.15
ALPHA_U = 8.0
LAMBDA_BATTERY = 1.0
# Preserve the legacy symbol, but do not change the scientific objective with
# a non-potential delivery bonus. Any future shaping needs a separate ablation.
ALPHA_DELIV_BONUS = 0.0

# Optional training-only shaping/Lagrangian defaults; see MandatoryTrainingAids.
# Both default to zero so plain five-cost training is unaffected until explicitly
# enabled from a config/CLI knob. Reported scoring never includes these terms.
TRAINING_DELIVERY_SHAPING_K = 0.0
TRAINING_BATTERY_LAGRANGIAN_LAMBDA = 0.0

alpha_asymmetry_ratio = ALPHA_P / ALPHA_D

OBJECTIVE_VERSION = 'weighted-travel-event-penalties-1'
MANDATORY_OBJECTIVE_VERSION = 'mandatory-five-raw-arrival-1'


class IncompleteTensorRouteError(ValueError):
    """A decoder prefix is not a completed mandatory-service episode."""


def native_physical_configuration(context):
    """Supported native modes only; never infer terminality from a residual.

    The shared transitions support more modes, but this decoder has no
    independently checked chronological whole-book terminal ledger yet.
    """
    from hardware_profiles import physical_configuration
    value = context.get('physical_config')
    if value is None:
        return None
    config = physical_configuration(value)
    import torch
    def scalar_fields(value):
        if isinstance(value, dict):
            return all(scalar_fields(item) for item in value.values())
        return not isinstance(value, (torch.Tensor, list, tuple))
    if not scalar_fields(config.to_dict()):
        raise ValueError('Native physical_config must be one explicit scalar configuration per batch, not collated modes')
    if config.continuation_policy != 'legacy_leg_only' or config.uav_phases.terminal_delivery != 'customer_pair':
        raise NotImplementedError('Native configured terminal/continuation safety requires a validated persistent '
                                  'whole-book completion ledger; residual snapshots cannot certify it')
    if config.uav_phases.has_accounting_gaps:
        raise NotImplementedError('Native configured physics requires explicit depot landing/takeoff durations')
    return config


def with_mandatory_contract(batch):
    """Bind supported scientific training/evaluation to the five-term model."""
    result = dict(batch)
    config = native_physical_configuration(result)
    if config is not None:
        result['physical_config'] = config.to_dict()
    if result.get('service_policy', 'mandatory') != 'mandatory':
        raise ValueError('Scientific route evaluation cannot use legacy_optional service')
    result['service_policy'] = 'mandatory'
    result['objective_version'] = MANDATORY_OBJECTIVE_VERSION
    result['objective_weights'] = asdict(mandatory_objective_weights(result.get('objective_weights')))
    return result


def route_checkpoint_contract():
    from model_config import MODEL_VERSION, ENERGY_MODEL_VERSION, GENERATOR_VERSION
    return dict(service_policy='mandatory', objective_version=MANDATORY_OBJECTIVE_VERSION,
                model_version=MODEL_VERSION, energy_model_version=ENERGY_MODEL_VERSION,
                generator_version=GENERATOR_VERSION)


def validate_route_checkpoint(checkpoint):
    if not isinstance(checkpoint, dict) or checkpoint.get('route_contract') != route_checkpoint_contract():
        raise ValueError('Checkpoint lacks the current mandatory route contract; legacy weights are diagnostics only')


def tensor_sharing_context(instance, request_ids, device):
    """Translate a real whole-book contract, never infer ownership from access."""
    import torch
    contract = instance.get('sharing_contract')
    if contract is None:
        return {}
    ledger = dict(instance.get('sharing_assignments', {}))
    mode = {'uav': 0, 'adr': 1}
    groups = {'shared': 0, 'uav': 1, 'adr': 2}
    if any(r not in contract.owners for r in request_ids):
        raise ValueError('Tensor request IDs are missing from the sharing contract')
    used, budgets = contract.usage(ledger), contract.budgets()
    if any(used[g] > budgets[g] for g in groups):
        raise ValueError('Existing whole-book sharing ledger exceeds its budget')
    return {
        'request_firm_owner': torch.tensor([[mode[contract.owners[r]] for r in request_ids]], device=device),
        'request_market_group': torch.tensor([[groups[contract.groups[r]] for r in request_ids]], device=device),
        'request_allowed_modes': torch.tensor([[
            [m in contract.members and m in contract.permissions[r] for m in ('uav', 'adr')]
            for r in request_ids]], dtype=torch.bool, device=device),
        'request_assigned_mode': torch.tensor([[mode[ledger[r]] if r in ledger else -1
                                               for r in request_ids]], device=device),
        'sharing_budget': torch.tensor([[budgets[g] for g in groups]], device=device),
        'sharing_used': torch.tensor([[used[g] for g in groups]], device=device),
    }


class TensorSharingState:
    """Candidate-local quota reservations; persistent dispatcher ledger is untouched."""
    KEYS = ('request_firm_owner', 'request_market_group', 'request_allowed_modes',
            'request_assigned_mode', 'sharing_budget', 'sharing_used')

    def __init__(self, batch, batch_size, n_req, num_uav, n_agents, device):
        import torch
        present = [key in batch for key in self.KEYS]
        if any(present) and not all(present):
            raise ValueError('Incomplete native tensor sharing context')
        self.enabled = all(present)
        if not self.enabled:
            if batch.get('sharing_contract') is not None:
                raise NotImplementedError('Materialize the whole-book tensor sharing context before decoding')
            return
        def integer(key, shape):
            value = torch.as_tensor(batch[key], device=device)
            if tuple(value.shape) != shape or not torch.isfinite(value).all() or (value != value.long()).any():
                raise ValueError('Invalid tensor sharing field: ' + key)
            return value.long().clone()
        self.owners = integer('request_firm_owner', (batch_size, n_req))
        self.groups = integer('request_market_group', (batch_size, n_req))
        self.assigned = integer('request_assigned_mode', (batch_size, n_req))
        self.budgets = integer('sharing_budget', (batch_size, 3))
        self.used = integer('sharing_used', (batch_size, 3))
        permissions = torch.as_tensor(batch['request_allowed_modes'], device=device)
        if tuple(permissions.shape) != (batch_size, n_req, 2) or ((permissions != 0) & (permissions != 1)).any():
            raise ValueError('Invalid tensor mode permissions')
        self.permissions = permissions.bool()
        self.modes = (torch.arange(n_agents, device=device) >= num_uav).long()
        if ((self.owners < 0) | (self.owners > 1) | (self.groups < 0) | (self.groups > 2)
                | (self.assigned < -1) | (self.assigned > 1)).any():
            raise ValueError('Invalid tensor ownership or market group')
        if ((self.groups > 0) & (self.groups != self.owners + 1)).any():
            raise ValueError('Exclusive market group disagrees with firm ownership')
        if (self.used < 0).any() or (self.budgets < self.used).any():
            raise ValueError('Invalid whole-book sharing usage/budget')
        for group in range(3):
            known_cross = ((self.assigned >= 0) & (self.assigned != self.owners)
                           & (self.groups == group)).sum(dim=1)
            if (known_cross > self.used[:, group]).any():
                raise ValueError('Known cross assignments exceed persistent sharing usage')

    def allowed(self, agent):
        mode = int(self.modes[agent])
        assigned = self.assigned
        permissions = self.permissions[:, :, mode]
        remaining = (self.used < self.budgets).gather(1, self.groups)
        return permissions & ((assigned == mode) | ((assigned < 0) & ((self.owners == mode) | remaining)))

    def reserve(self, req_ids, rows, agent):
        import torch
        if not rows.any():
            return
        indices = torch.where(rows)[0]
        requests = req_ids[indices]
        if not self.allowed(agent)[indices, requests].all():
            raise ValueError('Native tensor route violates sharing permissions or quota')
        mode = int(self.modes[agent])
        fresh = self.assigned[indices, requests] < 0
        cross = fresh & (self.owners[indices, requests] != mode)
        self.used[indices[cross], self.groups[indices[cross], requests[cross]]] += 1
        self.assigned[indices, requests] = mode


@dataclass(frozen=True)
class ObjectiveWeights:
    operating_uav: float = ALPHA_1
    operating_adr: float = ALPHA_2
    pickup_early: float = ALPHA_E
    pickup_late: float = ALPHA_P
    delivery_late: float = ALPHA_D
    unserved: float = ALPHA_U
    battery_shortfall: float = LAMBDA_BATTERY
    threshold_uav: float = .25
    threshold_adr: float = .20

    def __post_init__(self):
        if any(not math.isfinite(value) or value < 0 for value in asdict(self).values()):
            raise ValueError('Objective weights must be finite and nonnegative')
        if max(self.threshold_uav, self.threshold_adr) > 1:
            raise ValueError('Battery threshold fractions must be in [0, 1]')

    def operating(self, mode):
        if mode not in ('uav', 'adr'):
            raise ValueError('Unknown objective vehicle mode')
        return self.operating_uav if mode == 'uav' else self.operating_adr

    def battery_cost(self, remaining, full_pack, mode):
        if mode not in ('uav', 'adr'):
            raise ValueError('Unknown objective vehicle mode')
        fraction = self.threshold_uav if mode == 'uav' else self.threshold_adr
        return self.battery_shortfall * max(fraction * full_pack - remaining, 0.)


def objective_weights(value=None, alpha_p_ratio=None):
    weights = value if isinstance(value, ObjectiveWeights) else ObjectiveWeights(**(value or {}))
    if alpha_p_ratio is not None:
        weights = replace(weights, pickup_late=float(alpha_p_ratio) * weights.delivery_late)
    return weights


def objective_from_instance(instance):
    if instance.get('service_policy') == 'mandatory':
        return mandatory_objective_weights(instance.get('objective_weights'))
    return objective_weights(instance.get('objective_weights'))


def mandatory_service(instance):
    return instance.get('service_policy') == 'mandatory'


def mandatory_objective_weights(value=None, alpha_p_ratio=None):
    if value is None:
        value = replace(ObjectiveWeights(), unserved=0., battery_shortfall=0.)
    weights = objective_weights(value, alpha_p_ratio)
    if weights.unserved != 0 or weights.battery_shortfall != 0:
        raise ValueError('Mandatory service uses five costs, not unserved or soft battery penalties')
    return weights


def delivery_event_time(leg, instance):
    return leg.t_arrive_raw if mandatory_service(instance) else leg.t_arrive


def accept_service_insertion(delta, instance):
    return math.isfinite(delta) and (mandatory_service(instance)
                                    or delta < objective_from_instance(instance).unserved)


def pickup_penalty(actual, target, weights=None):
    weights = objective_weights(weights)
    return weights.pickup_early * max(target - actual, 0.) + weights.pickup_late * max(actual - target, 0.)


def delivery_penalty(actual, target, weights=None):
    return objective_weights(weights).delivery_late * max(actual - target, 0.)


def request_penalty(T_pickup_actual: float, t_p: float,
                    T_delivery_actual: float, t_d: float,
                    alpha_p_ratio: float = None, weights=None) -> float:
    weights = objective_weights(weights, alpha_p_ratio)
    return pickup_penalty(T_pickup_actual, t_p, weights) + delivery_penalty(T_delivery_actual, t_d, weights)


def score_route_plan(route_plan, vehicles, requests, full_instance):
    """Canonical residual cost after explicit physical/contract projection.

    This is a planned-route score, not a whole-shift optimum or an independent
    physics validator. Previously incurred pickup costs are not charged again.
    Incomplete mandatory prefixes retain their cost for heuristic ranking;
    their total is not a finite complete-obligation value. Deferral is legal.
    """
    from coalition_contract import constrain_plan
    weights = objective_from_instance(full_instance)
    checked, rejected = constrain_plan(route_plan, requests, vehicles, full_instance)
    operating = pickup = delivery = battery = 0.
    delivered = set()
    for vid, legs in checked.items():
        vehicle = vehicles[vid]
        from ce_cpdptw_alns import _full_battery
        for leg in legs:
            operating += leg.operating_cost
            battery += weights.battery_cost(leg.battery_after, _full_battery(vehicle), vehicle.normalized_mode())
            req = requests.get(leg.request_id)
            if leg.leg_type == 'pickup':
                pickup += pickup_penalty(leg.t_arrive_raw, req.t_pickup, weights)
            elif leg.leg_type == 'delivery':
                delivery += delivery_penalty(delivery_event_time(leg, full_instance), req.t_delivery, weights)
                delivered.add(req.req_id)
    obligations = {r.req_id for r in requests.values()
                   if r.status in ('waiting_for_pickup', 'onboard')
                   and (r.assigned_vehicle is None or vehicles[r.assigned_vehicle].committed_leg is None)}
    unserved = weights.unserved * len(obligations - delivered)
    complete = not (obligations - delivered)
    prefix_cost = operating + pickup + delivery + battery + unserved
    total = prefix_cost
    if mandatory_service(full_instance) and not complete:
        total = float('inf')
    return dict(total=total, prefix_cost=prefix_cost, complete=complete,
                unassigned_count=len(obligations - delivered),
                operating=operating, pickup=pickup, delivery=delivery,
                battery=battery, unserved=unserved, rejected_prefixes=len(rejected))


def trajectory_battery_cost(batch, tours, times, num_uav, num_depots, weights=None,
                            start_time=None, start_battery=None, start_capacity=None):
    """Replay actual pre-arc load and packs for the tensor objective.

    Resource verification only: this does not independently verify the power
    law, request revelation, contractual permissions or carrier precedence.
    """
    import torch
    from vrpUpdate import (update_state, _battery_reserve, _validate_configured_capacity,
                           recharge_depot_permissions, tensor_recharge_budget)
    config = native_physical_configuration(batch)
    weights = objective_weights(weights)
    B, A, S = tours.shape
    if A == 0 or not 0 <= num_uav <= A or S == 0 or tuple(times.shape) != (B, A, S - 1):
        raise ValueError('Invalid objective replay trajectory shape')
    if not torch.isfinite(times).all() or (times < 0).any():
        raise ValueError('Invalid objective replay completion times')
    reference = times.new_zeros(B, A)
    def fleet(key, override=None):
        value = override if override is not None else batch[key]
        return torch.as_tensor(value, device=tours.device, dtype=times.dtype).reshape(B, A)
    full_battery, full_capacity = fleet('battery'), fleet('capacity')
    reserve = _battery_reserve(full_battery, num_uav, config)
    _validate_configured_capacity(full_capacity, num_uav, config)
    initial_charge = start_battery if start_battery is not None else batch.get('start_battery')
    if initial_charge is None and config is not None:
        initial_charge = reference.new_tensor(
            [config.uav_battery.initial_charge_kj] * num_uav
            + [config.adr_battery.initial_charge_kj] * (A - num_uav)).expand(B, -1)
    battery = fleet('battery', initial_charge).clone()
    capacity = fleet('capacity', start_capacity if start_capacity is not None else batch.get('start_capacity')).clone()
    demand = batch['demand'].reshape(B, -1)
    if tours.dtype != torch.long or (tours < 0).any() or (tours >= demand.shape[1]).any():
        raise ValueError('Invalid objective replay node indices')
    if not torch.isfinite(torch.stack([full_battery, full_capacity, battery, capacity])).all():
        raise ValueError('Non-finite objective replay resources')
    # Singleton-coalition evaluation renders the complementary
    # fleet inert by zeroing its rated battery + capacity (see
    # tools/experiments/coalition_dispatcher_costs.py::_restrict_fleet). Those
    # slots must be skipped by the physical-sanity invariants that otherwise
    # require positive rated resources. Vehicles masked via available_mask
    # are also skipped. Active vehicles (positive rated + available) still
    # get full invariant enforcement.
    active = (full_battery > 0) & (full_capacity > 0)
    if 'available_mask' in batch:
        active = active & fleet('available_mask').bool()
    inv_breach = ((battery < reserve - 1e-5)
                   | (battery > full_battery + 1e-4)
                   | (capacity < 0)
                   | (capacity > full_capacity + 1e-4))
    if (active & inv_breach).any():
        raise ValueError('Invalid objective replay resources')
    fractions = reference.new_tensor([weights.threshold_uav] * num_uav + [weights.threshold_adr] * (A - num_uav))
    threshold = full_battery * fractions
    costs, violations = reference.clone(), torch.zeros_like(reference, dtype=torch.bool)
    initial_time = (torch.as_tensor(start_time, device=tours.device, dtype=times.dtype).reshape(B, A)
                    if start_time is not None else fleet('start_time') if 'start_time' in batch else reference)
    if not torch.isfinite(initial_time).all() or (initial_time < 0).any():
        raise ValueError('Invalid objective replay initial times')
    recharge_budget = tensor_recharge_budget(batch, (B, A), tours.device)
    allowed_depots = batch.get('agent_depot_allowed')
    from vrpUpdate import _pending_deliveries, pending_after_moves
    pending = _pending_deliveries(demand, num_depots, batch.get('remaining_deliveries'))
    for step in range(1, S):
        previous, current = tours[:, :, step - 1], tours[:, :, step]
        moved = previous != current
        if 'available_mask' in batch and (moved & ~fleet('available_mask').bool()).any():
            raise ValueError('Unavailable vehicle moved in objective replay')
        recharge = moved & (current < num_depots)
        if recharge_budget is not None and (recharge & (recharge_budget == 0)).any():
            raise ValueError('Route exceeds remaining recharge allowance')
        now = initial_time if step == 1 else times[:, :, step - 2]
        next_capacity, finish, next_battery = update_state(
            demand, batch['time_window'], battery, now, capacity, full_battery,
            num_uav, [previous.unsqueeze(-1), current.unsqueeze(-1)],
            batch['edge_attr_d'], batch['edge_attr_r'], num_depots=num_depots,
            max_capacity=full_capacity, wind=batch.get('wind'), coordinates=batch.get('coordinates'),
            physical_config=config, remaining_deliveries=pending,
            allowed_depots=recharge_depot_permissions(allowed_depots, recharge_budget, num_depots),
            strict=True,
        )
        if recharge_budget is not None:
            recharge_budget = recharge_budget - recharge.long()
        if not torch.allclose(finish, times[:, :, step - 1], atol=1e-4, rtol=1e-5):
            raise ValueError('Objective replay completion times disagree with physical transitions')
        pending = pending_after_moves(pending, current, moved, num_depots)
        capacity, battery = next_capacity, next_battery
        shortfall = torch.relu(threshold - battery) * moved
        costs = costs + weights.battery_shortfall * shortfall
        violations = violations | ((shortfall > 0) & moved)
    return dict(cost=costs, final_battery=battery, full_battery=full_battery, any_threshold_violation=violations)

def operating_cost_per_minute(is_uav: bool) -> float:
    return ALPHA_1 if is_uav else ALPHA_2


def replay_mandatory_trajectory(batch, tours, times, num_uav, num_depots,
                                weights=None, initial_visited=None,
                                active_request_mask=None, start_time=None,
                                start_battery=None, start_capacity=None,
                                allow_partial=False):
    """Replay the five raw-arrival costs, resources, carriers and native quotas.

    Partial residuals return prefix costs plus an explicit completion flag, not
    a finite complete-episode valuation. This reuses the transition physics;
    it is not independent physical calibration or an optimality certificate.
    """
    import torch
    from model_config import UAV_SWAP_MIN, ADR_SWAP_MIN
    from vrpUpdate import _pickup_speed, V_UAV_MAX, V_ADR_MAX, V_ADR_MIN_PICKUP
    config = native_physical_configuration(batch)
    weights = mandatory_objective_weights(weights if weights is not None else batch.get('objective_weights'))
    resources = trajectory_battery_cost(batch, tours, times, num_uav, num_depots, weights,
                                       start_time, start_battery, start_capacity)
    B, A, S = tours.shape
    demand = batch['demand'].reshape(B, -1)
    N = demand.shape[1]
    if num_depots < 1 or N < num_depots or (N - num_depots) % 2:
        raise ValueError('Invalid paired pickup/delivery layout')
    R = (N - num_depots) // 2
    delivery_start = num_depots + R
    pickup_load = demand[:, num_depots:delivery_start]
    if not torch.isfinite(demand).all() or not torch.allclose(
            pickup_load, -demand[:, delivery_start:], atol=1e-5, rtol=1e-5):
        raise ValueError('Pickup/delivery demands must be positive matched pairs')
    iv = initial_visited if initial_visited is not None else batch.get('initial_visited')
    iv = torch.zeros(B, N, device=tours.device) if iv is None else torch.as_tensor(iv, device=tours.device).reshape(B, N)
    if not torch.isfinite(iv).all() or ((iv != 0) & (iv != 1)).any():
        raise ValueError('Invalid initial visit state')
    picked, delivered = iv[:, num_depots:delivery_start].bool().clone(), iv[:, delivery_start:].bool().clone()
    if (delivered & ~picked).any():
        raise ValueError('Initial delivery cannot precede pickup')
    active = (~delivered if active_request_mask is None else
              torch.as_tensor(active_request_mask, device=tours.device).bool().reshape(B, R))
    if ((pickup_load <= 0) & active).any():
        raise ValueError('Active pickup/delivery demands must be positive matched pairs')
    owner = batch.get('delivery_owner')
    owner = (torch.full((B, R), -1, device=tours.device, dtype=torch.long) if owner is None else
             torch.as_tensor(owner, device=tours.device))
    if tuple(owner.shape) != (B, R) or not torch.isfinite(owner).all() or (owner != owner.long()).any():
        raise ValueError('Invalid delivery carrier context')
    owner = owner.long().clone()
    if ((owner < -1) | (owner >= A)).any() or (picked & ~delivered & (owner < 0)).any():
        raise ValueError('Onboard requests require a valid immutable delivery carrier')
    available = torch.as_tensor(batch.get('available_mask', torch.ones(B, A)), device=tours.device).bool().reshape(B, A)
    cap = torch.as_tensor(start_capacity if start_capacity is not None else batch.get('start_capacity', batch['capacity']),
                          device=tours.device, dtype=times.dtype).reshape(B, A)
    full_cap = batch['capacity'].reshape(B, A).to(cap)
    for agent in range(A):
        known_load = (pickup_load.to(cap) * (picked & ~delivered & (owner == agent))).sum(dim=1)
        if ((known_load - (full_cap[:, agent] - cap[:, agent])).abs()[available[:, agent]] > 1e-4).any():
            raise ValueError('Onboard carrier ledger disagrees with initial load')
    sharing = TensorSharingState(batch, B, R, num_uav, A, tours.device)
    tw = batch['time_window'].reshape(B, N).to(times)
    if not torch.isfinite(tw).all():
        raise ValueError('Non-finite request times')
    t0 = start_time if start_time is not None else batch.get('start_time', times.new_zeros(B, A))
    now = torch.as_tensor(t0, device=tours.device, dtype=times.dtype).reshape(B, A)
    components = {key: times.new_zeros(B, A) for key in
                  ('travel', 'recharge_dwell_cost', 'pickup_early_cost', 'pickup_late_cost',
                   'delivery_late_cost')}
    charge_dwell = bool(batch.get('recharge_cost_includes_dwell', True))
    rows = torch.arange(B, device=tours.device)
    matrices = [batch['edge_attr_d'].reshape(B, N, N), batch['edge_attr_r'].reshape(B, N, N)]
    for step in range(1, S):
        prev, current = tours[:, :, step - 1], tours[:, :, step]
        moved = prev != current
        for agent in range(A):
            is_uav = agent < num_uav
            mode = 0 if is_uav else 1
            node = current[:, agent]
            move = moved[:, agent]
            pickup = move & (node >= num_depots) & (node < delivery_start)
            delivery = move & (node >= delivery_start)
            customer = pickup | delivery
            distance = matrices[mode][rows, prev[:, agent], node].to(times)
            distance = torch.where(move, distance, 0.)
            vmax = V_UAV_MAX if is_uav else V_ADR_MAX
            speed = torch.full_like(distance, vmax)
            if not is_uav:
                speed = torch.where(pickup, _pickup_speed(distance, now[:, agent], tw[rows, node],
                                    speed.new_tensor(V_ADR_MIN_PICKUP), speed.new_tensor(vmax)), speed)
                speed = torch.where(node < num_depots, speed / 2., speed)
            travel = distance / speed
            arrival = now[:, agent] + travel
            if config is not None and is_uav:
                arrival = arrival + config.uav_phases.depot_takeoff_min * (move & (prev[:, agent] < num_depots))
            operating_rate = weights.operating('uav' if is_uav else 'adr')
            components['travel'][:, agent] += operating_rate * travel
            if charge_dwell:
                dwell_cost = operating_rate * (UAV_SWAP_MIN if is_uav else ADR_SWAP_MIN)
                dwell_cost = dwell_cost * (move & (node < num_depots))
                components['travel'][:, agent] += dwell_cost
                components['recharge_dwell_cost'][:, agent] += dwell_cost
            mask_key = 'mask_adjacency_uav' if is_uav else 'mask_adjacency_adr'
            if mask_key in batch:
                access = batch[mask_key].reshape(B, N, N)[rows, prev[:, agent], node] > .5
                if (move & ~access).any():
                    raise ValueError('Route violates physical/mode adjacency permissions')
            if 'agent_depot_allowed' in batch:
                depot = move & (node < num_depots)
                allowed = batch['agent_depot_allowed'][rows, agent, node.clamp(max=num_depots - 1)]
                if (depot & ~allowed.bool()).any():
                    raise ValueError('Route visits a depot forbidden to this vehicle')
            if 'x' in batch and (customer & (batch['x'][rows, node, 8 + mode] <= .5)).any():
                raise ValueError('Route violates node mode permissions')
            req = torch.where(delivery, node - delivery_start, node - num_depots).clamp(0, max(R - 1, 0))
            if R == 0:
                continue
            if (customer & ~active[rows, req]).any():
                raise ValueError('Route visits an inactive or already completed request')
            if (pickup & (picked[rows, req] | (owner[rows, req] >= 0))).any():
                raise ValueError('Duplicate pickup or immutable carrier overwrite in mandatory route')
            if (delivery & (~picked[rows, req] | delivered[rows, req] | (owner[rows, req] != agent))).any():
                raise ValueError('Delivery violates precedence, uniqueness or same-carrier pairing')
            if sharing.enabled:
                sharing.reserve(req, customer, agent)
            picked[rows[pickup], req[pickup]] = True
            owner[rows[pickup], req[pickup]] = agent
            delivered[rows[delivery], req[delivery]] = True
            delta = arrival - tw[rows, node]
            components['pickup_early_cost'][:, agent] += weights.pickup_early * torch.relu(-delta) * pickup
            components['pickup_late_cost'][:, agent] += weights.pickup_late * torch.relu(delta) * pickup
            components['delivery_late_cost'][:, agent] += weights.delivery_late * torch.relu(delta) * delivery
        now = times[:, :, step - 1]
    complete = (~active | delivered).all(dim=1)
    if not allow_partial and not complete.all():
        raise IncompleteTensorRouteError('Mandatory decoder stopped/truncated before every active delivery; no finite episode cost')
    components['tw_penalty'] = sum(components[key] for key in
                                   ('pickup_early_cost', 'pickup_late_cost', 'delivery_late_cost'))
    for key in ('undeliv', 'delivery_credit', 'battery'):
        components[key] = times.new_zeros(B, A)
    return dict(resources, components=components,
                prefix_cost=components['travel'] + components['tw_penalty'], complete=complete,
                unassigned_count=(active & ~delivered).sum(dim=1), delivery_owner=owner,
                objective_version=MANDATORY_OBJECTIVE_VERSION,
                valuation_scope='residual_prefix' if allow_partial else 'complete_episode')

def battery_penalty(remaining: float, min_threshold: float) -> float:
    return LAMBDA_BATTERY * max(min_threshold - remaining, 0.0)


@dataclass(frozen=True)
class MandatoryTrainingAids:
    """Optional training-only shaping and Lagrangian terms.

    Both coefficients default to zero so plain mandatory five-cost training is
    unaffected until a config or CLI knob explicitly enables one. Reported
    scoring (``score_route_plan``, ``replay_mandatory_trajectory``, every
    audit/CSV path) always uses only the five-cost objective; aids never leak
    into reported numbers. The aids are not part of ``ObjectiveWeights``.

    Rationale:

    - ``delivery_shaping_k`` activates a potential-based reward shaping (PBRS)
      term of the form ``F(s, s') = gamma * Phi(s') - Phi(s)`` with potential
      ``Phi(s) = -k * unserved_count(s)``. Under PBRS, [Ng, Harada and Russell
      1999] prove that the set of optimal policies is unchanged. Keep ``gamma``
      equal to the training discount factor so the invariance holds; verify
      episodic telescoping sums to a constant per episode.
    - ``battery_lagrangian_lambda`` activates a per-step soft penalty
      ``-lambda * max(0, reserve - SOC)`` applied only during training return
      calculation, following Reward Constrained Policy Optimization [Tessler,
      Mankowitz and Mannor 2019]. Physical battery feasibility is still
      enforced as hard masking in both training and evaluation; the penalty
      only helps credit assignment toward safer margins when plain-reward
      training fails to converge. ``lambda_schedule`` picks constant,
      ``linear`` (ramps from 0 to ``lambda`` across ``schedule_steps``) or
      ``dual`` (updated via dual gradient ascent on the measured constraint
      slack; the training loop supplies the update).

    This dataclass is deliberately kept narrow and configurable. Either knob
    can be retuned, replaced by a different potential, or disabled entirely in
    future work without touching reported numbers. If a change does affect the
    reported objective it stops being a training aid and should move into
    ``ObjectiveWeights``.
    """
    delivery_shaping_k: float = TRAINING_DELIVERY_SHAPING_K
    battery_lagrangian_lambda: float = TRAINING_BATTERY_LAGRANGIAN_LAMBDA
    lambda_schedule: str = 'constant'
    schedule_steps: int = 0

    def __post_init__(self):
        if any(not math.isfinite(value) or value < 0
               for value in (self.delivery_shaping_k, self.battery_lagrangian_lambda)):
            raise ValueError('Training aid coefficients must be finite and nonnegative')
        if self.lambda_schedule not in ('constant', 'linear', 'dual'):
            raise ValueError('Unknown lambda_schedule; expected constant, linear or dual')
        if self.lambda_schedule == 'linear' and self.schedule_steps <= 0:
            raise ValueError('linear lambda schedule requires schedule_steps > 0')

    @property
    def enabled(self) -> bool:
        return self.delivery_shaping_k > 0.0 or self.battery_lagrangian_lambda > 0.0

    def resolved_lambda(self, step: int = 0, dual_lambda: float = None) -> float:
        """Current lambda for the Lagrangian term; call from the training loop."""
        if self.battery_lagrangian_lambda == 0.0:
            return 0.0
        if self.lambda_schedule == 'constant':
            return self.battery_lagrangian_lambda
        if self.lambda_schedule == 'linear':
            fraction = min(max(step / float(self.schedule_steps), 0.0), 1.0)
            return fraction * self.battery_lagrangian_lambda
        if self.lambda_schedule == 'dual':
            if dual_lambda is None:
                raise ValueError('dual lambda schedule requires the current dual_lambda value')
            if not math.isfinite(dual_lambda) or dual_lambda < 0:
                raise ValueError('dual_lambda must be finite and nonnegative')
            return min(dual_lambda, self.battery_lagrangian_lambda)
        raise AssertionError('unreachable schedule branch')


def training_aids(value=None) -> MandatoryTrainingAids:
    """Build training aids from a dict or an existing dataclass; defaults disabled."""
    if value is None:
        return MandatoryTrainingAids()
    if isinstance(value, MandatoryTrainingAids):
        return value
    if isinstance(value, dict):
        return MandatoryTrainingAids(**value)
    raise TypeError('training_aids expects None, MandatoryTrainingAids or dict')


def pbrs_delivery_shaping_step(prev_unserved: int, next_unserved: int,
                               gamma: float, aids: MandatoryTrainingAids) -> float:
    """One-step PBRS term for the delivery-count potential; returns 0 if disabled.

    ``F = gamma * Phi(s') - Phi(s)`` with ``Phi(s) = -k * unserved_count(s)``.
    Telescoping over an episode leaves ``Phi(s_T) - Phi(s_0) = +k * delivered``
    under mandatory service, so the training-time contribution is constant per
    completed episode and policy-invariant.
    """
    if aids.delivery_shaping_k == 0.0:
        return 0.0
    if prev_unserved < 0 or next_unserved < 0:
        raise ValueError('unserved counts must be nonnegative')
    if not math.isfinite(gamma) or not 0.0 <= gamma <= 1.0:
        raise ValueError('gamma must be in [0, 1]')
    phi_prev = -aids.delivery_shaping_k * prev_unserved
    phi_next = -aids.delivery_shaping_k * next_unserved
    return gamma * phi_next - phi_prev


def lagrangian_battery_shortfall_step(battery: float, reserve: float,
                                      aids: MandatoryTrainingAids,
                                      step: int = 0, dual_lambda: float = None) -> float:
    """Per-step Lagrangian soft penalty for training only; 0 if disabled.

    ``penalty = lambda * max(0, reserve - battery)``, subtracted from the
    training return. Hard masking still enforces the physical constraint; this
    only helps credit assignment during training. Returns a nonnegative scalar
    (the training loop subtracts it from the return).
    """
    lam = aids.resolved_lambda(step=step, dual_lambda=dual_lambda)
    if lam == 0.0:
        return 0.0
    if not math.isfinite(battery) or not math.isfinite(reserve):
        raise ValueError('battery and reserve must be finite')
    shortfall = max(reserve - battery, 0.0)
    return lam * shortfall


def training_augmented_return(base_return: float, pbrs_sum: float,
                              lagrangian_sum: float) -> float:
    """Compose base five-cost return with optional training aids.

    This helper is for the training loop only. Report ``base_return`` from
    evaluation; never report the augmented value. PBRS adds (bonus for
    completing orders); Lagrangian subtracts (penalty for undercharging).
    """
    if not all(math.isfinite(value) for value in (base_return, pbrs_sum, lagrangian_sum)):
        raise ValueError('return components must be finite')
    if lagrangian_sum < 0:
        raise ValueError('lagrangian_sum must be nonnegative; subtracted inside this helper')
    return base_return + pbrs_sum - lagrangian_sum
