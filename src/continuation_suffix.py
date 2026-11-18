"""Check same-vehicle delivery continuations with the routing transition model."""

import torch


def _delivery_suffix(
    *, group_batch, nodes, charge, capacity, budget, jobs, outstanding,
    full_charge, full_capacity, reserve, depots, demand, time_window,
    distance_uav, distance_adr, adjacency, access, is_uav, config, wind, coordinates,
):
    from vrpUpdate import _leg_metrics

    groups, job_slots = jobs.shape
    depot_count = depots.size(1)
    options = torch.cat((jobs.clamp_min(depot_count), torch.arange(
        depot_count, device=jobs.device).expand(groups, -1)), dim=1)
    possible = torch.isfinite(charge).unsqueeze(1)
    cargo = (jobs >= 0).unsqueeze(1)
    nodes, charge, capacity, budget = (value.unsqueeze(1)
                                       for value in (nodes, charge, capacity, budget))
    completed = torch.zeros(groups, dtype=torch.bool, device=jobs.device)
    initial_count = cargo.sum(-1).squeeze(1)
    # Keep several feasible prefixes and a separate depot branch where possible.
    # This is a sufficient certificate: each surviving witness clears its cargo.
    width = min(4, job_slots + depot_count)
    for _ in range(2 * job_slots + 1):
        group, parent = (possible & ~completed.unsqueeze(1)).nonzero(as_tuple=True)
        if group.numel() == 0:
            break
        batch = group_batch[group]
        source = nodes[group, parent]
        targets = options[group]
        cargo_left = cargo[group, parent]
        next_cargo_count = cargo_left.sum(1, keepdim=True)
        budget_here = budget[group, parent]
        allowed = torch.cat((
            cargo_left,
            depots[group] & (budget_here > 0).unsqueeze(1) & (source >= depot_count).unsqueeze(1),
        ), dim=1)
        is_depot = targets < depot_count
        terminal = (~is_depot & (next_cargo_count == 1)
                    & ((outstanding[group] - initial_count[group]).unsqueeze(1) == 0))
        cap_here = capacity[group, parent]
        energy, _, reachable = _leg_metrics(
            demand[batch], time_window[batch], capacity.new_zeros((len(group), 1, 1)),
            cap_here[:, None, None], int(is_uav), source[:, None], targets.unsqueeze(1),
            distance_uav, distance_adr, depot_count, full_capacity[group, None],
            None if wind is None else wind[batch], None if coordinates is None else coordinates[batch],
            config, matrix_batch=batch, terminal_delivery=terminal.unsqueeze(1),
        )
        next_charge = charge[group, parent, None] - energy.squeeze(1)
        allowed &= reachable.squeeze(1) & (next_charge >= reserve[group, None] - 1e-5)
        if adjacency is not None:
            allowed &= adjacency[batch[:, None], source[:, None], targets] >= .5
        if access is not None:
            allowed &= access[batch[:, None], targets] >= .5
        completed[group[(allowed & ~is_depot & (next_cargo_count == 1)).any(1)]] = True

        allowed &= ~completed[group, None]
        next_charge = torch.where(is_depot, full_charge[group, None], next_charge)
        next_budget = budget_here[:, None] - is_depot.long()
        next_count = next_cargo_count - (~is_depot).long()
        score = (-2 * next_count + next_charge / full_charge[group, None]
                 - .1 * is_depot - .001 * energy.squeeze(1) / full_charge[group, None])
        score = score.masked_fill(~allowed, -torch.inf)
        option_count = targets.size(1)
        keep_count = min(width, possible.size(1) * option_count)
        scores = charge.new_full((groups, possible.size(1), option_count), -torch.inf)
        scores[group, parent] = score
        scores = scores.flatten(1)
        ranked = scores.topk(keep_count, dim=1)
        chosen = ranked.indices
        if keep_count > 1 and depot_count:
            depot_slots = torch.arange(scores.size(1), device=jobs.device) % option_count >= job_slots
            depot_pick = scores.masked_fill(~depot_slots, -torch.inf).argmax(1)
            use_depot = (scores.gather(1, depot_pick[:, None]).isfinite().squeeze(1)
                         & ~(chosen == depot_pick[:, None]).any(1))
            chosen[:, -1] = torch.where(use_depot, depot_pick, chosen[:, -1])
        possible = scores.gather(1, chosen).isfinite()
        prior, action = chosen // option_count, chosen % option_count
        row_for_parent = torch.full((groups, cargo.size(1)), -1, device=jobs.device, dtype=torch.long)
        row_for_parent[group, parent] = torch.arange(len(group), device=jobs.device)
        source_row = row_for_parent.gather(1, prior).clamp_min(0)
        nodes = targets[source_row, action]
        charge = next_charge[source_row, action]
        budget = next_budget[source_row, action]
        capacity = cap_here[source_row] - torch.where(
            nodes < depot_count, 0., demand[group_batch[:, None], nodes])
        cargo = cargo_left[source_row].clone()
        cargo &= torch.arange(job_slots, device=jobs.device) != action.unsqueeze(-1)
    return completed


def certify_onboard_moves(
    blocked, onboard, demand, time_window, capacity, current, battery, energy,
    num_uav, depot_count, full_battery, reserve, max_capacity, config,
    distance_uav, distance_adr, wind, coordinates, remaining_deliveries,
    allowed_depots, recharge_budget, available, adjacency_uav, adjacency_adr, access_uav, access_adr,
):
    from vrpUpdate import _fleet_values, _pending_deliveries

    batches, agents, n_nodes = blocked.shape
    requests = onboard.size(-1)
    delivery_start = depot_count + requests
    if requests == 0:
        return ~blocked
    device = demand.device
    options = torch.arange(n_nodes, device=device)
    pickup = (options >= depot_count) & (options < delivery_start)
    delivery = options >= delivery_start
    passed = ~blocked
    passed[:, :, delivery_start:] &= onboard
    first_allowed = passed & (options != current.unsqueeze(-1))
    if available is not None:
        first_allowed &= available.bool().unsqueeze(-1)
    count = onboard.sum(-1, keepdim=True) + pickup.long() - delivery.long()
    batch_ids, agent_ids, node_ids = (first_allowed & (count > 0)).nonzero(as_tuple=True)
    if batch_ids.numel() == 0:
        return passed
    cap = capacity.reshape(batches, agents)
    full_cap = _fleet_values(max_capacity, cap, num_uav)
    bat = battery.reshape(batches, agents)
    pending_count = _pending_deliveries(demand, depot_count, remaining_deliveries).sum(1)
    for is_uav, adjacency, access in ((True, adjacency_uav, access_uav), (False, adjacency_adr, access_adr)):
        group_ids = ((agent_ids < num_uav) == is_uav).nonzero(as_tuple=True)[0]
        for group in group_ids.split(2048):
            if group.numel() == 0:
                continue
            b, a, node = batch_ids[group], agent_ids[group], node_ids[group]
            obligations = onboard[b, a].clone()
            row = torch.arange(len(group), device=device)
            is_pickup = pickup[node]
            is_delivery = delivery[node]
            obligations[row[is_pickup], node[is_pickup] - depot_count] = True
            obligations[row[is_delivery], node[is_delivery] - delivery_start] = False
            width = int(obligations.sum(1).max().item())
            jobs = (torch.arange(requests, device=device) + delivery_start).expand_as(obligations)
            jobs = jobs.masked_fill(~obligations, -1).topk(width, dim=1).values
            depot = node < depot_count
            charge = torch.where(depot, full_battery[b, a], bat[b, a] - energy[b, a, node])
            left = (recharge_budget[b, a] if recharge_budget is not None
                    else torch.full_like(node, width + 1))
            depots = (allowed_depots[b, a] if allowed_depots is not None
                      else torch.ones(len(b), depot_count, device=device, dtype=torch.bool))
            passed[b, a, node] = _delivery_suffix(
                group_batch=b, nodes=node, charge=charge,
                capacity=cap[b, a] - torch.where(depot, 0., demand[b, node]),
                budget=left - depot.long(), jobs=jobs, outstanding=pending_count[b] - is_delivery.long(),
                full_charge=full_battery[b, a], full_capacity=full_cap[b, a],
                reserve=reserve[b, a, 0], depots=depots, demand=demand, time_window=time_window,
                distance_uav=distance_uav, distance_adr=distance_adr,
                adjacency=None if adjacency is None else adjacency.reshape(batches, n_nodes, n_nodes),
                access=None if access is None else access.reshape(batches, n_nodes),
                is_uav=is_uav, config=config, wind=wind, coordinates=coordinates,
            )
    return passed
