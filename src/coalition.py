import os
import hashlib
import json
import numpy as np
from typing import Dict, List, Any, Optional
from dispatch_sim import ALPHA_U, RollingHorizonDispatcher, Vehicle
from creat_vrp import create_instance
from coalition_contract import SharingContract, COALITION_VERSION
from dataclasses import asdict
from reward import OBJECTIVE_VERSION, objective_weights
from model_config import (MODEL_VERSION, ENERGY_MODEL_VERSION, GENERATOR_VERSION,
    UAV_BATTERY_KJ, ADR_BATTERY_KJ, UAV_CAPACITY_KG, ADR_CAPACITY_KG,
    UAV_PROFILE, ADR_PROFILE)

def _episode_total_cost(episode_log: List[Dict]) -> float:
    summary = next((e for e in episode_log if e.get('summary')), None)
    if summary is None or not np.isfinite(summary.get('total_cost', np.nan)):
        raise ValueError('Missing or non-finite episode cost')
    if summary.get('service_policy') == 'mandatory':
        if (not summary.get('service_complete') or summary.get('undelivered') != 0
                or summary.get('total_delivered') != summary.get('total_orders')):
            raise ValueError('Incomplete mandatory service is not a finite coalition value')
        value = summary.get('quality_cost', np.nan)
        if not np.isfinite(value):
            raise ValueError('Missing or non-finite five-term routing cost')
        return float(value)
    return float(summary['total_cost'])

def _episode_operating_split(episode_log: List[Dict]) -> tuple:
    summary = next((e for e in episode_log if e.get('summary')), {})
    op_uav = float(summary.get('operating_cost_uav', 0.0))
    op_adr = float(summary.get('operating_cost_adr', 0.0))
    return op_uav, op_adr

def _as_numpy_1d(value):
    return value.cpu().numpy().reshape(-1) if hasattr(value, 'cpu') else np.asarray(value).reshape(-1)

def arrivals_from_instance(full_instance: Dict[str, Any]) -> List[Dict[str, Any]]:
    n_depots = int(full_instance['n_depots'])
    n_req = int(full_instance['n_req'])
    tw = _as_numpy_1d(full_instance['time_window'])
    demand = _as_numpy_1d(full_instance['demand'])
    if 't_arrival' in full_instance:
        t_arrival = _as_numpy_1d(full_instance['t_arrival'])
    else:
        t_arrival = np.zeros(n_req, dtype=np.float32)

    arrivals = []
    for rid in range(n_req):
        pickup_node = n_depots + rid
        delivery_node = n_depots + n_req + rid
        arrivals.append({
            'req_id': rid,
            't_arrival': float(t_arrival[rid]),
            't_pickup': float(tw[pickup_node]),
            't_delivery': float(tw[delivery_node]),
            'pickup_node': pickup_node,
            'delivery_node': delivery_node,
            'demand': float(abs(demand[pickup_node])),
        })
    return sorted(arrivals, key=lambda row: row['t_arrival'])

def _episode_undelivered(episode_log: List[Dict]) -> int:
    summary = next((e for e in episode_log if e.get('summary')), {})
    return int(summary.get('undelivered', 0))

def _episode_late_pct(episode_log: List[Dict]) -> float:
    summary = next((e for e in episode_log if e.get('summary')), {})
    return float(summary.get('tw_delivery_violation_rate', 0.0))

def _episode_served_frac(episode_log: List[Dict]) -> float:
    summary = next((e for e in episode_log if e.get('summary')), {})
    revealed = int(summary.get('total_revealed', 0))
    delivered = int(summary.get('total_delivered', 0))
    return delivered / max(revealed, 1)

def _assign_market(arrivals: List[Dict], uav_frac: float, seed: int) -> List[str]:
    rng = np.random.default_rng(seed)
    return ['uav' if rng.random() < uav_frac else 'adr' for _ in arrivals]

def _allocation_metrics(costs_uav, costs_adr, costs_coal, result):
    try:
        from allocation import per_episode_analysis, conviction_score
    except ImportError as exc:
        result['allocation_metrics_available'] = False
        result['allocation_import_error'] = str(exc)
        result['conviction'] = float('nan')
        return result

    result.update(per_episode_analysis(costs_uav, costs_adr, costs_coal))
    result['conviction'] = conviction_score(result)
    result['allocation_metrics_available'] = True
    return result

def summarise(
    uav_vals: List[float],
    adr_vals: List[float],
    coalition_vals: List[float],
    empty_vals: List[float] = None,
    revenue_uav: float = 0.0,
    revenue_adr: float = 0.0,
    revenue_uav_coal: Optional[float] = None,
    revenue_adr_coal: Optional[float] = None,
) -> Dict[str, Any]:
    u = np.array(uav_vals)
    a = np.array(adr_vals)
    c = np.array(coalition_vals)
    e = np.array(empty_vals) if empty_vals is not None else None
    savings = u + a - c
    total_standalone = u.mean() + a.mean()
    uav_share = u.mean() / max(total_standalone, 1e-9)
    adr_share = a.mean() / max(total_standalone, 1e-9)
    sh_uav_cost = (u.mean() + c.mean() - a.mean()) / 2.0
    sh_adr_cost = (a.mean() + c.mean() - u.mean()) / 2.0
    out = {
        'phi_uav_mean': float(u.mean()),
        'phi_uav_std': float(u.std()),
        'phi_adr_mean': float(a.mean()),
        'phi_adr_std': float(a.std()),
        'phi_coalition_mean': float(c.mean()),
        'phi_coalition_std': float(c.std()),
        'coalition_gain_mean': float(savings.mean()),
        'coalition_gain_std': float(savings.std()),
        'core_exists': bool(savings.mean() >= -1e-9),
        'core_exists_scope': 'mean dispatcher-induced two-firm cost game',
        'core_freq': float((savings >= -1e-9).mean()),
        'uav_benefit_mean': float(u.mean() - sh_uav_cost),
        'adr_benefit_mean': float(a.mean() - sh_adr_cost),
        'uav_profit_standalone': float(revenue_uav - u.mean()),
        'adr_profit_standalone': float(revenue_adr - a.mean()),
        'uav_profit_coalition': float((revenue_uav if revenue_uav_coal is None else revenue_uav_coal) - sh_uav_cost),
        'adr_profit_coalition': float((revenue_adr if revenue_adr_coal is None else revenue_adr_coal) - sh_adr_cost),
    }
    if e is not None:
        out.update({
            'phi_empty_mean': float(e.mean()),
            'phi_empty_std': float(e.std()),
        })
    return out

def make_fleet(
    n_uav: int,
    n_adr: int,
    n_depots_uav: int,
    n_depots_adr: int,
    battery_uav: float = UAV_BATTERY_KJ,
    battery_adr: float = ADR_BATTERY_KJ,
    q_uav: float = UAV_CAPACITY_KG,
    q_adr: float = ADR_CAPACITY_KG,
    full_instance: Optional[Dict[str, Any]] = None,
    depot_shared: Optional[bool] = None,
) -> Dict[int, Vehicle]:
    vehicles: Dict[int, Vehicle] = {}

    for i in range(n_uav):
        start_node = i % max(n_depots_uav, 1)
        vehicles[i] = Vehicle(
            vehicle_id=i,
            mode='uav',
            current_node=start_node,
            current_time=0.0,
            battery=battery_uav,
            load=0.0,
            capacity=q_uav,
            battery_init=battery_uav,
        )

    for i in range(n_adr):
        vid = n_uav + i
        start_node = n_depots_uav + (i % max(n_depots_adr, 1))
        vehicles[vid] = Vehicle(
            vehicle_id=vid,
            mode='adr',
            current_node=start_node,
            current_time=0.0,
            battery=battery_adr,
            load=0.0,
            capacity=q_adr,
            battery_init=battery_adr,
        )

    if full_instance is not None:
        import numpy as np
        def values(key):
            value = full_instance[key]
            if hasattr(value, 'detach'):
                value = value.detach().cpu().numpy()
            return np.asarray(value, dtype=float).reshape(-1)
        packs, limits = values('battery'), values('capacity')
        charge = values('start_battery') if 'start_battery' in full_instance else packs
        if len(packs) != len(vehicles) or len(limits) != len(vehicles) or len(charge) != len(vehicles):
            raise ValueError('Fleet size does not match instance resources')
        if not np.isfinite(np.concatenate([packs, limits, charge])).all():
            raise ValueError('Non-finite fleet resources')
        if (packs <= 0).any() or (limits <= 0).any() or (charge < 0).any() or (charge > packs).any():
            raise ValueError('Invalid pack capacity, charge or payload capacity')
        for vid, veh in vehicles.items():
            veh.battery_init = float(packs[vid])
            veh.battery = float(charge[vid])
            veh.capacity = float(limits[vid])
    if depot_shared is not None:
        n_depots = n_depots_uav + n_depots_adr
        x = full_instance.get('x') if full_instance is not None else None
        if x is not None:
            x = x.detach().cpu().numpy() if hasattr(x, 'detach') else np.asarray(x)
        for veh in vehicles.values():
            own = range(n_depots_uav) if veh.mode == 'uav' else range(n_depots_uav, n_depots)
            candidates = range(n_depots) if depot_shared else own
            column = 8 if veh.mode == 'uav' else 9
            veh.allowed_depots = tuple(d for d in candidates if x is None or x[d, column] > 0.5)
    return vehicles

def coalition_cost_for_config(
    solver,
    n_episodes: int,
    market_split: Dict[str, float],
    eps_shared: float,
    depot_shared: bool,
    alpha_asymmetry: float,
    delta_minutes: float = 10.0,
    shift_minutes: float = 120.0,
    peak_rate: float = 20.0,
    n_uav: int = 3,
    n_adr: int = 3,
    n_req: int = 25,
    seed_base: int = 0,
    charging_cost_uav: float = 0.0,
    charging_cost_adr: float = 0.0,
    revenue_per_req_uav: float = 0.0,
    revenue_per_req_adr: float = 0.0,
    use_real_world: bool = False,
    networks=None,
    valuation_mode: str = 'market_books',
    wind_magnitude: float = 0.0,
    battery_uav: float = UAV_BATTERY_KJ,
    battery_adr: float = ADR_BATTERY_KJ,
    demand_range: tuple = (1.0, 10.0),
    demand_heavy_tail: bool = False,
    q_uav: float = UAV_CAPACITY_KG,
    return_episodes: bool = False,
    eps_uav_to_adr: float = 0.0,
    eps_adr_to_uav: float = 0.0,
    market_group_by_request: Optional[Dict[int, str]] = None,
    joint_infrastructure_cost: Optional[float] = None,
):
    if valuation_mode != 'market_books':
        raise ValueError('masked_full_stream is not an owned-book coalition game; use market_books')
    uav_frac = float(market_split.get('uav', 0.5))
    adr_frac = float(market_split.get('adr', 1.0 - uav_frac))
    if not np.isfinite([uav_frac, adr_frac]).all() or not 0 <= uav_frac <= 1 or abs(uav_frac + adr_frac - 1) > 1e-9:
        raise ValueError('Market fractions must be nonnegative and sum to one')
    if n_episodes <= 0 or n_req <= 0 or min(n_uav, n_adr) < 1:
        raise ValueError('Positive episodes, requests and both firm fleets are required')
    resources = [battery_uav, battery_adr, q_uav, delta_minutes, shift_minutes]
    if not np.isfinite(resources).all() or min(resources) <= 0:
        raise ValueError('Invalid resources or time horizon')
    fees = [charging_cost_uav, charging_cost_adr]
    if joint_infrastructure_cost is not None:
        fees.append(joint_infrastructure_cost)
    if not np.isfinite(fees).all() or min(fees) < 0:
        raise ValueError('Infrastructure costs must be explicit nonnegative costs')
    # No automatic min(singleton fees): depot access alone does not establish
    # infrastructure economies. An independently specified joint fee is needed.
    joint_fee = sum(fees[:2]) if joint_infrastructure_cost is None else joint_infrastructure_cost
    costs_uav, costs_adr, costs_coal = [], [], []
    costs_empty = []
    violations = []
    late_uav, late_adr, late_coal = [], [], []
    served_uav, served_adr, served_coal = [], [], []
    op_uav_coal, op_adr_coal = [], []
    policy_uav_share, policy_adr_share = [], []
    owned_uav, owned_adr = [], []
    revenues_uav, revenues_adr, revenues_joint_uav, revenues_joint_adr = [], [], [], []
    plan_projections = []
    _episodes: List[Dict] = [] if return_episodes else None  # type: ignore[assignment]
    uav_frac = market_split.get('uav', 0.5)

    if use_real_world:
        from mississauga_instance import (
            load_mississauga_networks,
            create_mississauga_instance,
        )
        _nets = networks if networks is not None else load_mississauga_networks()

    for ep in range(n_episodes):
        rng = np.random.default_rng(seed_base + ep)
        n_dep_uav = n_dep_adr = 1

        if use_real_world:
            instance = create_mississauga_instance(
                n_req=n_req,
                n_uav=n_uav,
                n_adr=n_adr,
                n_depots_uav=n_dep_uav,
                n_depots_adr=n_dep_adr,
                rng=rng,
                networks=_nets,
                wind_override=(wind_magnitude, 0.0),
            )
        else:
            instance = create_instance(
                n_req=n_req,
                n_uav=n_uav,
                n_adr=n_adr,
                n_depots_uav=n_dep_uav,
                n_depots_adr=n_dep_adr,
                rng=rng,
                demand_range=demand_range,
                demand_heavy_tail=demand_heavy_tail,
                wind_override=(wind_magnitude, 0.0),
            )

        instance['battery'] = instance['battery'].new_tensor([battery_uav] * n_uav + [battery_adr] * n_adr)
        instance['capacity'] = instance['capacity'].new_tensor([q_uav] * n_uav + [ADR_CAPACITY_KG] * n_adr)

        arrivals = arrivals_from_instance(instance)

        market_assign = _assign_market(arrivals, uav_frac, seed=seed_base + ep + 100_000)
        for row, owner in zip(arrivals, market_assign):
            row['firm_owner'] = owner
            row['market_group'] = (market_group_by_request or {}).get(row['req_id'], 'shared')
            row['allowed_modes'] = ('uav', 'adr')
        uav_ext = [a for a in arrivals if a['firm_owner'] == 'uav']
        adr_ext = [a for a in arrivals if a['firm_owner'] == 'adr']
        owned_uav.append(len(uav_ext))
        owned_adr.append(len(adr_ext))
        contracts = {
            'uav': SharingContract.from_arrivals(uav_ext, members=('uav',)),
            'adr': SharingContract.from_arrivals(adr_ext, members=('adr',)),
            'joint': SharingContract.from_arrivals(arrivals, eps_shared=eps_shared,
                eps_uav_to_adr=eps_uav_to_adr, eps_adr_to_uav=eps_adr_to_uav),
        }
        geometry = np.asarray(instance['coordinates']).tolist()
        geometry_hash = hashlib.sha256(json.dumps(geometry, separators=(',', ':')).encode()).hexdigest()
        book_hash = hashlib.sha256(json.dumps(arrivals, sort_keys=True, separators=(',', ':')).encode()).hexdigest()

        base_fleet = make_fleet(
            n_uav=n_uav,
            n_adr=n_adr,
            n_depots_uav=n_dep_uav,
            n_depots_adr=n_dep_adr,
            battery_uav=battery_uav,
            battery_adr=battery_adr,
            q_uav=q_uav,
            full_instance=instance,
            depot_shared=False,
        )

        uav_fleet = {k: v for k, v in base_fleet.items() if v.mode == 'uav'}
        uav_log = RollingHorizonDispatcher(
            solver, delta_minutes, shift_minutes, alpha_p_ratio=alpha_asymmetry
        ).run_shift(uav_ext, uav_fleet, instance, sharing_contract=contracts['uav'])
        phi_uav = _episode_total_cost(uav_log)

        adr_fleet = {k: v for k, v in base_fleet.items() if v.mode == 'adr'}
        adr_log = RollingHorizonDispatcher(
            solver, delta_minutes, shift_minutes, alpha_p_ratio=alpha_asymmetry
        ).run_shift(adr_ext, adr_fleet, instance, sharing_contract=contracts['adr'])
        phi_adr = _episode_total_cost(adr_log)

        joint_fleet = make_fleet(
            n_uav=n_uav,
            n_adr=n_adr,
            n_depots_uav=n_dep_uav,
            n_depots_adr=n_dep_adr,
            battery_uav=battery_uav,
            battery_adr=battery_adr,
            q_uav=q_uav,
            full_instance=instance,
            depot_shared=depot_shared,
        )
        joint_log = RollingHorizonDispatcher(
            solver, delta_minutes, shift_minutes, alpha_p_ratio=alpha_asymmetry
        ).run_shift(arrivals, joint_fleet, instance, sharing_contract=contracts['joint'])
        phi_coal = _episode_total_cost(joint_log) + joint_fee
        summaries = [next(e for e in log if e.get('summary')) for log in (uav_log, adr_log, joint_log)]
        plan_projections.append(sum(s['plan_projection_count'] for s in summaries))
        def delivered_count(summary, owner):
            return sum(r['status'] == 'delivered' and r['firm_owner'] == owner for r in summary['request_history'])
        revenues_uav.append(revenue_per_req_uav * delivered_count(summaries[0], 'uav'))
        revenues_adr.append(revenue_per_req_adr * delivered_count(summaries[1], 'adr'))
        revenues_joint_uav.append(revenue_per_req_uav * delivered_count(summaries[2], 'uav'))
        revenues_joint_adr.append(revenue_per_req_adr * delivered_count(summaries[2], 'adr'))
        joint_op_uav, joint_op_adr = _episode_operating_split(joint_log)
        joint_op_total = joint_op_uav + joint_op_adr
        if joint_op_total > 1e-9:
            uav_share_frac = joint_op_uav / joint_op_total
        else:
            uav_share_frac = 0.5
        policy_uav = phi_coal * uav_share_frac
        policy_adr = phi_coal * (1.0 - uav_share_frac)
        op_uav_coal.append(joint_op_uav)
        op_adr_coal.append(joint_op_adr)
        policy_uav_share.append(policy_uav)
        policy_adr_share.append(policy_adr)
        phi_empty = 0.0

        costs_uav.append(phi_uav + charging_cost_uav)
        costs_adr.append(phi_adr + charging_cost_adr)
        costs_empty.append(phi_empty)
        costs_coal.append(phi_coal)
        violations.append(_episode_undelivered(joint_log))

        late_uav.append(_episode_late_pct(uav_log))
        late_adr.append(_episode_late_pct(adr_log))
        late_coal.append(_episode_late_pct(joint_log))
        served_uav.append(_episode_served_frac(uav_log))
        served_adr.append(_episode_served_frac(adr_log))
        served_coal.append(_episode_served_frac(joint_log))

        if _episodes is not None:
            _episodes.append({
                'episode_idx':         ep,
                'episode_seed':        seed_base + ep,
                'geometry_sha256':     geometry_hash,
                'task_book_sha256':    book_hash,
                'coalition_version':   COALITION_VERSION,
                'n_owned_uav':         owned_uav[-1],
                'n_owned_adr':         owned_adr[-1],
                'sharing_budgets':     summaries[2]['sharing_budgets'],
                'sharing_usage':       summaries[2]['sharing_usage'],
                'plan_projection_count': plan_projections[-1],
                'phi_uav':             costs_uav[-1],
                'phi_adr':             costs_adr[-1],
                'phi_coal':            costs_coal[-1],
                'coalition_gain':      costs_uav[-1] + costs_adr[-1] - costs_coal[-1],
                'undelivered':         violations[-1],
                'late_uav_pct':        late_uav[-1],
                'late_adr_pct':        late_adr[-1],
                'late_coal_pct':       late_coal[-1],
                'served_uav_frac':     served_uav[-1],
                'served_adr_frac':     served_adr[-1],
                'served_coal_frac':    served_coal[-1],
                'op_uav_coal':         op_uav_coal[-1],
                'op_adr_coal':         op_adr_coal[-1],
                'policy_uav_share':    policy_uav_share[-1],
                'policy_adr_share':    policy_adr_share[-1],
                'policy_uav_gain':     costs_uav[-1] - policy_uav_share[-1],
                'policy_adr_gain':     costs_adr[-1] - policy_adr_share[-1],
            })

    result = summarise(
        costs_uav, costs_adr, costs_coal,
        empty_vals=costs_empty,
        revenue_uav=float(np.mean(revenues_uav)),
        revenue_adr=float(np.mean(revenues_adr)),
        revenue_uav_coal=float(np.mean(revenues_joint_uav)),
        revenue_adr_coal=float(np.mean(revenues_joint_adr)),
    )
    result['mean_undelivered'] = float(np.mean(violations))
    result['n_req'] = n_req

    result['uav_late_pct_mean']    = float(np.mean(late_uav))
    result['adr_late_pct_mean']    = float(np.mean(late_adr))
    result['coal_late_pct_mean']   = float(np.mean(late_coal))
    result['uav_served_frac_mean'] = float(np.mean(served_uav))
    result['adr_served_frac_mean'] = float(np.mean(served_adr))
    result['coal_served_frac_mean']= float(np.mean(served_coal))
    result['n_owned_uav_mean'] = float(np.mean(owned_uav))
    result['n_owned_adr_mean'] = float(np.mean(owned_adr))
    result['uav_cost_per_req']     = result['phi_uav_mean']       / np.mean(owned_uav) if np.mean(owned_uav) > 0 else float('nan')
    result['adr_cost_per_req']     = result['phi_adr_mean']       / np.mean(owned_adr) if np.mean(owned_adr) > 0 else float('nan')
    result['coal_cost_per_req']    = result['phi_coalition_mean'] / max(n_req, 1)

    result['op_uav_coal_mean']       = float(np.mean(op_uav_coal)) if op_uav_coal else 0.0
    result['op_adr_coal_mean']       = float(np.mean(op_adr_coal)) if op_adr_coal else 0.0
    result['policy_uav_share_mean']  = float(np.mean(policy_uav_share)) if policy_uav_share else 0.0
    result['policy_adr_share_mean']  = float(np.mean(policy_adr_share)) if policy_adr_share else 0.0
    result['policy_uav_gain_mean']   = result['phi_uav_mean'] - result['policy_uav_share_mean']
    result['policy_adr_gain_mean']   = result['phi_adr_mean'] - result['policy_adr_share_mean']

    joint_summary = next(row for row in joint_log if row.get('summary'))
    result['config'] = {
        'model_version': MODEL_VERSION,
        'energy_model_version': ENERGY_MODEL_VERSION,
        'generator_version': GENERATOR_VERSION,
        'objective_version': joint_summary['objective_version'],
        'objective_weights': joint_summary['objective_weights'],
        'service_policy': joint_summary['service_policy'],
        'delivery_penalty_clock': joint_summary['delivery_penalty_clock'],
        'uav_hardware_profile': UAV_PROFILE,
        'adr_hardware_profile': ADR_PROFILE,
        'market_split': market_split,
        'eps_shared': eps_shared,
        'depot_shared': depot_shared,
        'alpha_asymmetry': alpha_asymmetry,
        'delta_minutes': delta_minutes,
        'charging_cost_uav': charging_cost_uav,
        'charging_cost_adr': charging_cost_adr,
        'valuation_mode': valuation_mode,
        'wind_magnitude': wind_magnitude,
        'battery_uav': battery_uav,
        'battery_adr': battery_adr,
        'demand_max': demand_range[1],
        'demand_range': list(demand_range),
        'demand_heavy_tail': demand_heavy_tail,
        'q_uav': q_uav,
        'q_adr': ADR_CAPACITY_KG,
        'use_real_world': use_real_world,
        'revenue_per_req_uav': revenue_per_req_uav,
        'revenue_per_req_adr': revenue_per_req_adr,
        'coalition_version': COALITION_VERSION,
        'seed_base': seed_base,
        'n_episodes': n_episodes,
        'n_req': n_req,
        'n_uav': n_uav,
        'n_adr': n_adr,
        'shift_minutes': shift_minutes,
        'eps_uav_to_adr': eps_uav_to_adr,
        'eps_adr_to_uav': eps_adr_to_uav,
        'sharing_scope': 'fixed whole-book pickup-commitment budget',
        'market_group_by_request': market_group_by_request,
        'joint_infrastructure_cost': joint_fee,
    }
    result['coalition_version'] = COALITION_VERSION
    result['valuation_estimand'] = 'achieved dispatcher-induced cost on owned books'
    result['scientific_release_validated'] = False
    result['plan_projection_count'] = sum(plan_projections)

    result = _allocation_metrics(costs_uav, costs_adr, costs_coal, result)

    if return_episodes:
        return result, _episodes
    return result

def _run_one_config(args, return_episodes=False):
    (solver, n_episodes, ms, eps, ds, aa, cu, ca,
     delta_minutes, seed, n_req, n_uav, n_adr, valuation_mode,
     wind_magnitude, battery_uav, battery_adr) = args
    return coalition_cost_for_config(
        solver=solver,
        n_episodes=n_episodes,
        market_split=ms,
        eps_shared=eps,
        depot_shared=ds,
        alpha_asymmetry=aa,
        delta_minutes=delta_minutes,
        seed_base=seed,
        charging_cost_uav=cu,
        charging_cost_adr=ca,
        n_req=n_req,
        n_uav=n_uav,
        n_adr=n_adr,
        valuation_mode=valuation_mode,
        wind_magnitude=wind_magnitude,
        battery_uav=battery_uav,
        battery_adr=battery_adr,
        return_episodes=return_episodes,
    )


def _run_one_config_with_episodes(args):
    result, episodes = _run_one_config(args, return_episodes=True)
    for episode in episodes:
        episode['config_json'] = json.dumps(result['config'], sort_keys=True)
    return result, episodes

def _run_one_config_c(args):
    """Extended worker for complementarity sweep."""
    (solver, n_episodes, ms, eps, ds, aa,
     delta_minutes, seed, n_req, n_uav, n_adr, valuation_mode,
     wind_magnitude, battery_uav, battery_adr,
     demand_range, demand_heavy_tail, q_uav) = args
    return coalition_cost_for_config(
        solver=solver,
        n_episodes=n_episodes,
        market_split=ms,
        eps_shared=eps,
        depot_shared=ds,
        alpha_asymmetry=aa,
        delta_minutes=delta_minutes,
        seed_base=seed,
        n_req=n_req,
        n_uav=n_uav,
        n_adr=n_adr,
        valuation_mode=valuation_mode,
        wind_magnitude=wind_magnitude,
        battery_uav=battery_uav,
        battery_adr=battery_adr,
        demand_range=demand_range,
        demand_heavy_tail=demand_heavy_tail,
        q_uav=q_uav,
    )


def _run_one_config_c_with_episodes(args):
    """Like _run_one_config_c but returns (result, episodes).

    Args tuple is identical to _run_one_config_c PLUS combo_idx appended at
    the end. Each episode dict is tagged with combo_idx and key config params
    so the main process can write them to a per-episode CSV without needing
    to join on any separate metadata table.
    """
    (solver, n_episodes, ms, eps, ds, aa,
     delta_minutes, seed, n_req, n_uav, n_adr, valuation_mode,
     wind_magnitude, battery_uav, battery_adr,
     demand_range, demand_heavy_tail, q_uav, combo_idx) = args
    result, episodes = coalition_cost_for_config(
        solver=solver,
        n_episodes=n_episodes,
        market_split=ms,
        eps_shared=eps,
        depot_shared=ds,
        alpha_asymmetry=aa,
        delta_minutes=delta_minutes,
        seed_base=seed,
        n_req=n_req,
        n_uav=n_uav,
        n_adr=n_adr,
        valuation_mode=valuation_mode,
        wind_magnitude=wind_magnitude,
        battery_uav=battery_uav,
        battery_adr=battery_adr,
        demand_range=demand_range,
        demand_heavy_tail=demand_heavy_tail,
        q_uav=q_uav,
        return_episodes=True,
    )
    for ep in episodes:
        ep.update({
            'combo_idx':       combo_idx,
            'n_req':           n_req,
            'battery_uav':     battery_uav,
            'wind_magnitude':  wind_magnitude,
            'q_uav':           q_uav,
            'uav_market_frac': ms['uav'],
            'eps_shared':      eps,
            'depot_shared':    int(ds),
            'alpha_asymmetry': aa,
            'delta_minutes':   delta_minutes,
        })
    return result, episodes


# ── Mechanism axes (who gets what, what infrastructure is shared) ─────────────
SWEEP_AXES = {
    'market_split': [
        {'uav': 0.30, 'adr': 0.70},
        {'uav': 0.50, 'adr': 0.50},
        {'uav': 0.70, 'adr': 0.30},
    ],
    'eps_shared': [0.25, 0.50],
    'depot_shared': [False, True],
    'alpha_asymmetry': [1.0, 0.67, 0.50],
    'charging_infra': [
        (0.0, 0.0),
        (5.0, 2.0),
        (2.0, 5.0),
    ],
}

# ── Operational stress axes (UAV/ADR complementarity stressors) ──────────────
# wind_magnitude is m/s; 18 exceeds FlyCart 30's published wind limit and
# is a synthetic out-of-envelope stress, not validated FlyCart operation.
#   → increases UAV energy drain; ADR unaffected → coalition lets ADR cover more
# battery_uav: usable simulated pack energy in kJ.
#   → tight battery forces more UAV depot returns → ADR substitution value rises
# delta_minutes: 2=reactive, 10=standard
#   → short delta = more replanning overhead for standalone firms
STRESS_AXES = {
    'wind_magnitude': [0.0, 8.0, 12.0],
    'battery_uav':    [UAV_BATTERY_KJ, 1800.0, 600.0],
    'delta_minutes':  [2.0, 10.0],
}

def run_full_sweep(
    solver,
    n_episodes: int = 200,
    delta_minutes: float = 10.0,
    seed_base: int = 0,
    n_workers: int = 1,
    n_req: int = 25,
    n_uav: int = 5,
    n_adr: int = 4,
    valuation_mode: str = 'market_books',
    stress_sweep: bool = False,
    per_episode_csv: Optional[str] = None,
) -> List[Dict]:
    """
    stress_sweep=False  : mechanism sweep only (3×2×2×3×3 = 108 configs)
    stress_sweep=True   : mechanism × stress axes (108 × 3 wind × 3 battery × 2 delta = 1944 configs)
                          Run on cluster; use n_workers >= 8.
    """
    from itertools import product as _product

    mech_combos = list(_product(
        SWEEP_AXES['market_split'],
        SWEEP_AXES['eps_shared'],
        SWEEP_AXES['depot_shared'],
        SWEEP_AXES['alpha_asymmetry'],
        SWEEP_AXES['charging_infra'],
    ))

    if stress_sweep:
        stress_combos = list(_product(
            STRESS_AXES['wind_magnitude'],
            STRESS_AXES['battery_uav'],
            STRESS_AXES['delta_minutes'],
        ))
    else:
        stress_combos = [(0.0, UAV_BATTERY_KJ, delta_minutes)]

    all_combos = list(_product(mech_combos, stress_combos))

    print(
        f'Running sweep: {len(all_combos)} configs × {n_episodes} episodes = '
        f'{len(all_combos) * n_episodes} dispatcher runs'
        + (f'  [{n_workers} workers]' if n_workers > 1 else '')
        + (f'  [stress_sweep=True]' if stress_sweep else '')
    )

    task_args = []
    for i, ((ms, eps, ds, aa, (cu, ca)), (wind, bat_u, dt)) in enumerate(all_combos):
        effective_delta = dt if stress_sweep else delta_minutes
        task_args.append((
            solver, n_episodes, ms, eps, ds, aa, cu, ca,
            effective_delta, seed_base, n_req, n_uav, n_adr,
            valuation_mode, wind, bat_u, ADR_BATTERY_KJ,
        ))

    if per_episode_csv and os.path.exists(per_episode_csv):
        raise FileExistsError('Use a new per-episode output file; do not mix validation runs')
    worker = _run_one_config_with_episodes if per_episode_csv else _run_one_config
    if n_workers > 1:
        from concurrent.futures import ProcessPoolExecutor
        with ProcessPoolExecutor(max_workers=n_workers) as pool:
            results = list(pool.map(worker, task_args))
    else:
        results = []
        for i, args in enumerate(task_args):
            ms, eps, ds, aa, cu, ca = args[2], args[3], args[4], args[5], args[6], args[7]
            wind, bat_u = args[14], args[15]
            print(
                f'Config {i + 1}/{len(all_combos)}: '
                f'split={ms}, eps={eps}, depot_shared={ds}, alpha_asym={aa}, '
                f'charging=({cu},{ca}), wind={wind}, bat_uav={bat_u}'
            )
            results.append(worker(args))

    if per_episode_csv:
        import csv
        episodes = [episode for _, rows in results for episode in rows]
        with open(per_episode_csv, 'x', newline='', encoding='utf-8') as stream:
            writer = csv.DictWriter(stream, fieldnames=list(episodes[0]))
            writer.writeheader()
            writer.writerows(episodes)
        results = [result for result, _ in results]

    return results

# ── Focused complementarity sweep axes (Rung C, 12h budget) ──────────────────
# Designed to isolate *when* coalition matters:
#   battery_uav: ADR substitution value (low battery → UAV needs frequent depot returns)
#   wind_magnitude: UAV energy stress (coalition lets ADR absorb overflow)
#   n_req: load pressure on fixed fleet (coalition routing consolidation value)
#   market_split / eps_shared: mechanism levers that interact with operational stress
#   alpha_asymmetry: cost structure asymmetry (changes which firm benefits more)
#   depot_shared / charging_infra: infrastructure sharing benefits
#
# 3×3×2×3×2×3×3×3 = 2916 configs
# charging_infra dropped (depot_shared already captures infrastructure sharing);
# q_uav added to stress ADR capacity advantage (q_adr=10 fixed).
# At 50 eps × 4 workers × ~0.6s/ep: ~6h greedy → RL at 15 eps × ~0.3s/ep: ~3.6h → ~10h total
COMPLEMENTARITY_AXES_C = {
    'market_split': [
        {'uav': 0.25, 'adr': 0.75},
        {'uav': 0.50, 'adr': 0.50},
        {'uav': 0.75, 'adr': 0.25},
    ],
    'eps_shared':      [0.0, 0.25, 0.50],
    'depot_shared':    [False, True],
    'battery_uav':     [600.0, 2000.0, UAV_BATTERY_KJ],
    'wind_magnitude':  [0.0, 8.0],
    'alpha_asymmetry': [0.5, 1.0, 2.0],
    'n_req':           [15, 25, 40],
    'q_uav':           [5.0, 10.0, UAV_CAPACITY_KG],
}


def run_complementarity_sweep_c(
    solver,
    n_episodes: int = 100,
    delta_minutes: float = 10.0,
    seed_base: int = 0,
    n_workers: int = 4,
    n_uav: int = 5,
    n_adr: int = 4,
    valuation_mode: str = 'market_books',
    checkpoint_path: str = None,
    flush_every: int = 50,
    per_episode_csv: str = None,
    demand_heavy_tail: bool = False,
) -> List[Dict]:
    """
    Focused complementarity stress sweep for Rung C.

    Tests 2916 configs across 8 operational axes. Uses as_completed so
    completed results are flushed to checkpoint_path every flush_every configs —
    a wall-time kill loses at most flush_every configs worth of work.

    If checkpoint_path already contains partial results, those configs are
    skipped and the sweep resumes from where it left off.

    Demand distribution is explicit and independent of request count.
    """
    import csv as _csv
    from itertools import product as _product
    if flush_every < 1:
        raise ValueError('Checkpoint flush interval must be positive')

    # ── Checkpoint I/O helpers ────────────────────────────────
    # Checkpoint rows are flat dicts (same schema as _flatten_result in
    # run_coalition.py). New results from _run_one_config_c are nested dicts.
    # _flat() / _nested() convert between the two so the checkpoint file stays
    # consistent on resume and the return value is always nested dicts.

    def _flat(r: dict) -> dict:
        cfg = r['config']
        ms  = cfg['market_split']
        uav = int(ms['uav'] * 100); adr = 100 - uav
        eps = int(cfg['eps_shared'] * 100)
        dep = 'shared' if cfg['depot_shared'] else 'sep'
        aa  = f"a{cfg['alpha_asymmetry']:.2f}"
        cu  = cfg.get('charging_cost_uav', 0.0)
        ca  = cfg.get('charging_cost_adr', 0.0)
        row: dict = {
            'config_json': json.dumps(cfg, sort_keys=True),
            'coalition_version': COALITION_VERSION,
            'valuation_mode': cfg['valuation_mode'],
            'seed_base': cfg['seed_base'],
            'n_episodes': cfg['n_episodes'],
            'n_uav': cfg['n_uav'],
            'n_adr': cfg['n_adr'],
            'scenario':          f'U{uav}A{adr}_e{eps}_{dep}_{aa}_cu{cu:.0f}ca{ca:.0f}',
            'uav_market_frac':   ms['uav'],
            'adr_market_frac':   ms['adr'],
            'eps_shared':        cfg['eps_shared'],
            'depot_shared':      int(cfg['depot_shared']),
            'alpha_asymmetry':   cfg['alpha_asymmetry'],
            'charging_cost_uav': cu,
            'charging_cost_adr': ca,
            'delta_minutes':     cfg.get('delta_minutes', 10.0),
            'wind_magnitude':    cfg.get('wind_magnitude', 0.0),
            'battery_uav':       cfg.get('battery_uav', UAV_BATTERY_KJ),
            'battery_adr':       cfg.get('battery_adr', ADR_BATTERY_KJ),
            'demand_max':        cfg.get('demand_max', 10.0),
            'demand_heavy_tail': int(cfg.get('demand_heavy_tail', False)),
            'q_uav':             cfg.get('q_uav', UAV_CAPACITY_KG),
            'phi_uav_mean':      r['phi_uav_mean'],
            'phi_uav_std':       r['phi_uav_std'],
            'phi_adr_mean':      r['phi_adr_mean'],
            'phi_adr_std':       r['phi_adr_std'],
            'phi_coalition_mean': r['phi_coalition_mean'],
            'phi_coalition_std':  r['phi_coalition_std'],
            'coalition_gain_mean': r['coalition_gain_mean'],
            'coalition_gain_std':  r.get('coalition_gain_std', ''),
            'coalition_gain_p5':   r.get('coalition_gain_p5', ''),
            'coalition_gain_positive_rate': r.get('coalition_gain_positive_rate', ''),
            'core_exists':  int(r['core_exists']),
            'core_freq':    r.get('core_freq', float(r['core_exists'])),
            'conviction':   r.get('conviction', ''),
            'mean_undelivered': r.get('mean_undelivered', 0.0),
            'winner': 'coalition' if r['phi_coalition_mean'] <= r['phi_uav_mean'] + r['phi_adr_mean'] else 'separate_firms',
            'plan_projection_count': r.get('plan_projection_count', 0),
            'scientific_release_validated': True,
        }
        for m in ('shapley', 'epm', 'pam'):
            for k in (f'uav_ir_rate_{m}', f'adr_ir_rate_{m}', f'core_rate_{m}',
                      f'uav_gain_mean_{m}', f'adr_gain_mean_{m}',
                      f'uav_gain_p5_{m}', f'adr_gain_p5_{m}'):
                row[k] = r.get(k, '')
        for k in ('uav_benefit_mean', 'adr_benefit_mean', 'n_req',
                  'uav_late_pct_mean', 'adr_late_pct_mean', 'coal_late_pct_mean',
                  'uav_served_frac_mean', 'adr_served_frac_mean', 'coal_served_frac_mean',
                  'uav_cost_per_req', 'adr_cost_per_req', 'coal_cost_per_req'):
            row[k] = r.get(k, '')
        return row

    def _nested(row: dict) -> dict:
        def _f(v, d: float = 0.0) -> float:
            try:    return float(v) if v not in ('', None) else d
            except: return d
        config = json.loads(row['config_json'])
        r: dict = {'config': config, 'coalition_version': COALITION_VERSION,
                   'scientific_release_validated': True,
                   'plan_projection_count': int(row.get('plan_projection_count', 0))}
        for k in ('phi_uav_mean', 'phi_uav_std', 'phi_adr_mean', 'phi_adr_std',
                  'phi_coalition_mean', 'phi_coalition_std', 'coalition_gain_mean',
                  'coalition_gain_std', 'coalition_gain_p5', 'coalition_gain_positive_rate',
                  'core_freq', 'conviction', 'n_req', 'mean_undelivered',
                  'uav_benefit_mean', 'adr_benefit_mean',
                  'uav_late_pct_mean', 'adr_late_pct_mean', 'coal_late_pct_mean',
                  'uav_served_frac_mean', 'adr_served_frac_mean', 'coal_served_frac_mean',
                  'uav_cost_per_req', 'adr_cost_per_req', 'coal_cost_per_req'):
            r[k] = _f(row.get(k, 0))
        r['core_exists'] = bool(int(_f(row.get('core_exists', 0))))
        for m in ('shapley', 'epm', 'pam'):
            for k in (f'uav_ir_rate_{m}', f'adr_ir_rate_{m}', f'core_rate_{m}',
                      f'uav_gain_mean_{m}', f'adr_gain_mean_{m}',
                      f'uav_gain_p5_{m}', f'adr_gain_p5_{m}'):
                r[k] = _f(row.get(k, 0))
        return r

    # ── Combo list ────────────────────────────────────────────
    combos = list(_product(
        COMPLEMENTARITY_AXES_C['market_split'],
        COMPLEMENTARITY_AXES_C['eps_shared'],
        COMPLEMENTARITY_AXES_C['depot_shared'],
        COMPLEMENTARITY_AXES_C['battery_uav'],
        COMPLEMENTARITY_AXES_C['wind_magnitude'],
        COMPLEMENTARITY_AXES_C['alpha_asymmetry'],
        COMPLEMENTARITY_AXES_C['n_req'],
        COMPLEMENTARITY_AXES_C['q_uav'],
    ))

    # ── Resume: load already-completed configs ────────────────
    # Workers finish out of order; resume by full configuration identity.
    done_flat_rows: List[Dict] = []
    if checkpoint_path and os.path.exists(checkpoint_path):
        with open(checkpoint_path, newline='', encoding='utf-8') as _fh:
            done_flat_rows = list(_csv.DictReader(_fh))
        for row in done_flat_rows:
            cfg = json.loads(row.get('config_json', '{}'))
            if (row.get('coalition_version') != COALITION_VERSION
                    or cfg.get('model_version') != MODEL_VERSION
                    or cfg.get('energy_model_version') != ENERGY_MODEL_VERSION
                    or cfg.get('generator_version') != GENERATOR_VERSION
                    or int(row.get('seed_base', -1)) != seed_base
                    or int(row.get('n_episodes', -1)) != n_episodes
                    or int(row.get('n_uav', -1)) != n_uav
                    or int(row.get('n_adr', -1)) != n_adr
                    or row.get('valuation_mode') != valuation_mode):
                raise ValueError('Checkpoint has incompatible coalition semantics, bank or fleet')
        print(f'[resume] Loaded {len(done_flat_rows)} completed configs from {checkpoint_path}')

    n_done_already = len(done_flat_rows)
    def config_key(ms, eps, ds, bat_u, wind, aa, n_req, q_uav):
        return (float(ms['uav']), float(eps), bool(ds), float(bat_u), float(wind), float(aa), int(n_req), float(q_uav))
    completed_keys = {config_key({'uav': r['uav_market_frac']}, r['eps_shared'], bool(int(r['depot_shared'])),
        r['battery_uav'], r['wind_magnitude'], r['alpha_asymmetry'], r['n_req'], r['q_uav']) for r in done_flat_rows}
    if len(completed_keys) != n_done_already:
        raise ValueError('Duplicate configuration rows in checkpoint')
    if not completed_keys <= {config_key(*combo) for combo in combos}:
        raise ValueError('Checkpoint contains configurations outside this sweep')
    if any(bool(int(r['demand_heavy_tail'])) != demand_heavy_tail or float(r['delta_minutes']) != delta_minutes for r in done_flat_rows):
        raise ValueError('Checkpoint has incompatible demand distribution or replanning interval')

    # Aggregate and episode files are separate writes. A kill between them
    # requires explicit recovery, never silent re-append of orphan episodes.
    if per_episode_csv:
        saved_episodes = []
        if os.path.exists(per_episode_csv):
            with open(per_episode_csv, newline='', encoding='utf-8') as stream:
                saved_episodes = list(_csv.DictReader(stream))
        completed_configs = {json.dumps(json.loads(row['config_json']), sort_keys=True)
                             for row in done_flat_rows}
        episode_ids = {config: set() for config in completed_configs}
        for row in saved_episodes:
            config = json.dumps(json.loads(row.get('config_json', '{}')), sort_keys=True)
            index = int(row.get('episode_idx', -1))
            if (config not in episode_ids or row.get('coalition_version') != COALITION_VERSION
                    or index not in range(n_episodes) or index in episode_ids[config]
                    or int(row.get('episode_seed', -1)) != seed_base + index):
                raise ValueError('Episode/aggregate checkpoint mismatch; recover explicitly or use new output files')
            episode_ids[config].add(index)
        if any(indices != set(range(n_episodes)) for indices in episode_ids.values()):
            raise ValueError('Completed aggregate lacks its episode bank; do not silently resume')

    # ── Build task list, skipping already-done configs ────────
    use_episodes = per_episode_csv is not None
    task_args = []
    for i, (ms, eps, ds, bat_u, wind, aa, n_req, q_uav) in enumerate(combos):
        if config_key(ms, eps, ds, bat_u, wind, aa, n_req, q_uav) in completed_keys:
            continue
        heavy = demand_heavy_tail
        base = (
            solver, n_episodes, ms, eps, ds, aa,
            delta_minutes, seed_base, n_req, n_uav, n_adr,
            valuation_mode, wind, bat_u, ADR_BATTERY_KJ,
            (1.0, 10.0), heavy, q_uav,
        )
        task_args.append(base + (i,) if use_episodes else base)

    n_total = len(combos)
    print(
        f'Complementarity sweep (Rung C): {n_total} configs × {n_episodes} episodes = '
        f'{n_total * n_episodes} dispatcher runs'
        + (f'  [{n_workers} workers]' if n_workers > 1 else '')
        + (f'  ({n_done_already} already done, {len(task_args)} remaining)' if n_done_already else '')
    )

    order = {config_key(*combo): i for i, combo in enumerate(combos)}
    def result_order(result):
        cfg = result['config']
        return order[config_key(cfg['market_split'], cfg['eps_shared'], cfg['depot_shared'], cfg['battery_uav'],
            cfg['wind_magnitude'], cfg['alpha_asymmetry'], result['n_req'], cfg['q_uav'])]

    if not task_args:
        print('All configs already complete — returning cached results.')
        return sorted([_nested(r) for r in done_flat_rows], key=result_order)

    # ── Per-episode CSV helper ────────────────────────────────
    # Written from the main process (no locking needed). Appends rows after
    # each config completes. Header is written once when the file is new.
    _ep_header_written = (per_episode_csv is not None and os.path.exists(per_episode_csv))

    def _write_episodes(episodes: List[Dict]) -> None:
        nonlocal _ep_header_written
        if not per_episode_csv or not episodes:
            return
        with open(per_episode_csv, 'a', newline='', encoding='utf-8') as _fh:
            w = _csv.DictWriter(_fh, fieldnames=list(episodes[0].keys()))
            if not _ep_header_written:
                w.writeheader()
                _ep_header_written = True
            w.writerows(episodes)

    # ── Flush helper ──────────────────────────────────────────
    # Writes done_flat_rows (already flat) + flatten(new_results) to checkpoint.
    def _flush(new_results: List[Dict], path: str) -> None:
        if not path:
            return
        flat = list(done_flat_rows) + [_flat(r) for r in new_results]
        if not flat:
            return
        fieldnames = list(flat[0].keys())
        with open(path, 'w', newline='', encoding='utf-8') as _fh:
            w = _csv.DictWriter(_fh, fieldnames=fieldnames, extrasaction='ignore')
            w.writeheader()
            w.writerows(flat)

    # ── Run remaining configs ─────────────────────────────────
    new_results: List[Dict] = []

    if n_workers > 1:
        from concurrent.futures import ProcessPoolExecutor, as_completed
        worker_fn = _run_one_config_c_with_episodes if use_episodes else _run_one_config_c
        with ProcessPoolExecutor(max_workers=n_workers) as pool:
            futures = {pool.submit(worker_fn, a): idx
                       for idx, a in enumerate(task_args)}
            n_done = n_done_already
            for fut in as_completed(futures):
                if use_episodes:
                    result, episodes = fut.result()
                    new_results.append(result)
                    _write_episodes(episodes)
                else:
                    new_results.append(fut.result())
                n_done += 1
                if n_done % flush_every == 0 or n_done == n_total:
                    _flush(new_results, checkpoint_path)
                    print(f'  [checkpoint] {n_done}/{n_total} configs done', flush=True)
    else:
        for i, args in enumerate(task_args):
            ms, eps, ds, aa = args[2], args[3], args[4], args[5]
            n_req, wind, bat_u, q_uav = args[8], args[12], args[13], args[17]
            n_done = n_done_already + i + 1
            print(
                f'Config {n_done}/{n_total}: '
                f'n_req={n_req}, q_uav={q_uav}, wind={wind}, bat_uav={bat_u}, '
                f'split={ms}, eps={eps}, depot={ds}, alpha={aa}'
            )
            if use_episodes:
                result, episodes = _run_one_config_c_with_episodes(args)
                new_results.append(result)
                _write_episodes(episodes)
            else:
                new_results.append(_run_one_config_c(args))
            if n_done % flush_every == 0 or n_done == n_total:
                _flush(new_results, checkpoint_path)
                print(f'  [checkpoint] {n_done}/{n_total} configs done', flush=True)

    return sorted([_nested(r) for r in done_flat_rows] + new_results, key=result_order)
