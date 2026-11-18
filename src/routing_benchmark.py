"""Solver adapters for frozen rolling-horizon evaluations."""
from __future__ import annotations

from artifacts import load_tensor_file

import argparse
import csv
from hashlib import file_digest
import json
import os
import sys
import time
from pathlib import Path


from tools.experiments.paired_equal_budget_benchmark import (
    MethodRegistry, _coerce_episode_log, run_benchmark,
    RESULT_FIELDS, BENCHMARK_FORMAT_VERSION, _done_pairs, run_one)


def _configure_tensor_physics(instance, uav_pack, adr_pack, uav_reserve, adr_reserve):
    from dataclasses import replace
    from hardware_profiles import physical_configuration

    config = physical_configuration(instance.get('physical_config'))
    phases = config.uav_phases
    config = replace(
        config,
        uav_battery=replace(config.uav_battery,
                            usable_fraction=uav_pack / config.uav_battery.rated_energy_kj,
                            reserve_soc=uav_reserve),
        adr_battery=replace(config.adr_battery,
                            usable_fraction=adr_pack / config.adr_battery.rated_energy_kj,
                            reserve_soc=adr_reserve),
        uav_phases=replace(
            phases,
            depot_landing_min=1.0 if phases.depot_landing_min is None else phases.depot_landing_min,
            depot_takeoff_min=1.0 if phases.depot_takeoff_min is None else phases.depot_takeoff_min,
        ),
        continuation_policy='legacy_leg_only',
    )
    instance['physical_config'] = config.to_dict()


def _incomplete_shift(error):
    completed = [leg for tick in error.episode_log for leg in tick.get('completed', ())]
    return dict(service_complete=False, total_delivered=sum(
        leg['leg_type'] == 'delivery' for leg in completed), total_cost=float('inf'),
        recharge_events=sum(leg['leg_type'] == 'depot' for leg in completed),
        replan_count=sum(bool(tick.get('replanned')) for tick in error.episode_log),
        solver_status=error.reason)


def _recharge_cap_from_n_req(n_req):
    """Rung-indexed uniform recharge cap; see Table 2."""
    n_req = int(n_req)
    if n_req <= 5: return 1
    if n_req <= 10: return 2
    if n_req <= 25: return 3
    if n_req <= 60: return 4
    return None


def _enrich_bank_entry(entry, n_uav=2, n_adr=2, *,
                       reserve_frac_uav=0.25, reserve_frac_adr=0.20,
                       capacity_kj_uav=None, capacity_kj_adr=None,
                       max_recharges_per_vehicle=None,
                       recharge_cost_includes_dwell=True, allow_delayed_departure=False):
    """Inject fleet, demand, time windows, edge attrs, and C1/C2/V3 keys
    into a bank entry so every solver shares the same planning budget."""
    import numpy as np
    from model_config import (ADR_BATTERY_KJ, ADR_CAPACITY_KG, UAV_BATTERY_KJ,
                                UAV_CAPACITY_KG)
    e = dict(entry)
    n_req = int(e['n_req'])
    n_depots = int(e['n_depots'])
    n_total = n_depots + 2 * n_req

    e.setdefault('n_uav', n_uav)
    e.setdefault('n_adr', n_adr)
    e.setdefault('n_depots_adr', n_depots - int(e['n_depots_uav']))
    e.setdefault('capacity', np.array(
        [UAV_CAPACITY_KG] * n_uav + [ADR_CAPACITY_KG] * n_adr, dtype=np.float32))
    uav_pack = float(capacity_kj_uav) if capacity_kj_uav is not None else float(UAV_BATTERY_KJ)
    adr_pack = float(capacity_kj_adr) if capacity_kj_adr is not None else float(ADR_BATTERY_KJ)
    e.setdefault('battery', np.array(
        [uav_pack] * n_uav + [adr_pack] * n_adr, dtype=np.float32))
    e['reserve_frac_by_mode'] = {'uav': float(reserve_frac_uav),
                                 'adr': float(reserve_frac_adr)}
    _cap = max_recharges_per_vehicle if max_recharges_per_vehicle is not None else _recharge_cap_from_n_req(n_req)
    if _cap is not None:
        e['max_recharges_per_vehicle'] = int(_cap)
    e['recharge_cost_includes_dwell'] = bool(recharge_cost_includes_dwell)
    e['allow_delayed_departure'] = bool(allow_delayed_departure)

    arrivals_sorted = sorted(e['arrivals'],
                             key=lambda r: float(r.get('t_arrival', r.get('t_pickup', 0.0))))
    e['t_arrival'] = np.array([float(r['t_arrival']) for r in arrivals_sorted], dtype=np.float32)
    demand_vals = np.array([float(r['demand']) for r in arrivals_sorted], dtype=np.float32)
    t_pickup = np.array([float(r['t_pickup']) for r in arrivals_sorted], dtype=np.float32)
    t_delivery = np.array([float(r['t_delivery']) for r in arrivals_sorted], dtype=np.float32)

    if 'demand' not in e:
        demand = np.zeros(n_total, dtype=np.float32)
        demand[n_depots:n_depots + n_req] = demand_vals
        demand[n_depots + n_req:] = -demand_vals
        e['demand'] = demand
    if 'time_window' not in e:
        tw = np.zeros(n_total, dtype=np.float32)
        tw[n_depots:n_depots + n_req] = t_pickup
        tw[n_depots + n_req:] = t_delivery
        e['time_window'] = tw
    if 'edge_attr_d' not in e:
        d_uav = np.asarray(e['d_uav'], dtype=np.float32)
        e['edge_attr_d'] = d_uav.reshape(-1, 1)
    if 'edge_attr_r' not in e:
        d_adr = np.asarray(e['d_adr'], dtype=np.float32)
        e['edge_attr_r'] = d_adr.reshape(-1, 1)
    return e


def _build_classical_registry(n_uav=2, n_adr=2, *,
                              reserve_frac_uav=0.25, reserve_frac_adr=0.20,
                              capacity_kj_uav=None, capacity_kj_adr=None,
                              max_recharges_per_vehicle=None,
                              recharge_cost_includes_dwell=True,
                              allow_delayed_departure=False,
                              depot_sharing=True,
                              depot_visit_limit=None,
                              mip_time_budget_s=None, gurobi_threads=0,
                              ortools_time_budget_s=None):
    """Classical-solver registry with fleet/C1/C2/V3 keys wired consistently."""
    reg = MethodRegistry()

    def _enrich(fi):
        return _enrich_bank_entry(
            fi, n_uav=n_uav, n_adr=n_adr,
            reserve_frac_uav=reserve_frac_uav,
            reserve_frac_adr=reserve_frac_adr,
            capacity_kj_uav=capacity_kj_uav,
            capacity_kj_adr=capacity_kj_adr,
            max_recharges_per_vehicle=max_recharges_per_vehicle,
            recharge_cost_includes_dwell=recharge_cost_includes_dwell,
            allow_delayed_departure=allow_delayed_departure,
        )

    def _exact_cfg_kwargs():
        kw = dict(
            reserve_frac_uav=float(reserve_frac_uav),
            reserve_frac_adr=float(reserve_frac_adr),
            recharge_cost_includes_dwell=bool(recharge_cost_includes_dwell),
            depot_sharing=depot_sharing,
        )
        if capacity_kj_uav is not None: kw['capacity_kj_uav'] = float(capacity_kj_uav)
        if capacity_kj_adr is not None: kw['capacity_kj_adr'] = float(capacity_kj_adr)
        if max_recharges_per_vehicle is not None:
            kw['max_recharges_per_vehicle'] = int(max_recharges_per_vehicle)
        if depot_visit_limit is not None:
            kw['depot_visit_limit'] = int(depot_visit_limit)
        if gurobi_threads:
            kw['n_threads'] = int(gurobi_threads)
        return kw

    # Cache one Gurobi Env across calls: WLS fast-fails under per-instance env construction.
    _gurobi_state = {'env': None, 'tried': False}

    def _make_alns_wrap(_strong: bool, _label: str):
        def _run(full_instance, *, time_budget_s, seed):
            from ce_cpdptw_alns import ALNSBaseline

            solver = ALNSBaseline(seed=seed,
                                    time_budget_small_s=time_budget_s,
                                    time_budget_large_s=time_budget_s,
                                    strong=_strong)
            return _dispatch_solver(solver, full_instance, _label)
        return _run

    _wrap_alns        = _make_alns_wrap(False, 'alns')
    _wrap_alns_strong = _make_alns_wrap(True,  'alns_strong')

    def _wrap_ortools_cpsat(full_instance, *, time_budget_s, seed):
        # Backend is CBC via pywraplp; label 'ortools_cpsat' kept for manifest continuity.
        from ce_cpdptw_rolling_baselines import (
            ExactRollingConfig, solve_static_instance_with_ortools_rolling)
        inst = _enrich(full_instance)
        _ort_budget = (ortools_time_budget_s if ortools_time_budget_s is not None
                       else (mip_time_budget_s if mip_time_budget_s is not None
                             else time_budget_s))
        cfg = ExactRollingConfig(
            n_uav=int(inst['n_uav']),
            n_adr=int(inst['n_adr']),
            n_depots_uav=int(inst['n_depots_uav']),
            n_depots_adr=int(inst['n_depots_adr']),
            delta_minutes=5.0,
            shift_minutes=120.0,
            time_limit_seconds=float(_ort_budget),
            **_exact_cfg_kwargs(),
        )
        episode_log = solve_static_instance_with_ortools_rolling(inst, cfg)
        return _coerce_episode_log(episode_log, inst, method='ortools_cpsat')

    def _wrap_gurobi(full_instance, *, time_budget_s, seed):
        from ce_cpdptw_rolling_baselines import (
            ExactRollingConfig, solve_static_instance_with_gurobi_rolling)
        if not _gurobi_state['tried']:
            _gurobi_state['tried'] = True
            try:
                import gurobipy as _gp
                _gurobi_state['env'] = _gp.Env()
            except Exception:
                _gurobi_state['env'] = None
        inst = _enrich(full_instance)
        cfg = ExactRollingConfig(
            n_uav=int(inst['n_uav']),
            n_adr=int(inst['n_adr']),
            n_depots_uav=int(inst['n_depots_uav']),
            n_depots_adr=int(inst['n_depots_adr']),
            delta_minutes=5.0,
            shift_minutes=120.0,
            time_limit_seconds=float(mip_time_budget_s) if mip_time_budget_s is not None else float(time_budget_s),
            gurobi_env=_gurobi_state['env'],
            **_exact_cfg_kwargs(),
        )
        episode_log = solve_static_instance_with_gurobi_rolling(inst, cfg)
        return _coerce_episode_log(episode_log, inst, method='gurobi_milp')

    def _wrap_offline_alns(full_instance, *, time_budget_s, seed):
        from offline_alns import solve_static_instance_with_offline_alns
        inst = _enrich(full_instance)
        episode_log = solve_static_instance_with_offline_alns(
            inst, time_budget_s=time_budget_s, seed=seed)
        return _coerce_episode_log(episode_log, inst, method='offline_alns')

    def _dispatch_solver(solver, full_instance, method_name):
        """Run a solve(residual)-style solver through RollingHorizonDispatcher."""
        from dispatch_sim import (MandatoryServiceError, RollingHorizonDispatcher,
                                     Vehicle)
        from model_config import (ADR_BATTERY_KJ, ADR_CAPACITY_KG,
                                     UAV_BATTERY_KJ, UAV_CAPACITY_KG)
        from tools.diagnostics.rl_backend_conformance import \
            _bank_instance_to_tensor

        mock_bank = {'instances': [full_instance],
                     'spec': {'n_uav': n_uav, 'n_adr': n_adr}}
        tensor_instance = _bank_instance_to_tensor(mock_bank, 0)
        tensor_instance['service_policy'] = 'mandatory'
        tensor_instance['reserve_frac_by_mode'] = {
            'uav': float(reserve_frac_uav), 'adr': float(reserve_frac_adr)}
        _cap = max_recharges_per_vehicle if max_recharges_per_vehicle is not None \
            else _recharge_cap_from_n_req(int(tensor_instance.get('n_req', 5)))
        if _cap is not None:
            tensor_instance['max_recharges_per_vehicle'] = int(_cap)
        tensor_instance['recharge_cost_includes_dwell'] = bool(recharge_cost_includes_dwell)
        tensor_instance['allow_delayed_departure'] = bool(allow_delayed_departure)
        # physical_config must use `legacy_leg_only` or the native-config guard
        # (reward.py) raises; depot phase minutes must be set or the bench aborts.
        if 'physical_config' not in tensor_instance:
            from hardware_profiles import (physical_configuration,
                                             UAVPhasePolicy, PhysicalConfiguration)
            from dataclasses import replace
            _default = physical_configuration()
            _phases = replace(_default.uav_phases,
                              depot_landing_min=1.0, depot_takeoff_min=1.0)
            _cfg = replace(_default,
                           uav_phases=_phases,
                           continuation_policy='legacy_leg_only').to_dict()
            tensor_instance['physical_config'] = _cfg

        _n_uav_i = int(tensor_instance.get('n_uav', 2))
        _n_adr_i = int(tensor_instance.get('n_adr', 2))
        n_dep_uav = int(tensor_instance.get('n_depots_uav', 1))
        n_dep_adr = int(tensor_instance['n_depots']) - n_dep_uav
        _uav_pack = float(capacity_kj_uav) if capacity_kj_uav is not None else float(UAV_BATTERY_KJ)
        _adr_pack = float(capacity_kj_adr) if capacity_kj_adr is not None else float(ADR_BATTERY_KJ)
        _configure_tensor_physics(
            tensor_instance, _uav_pack, _adr_pack, reserve_frac_uav, reserve_frac_adr)
        fleet = {}
        vid = 0
        for u in range(_n_uav_i):
            fleet[vid] = Vehicle(vid, 'uav', u % n_dep_uav, 0., _uav_pack,
                                 0., UAV_CAPACITY_KG,
                                 battery_init=_uav_pack,
                                 allowed_depots=tuple(range(n_dep_uav)))
            vid += 1
        for a in range(_n_adr_i):
            depot = n_dep_uav + a % max(n_dep_adr, 1)
            fleet[vid] = Vehicle(vid, 'adr', depot, 0., _adr_pack, 0.,
                                 ADR_CAPACITY_KG, battery_init=_adr_pack,
                                 allowed_depots=tuple(range(n_dep_uav,
                                                              n_dep_uav + n_dep_adr)))
            vid += 1

        arrivals = [dict(row) for row in tensor_instance['arrivals']]
        disp = RollingHorizonDispatcher(solver, delta_minutes=5.,
                                         shift_minutes=120.,
                                         service_policy='mandatory')
        try:
            log = disp.run_shift(arrivals, fleet, tensor_instance)
            summary = log[-1]
        except MandatoryServiceError as error:
            summary = _incomplete_shift(error)
        return _coerce_episode_log(summary, full_instance, method=method_name)

    def _wrap_greedy(full_instance, *, time_budget_s, seed):
        from greedy_insertion import GreedyInsertion
        return _dispatch_solver(GreedyInsertion(), full_instance, 'greedy')

    def _wrap_fifo(full_instance, *, time_budget_s, seed):
        from fifo_baseline import FIFOSolver
        return _dispatch_solver(FIFOSolver(), full_instance, 'fifo')

    def _make_vns_wrap(_robust: bool, _label: str):
        def _run(full_instance, *, time_budget_s, seed):
            from ce_cpdptw_vrp_heuristics import VNSSolver

            solver = VNSSolver(
                time_limit_seconds=float(mip_time_budget_s) if mip_time_budget_s is not None else float(time_budget_s),
                robust=_robust,
            )
            return _dispatch_solver(solver, full_instance, _label)
        return _run

    _wrap_vns        = _make_vns_wrap(False, 'vns')
    _wrap_vns_robust = _make_vns_wrap(True,  'vns_robust')

    reg.register('alns', _wrap_alns)
    reg.register('alns_strong', _wrap_alns_strong)
    reg.register('ortools_cpsat', _wrap_ortools_cpsat)
    reg.register('gurobi_milp', _wrap_gurobi)
    reg.register('offline_alns', _wrap_offline_alns)
    reg.register('greedy', _wrap_greedy)
    reg.register('fifo', _wrap_fifo)
    reg.register('vns', _wrap_vns)
    reg.register('vns_robust', _wrap_vns_robust)
    return reg


def _make_hetgat_coop_runner(checkpoint_path: Path, n_uav=2, n_adr=2,
                              n_samples: int = 1, *,
                              reserve_frac_uav: float = 0.25,
                              reserve_frac_adr: float = 0.20,
                              capacity_kj_uav=None,
                              capacity_kj_adr=None,
                              max_recharges_per_vehicle=None,
                              recharge_cost_includes_dwell: bool = True,
                              allow_delayed_departure: bool = False,
                              chain_lookahead_in_mask: bool = False,
                              continuation_check: str = 'off',
                              recover_stalled: bool = False):
    """Build a method wrapper that runs the HetGAT coop policy via
    RollingHorizonDispatcher on a bank instance. Solver is cached across calls."""
    state = {'solver': None, 'method_name': None}

    def _run(full_instance, *, time_budget_s, seed):
        import copy
        from dispatch_sim import (MandatoryServiceError, RollingHorizonDispatcher,
                                     Vehicle)
        from main import build_solver
        from model_config import (ADR_BATTERY_KJ, ADR_CAPACITY_KG,
                                     UAV_BATTERY_KJ, UAV_CAPACITY_KG)
        from tools.diagnostics.rl_backend_conformance import \
            _bank_instance_to_tensor

        if state['solver'] is None:
            # Coop ckpts carry 'request_encoder.*' keys; route ckpts do not.
            import torch as _torch
            _raw = load_tensor_file(str(checkpoint_path), map_location='cpu',
                               weights_only=True)
            _sd = (_raw.get('model_state_dict', _raw)
                   if isinstance(_raw, dict) else _raw)
            _is_coop = any(str(k).startswith('request_encoder.') for k in _sd)
            _base = 'hetgat_coop' if _is_coop else 'hetgat_route'
            if n_samples and int(n_samples) > 1:
                state['method_name'] = f'{_base}_pomo{int(n_samples)}'
            else:
                state['method_name'] = _base
            del _raw, _sd
            state['solver'] = build_solver(str(checkpoint_path), arch='hetgat',
                                            n_samples=int(n_samples or 1))
            # build_solver defaults omit the inference-time hard-mask flags;
            # walk solver → actor → model → decoder to set them.
            _outer = state['solver']
            for _n in ('actor', 'model', 'solver'):
                _outer = getattr(_outer, _n, _outer)
                if hasattr(_outer, 'decoder'):
                    break
            _dec = getattr(_outer, 'decoder', None)
            if _dec is not None:
                _dec.battery_mask = True
                _dec.battery_mask_lookahead = False
                _dec.battery_mask_reserve_frac = float(reserve_frac_uav)
                _dec.battery_mask_reserve_frac_uav = float(reserve_frac_uav)
                _dec.battery_mask_reserve_frac_adr = float(reserve_frac_adr)
                if hasattr(_dec, 'paper_strict_depot_mask'):
                    _dec.paper_strict_depot_mask = False
            if _outer is not None and hasattr(_outer, 'paper_strict_depot_mask'):
                _outer.paper_strict_depot_mask = False
            # Model re-threads chain_lookahead_in_mask to the decoder each forward;
            # set it on both so the bench matches training-time masking.
            for _node in (_outer, _dec):
                if _node is not None and hasattr(_node, 'use_chain_lookahead_in_mask'):
                    _node.use_chain_lookahead_in_mask = bool(chain_lookahead_in_mask)
            if _outer is not None and hasattr(_outer, 'continuation_check'):
                _outer.continuation_check = continuation_check
        rl_solver = state['solver']

        mock_bank = {'instances': [full_instance],
                     'spec': {'n_uav': n_uav, 'n_adr': n_adr}}
        tensor_instance = _bank_instance_to_tensor(mock_bank, 0)
        tensor_instance['service_policy'] = 'mandatory'
        tensor_instance['reserve_frac_by_mode'] = {
            'uav': float(reserve_frac_uav), 'adr': float(reserve_frac_adr)}
        _cap = max_recharges_per_vehicle if max_recharges_per_vehicle is not None \
            else _recharge_cap_from_n_req(int(tensor_instance.get('n_req', 5)))
        if _cap is not None:
            tensor_instance['max_recharges_per_vehicle'] = int(_cap)
        tensor_instance['recharge_cost_includes_dwell'] = bool(recharge_cost_includes_dwell)
        tensor_instance['allow_delayed_departure'] = bool(allow_delayed_departure)
        # physical_config must use `legacy_leg_only` or the native-config guard
        # (reward.py) raises; depot phase minutes must be set or the bench aborts.
        if 'physical_config' not in tensor_instance:
            from hardware_profiles import (physical_configuration,
                                             UAVPhasePolicy, PhysicalConfiguration)
            from dataclasses import replace
            _default = physical_configuration()
            _phases = replace(_default.uav_phases,
                              depot_landing_min=1.0, depot_takeoff_min=1.0)
            _cfg = replace(_default,
                           uav_phases=_phases,
                           continuation_policy='legacy_leg_only').to_dict()
            tensor_instance['physical_config'] = _cfg

        _n_uav_i = int(tensor_instance.get('n_uav', 2))
        _n_adr_i = int(tensor_instance.get('n_adr', 2))
        n_dep_uav = int(tensor_instance.get('n_depots_uav', 1))
        n_dep_adr = int(tensor_instance['n_depots']) - n_dep_uav
        _uav_pack = float(capacity_kj_uav) if capacity_kj_uav is not None else float(UAV_BATTERY_KJ)
        _adr_pack = float(capacity_kj_adr) if capacity_kj_adr is not None else float(ADR_BATTERY_KJ)
        _configure_tensor_physics(
            tensor_instance, _uav_pack, _adr_pack, reserve_frac_uav, reserve_frac_adr)
        fleet = {}
        vid = 0
        for u in range(_n_uav_i):
            fleet[vid] = Vehicle(vid, 'uav', u % n_dep_uav, 0., _uav_pack,
                                 0., UAV_CAPACITY_KG,
                                 battery_init=_uav_pack,
                                 allowed_depots=tuple(range(n_dep_uav)))
            vid += 1
        for a in range(_n_adr_i):
            depot = n_dep_uav + a % max(n_dep_adr, 1)
            fleet[vid] = Vehicle(vid, 'adr', depot, 0., _adr_pack, 0.,
                                 ADR_CAPACITY_KG, battery_init=_adr_pack,
                                 allowed_depots=tuple(range(n_dep_uav,
                                                              n_dep_uav + n_dep_adr)))
            vid += 1

        arrivals = [dict(row) for row in tensor_instance['arrivals']]

        recovery = {}
        if recover_stalled:
            from ce_cpdptw_alns import ALNSBaseline
            recovery['stalled_solver'] = ALNSBaseline(
                seed=seed, strong=True, time_budget_small_s=.25, time_budget_large_s=.25,
            )
        disp = RollingHorizonDispatcher(rl_solver, delta_minutes=5.,
                                         shift_minutes=120.,
                                         service_policy='mandatory', **recovery)
        try:
            log = disp.run_shift(arrivals, fleet, tensor_instance)
            summary = log[-1]
        except MandatoryServiceError as error:
            summary = _incomplete_shift(error)

        return _coerce_episode_log(summary, full_instance, method=state['method_name'])

    return _run


# Parallel driver: workers use "spawn" start method and rebuild their own registry
# (torch-model closures can't be pickled). Main process writes CSV sequentially.

_WORKER_REGISTRY = None

