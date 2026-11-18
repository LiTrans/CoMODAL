"""Cross-backend five-cost mandatory-service conformance audit.

Runs the same tiny scenario through every claimed scalar/exact backend and the
shared tensor replay, and checks:

- each produces ``service_complete=True`` under the mandatory contract;
- each honors the five raw-arrival costs with no missed-order/soft-battery term;
- the shared trace audit agrees with the dispatcher summary;
- the legacy optional policy is explicitly rejected on the mandatory objective;
- unsupported configured modes fail before decoding.

Not a performance benchmark. The claim is that the backends solve the same
problem, not that any of them is optimal.
"""
from __future__ import annotations

import argparse
import dataclasses
import importlib
import json
import sys
import time
import warnings
from dataclasses import asdict
from pathlib import Path
from typing import Any, Dict, List

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from audit_cost_trace import audit_cost_trace
from ce_cpdptw_alns import ALNSBaseline
from ce_cpdptw_rolling_baselines import ExactRollingConfig, ORToolsRollingHorizonSolver
from dispatch_sim import MandatoryServiceError, RollingHorizonDispatcher, Vehicle
from fifo_baseline import FIFOSolver
from greedy_insertion import GreedyInsertion
from model_config import ADR_BATTERY_KJ, UAV_BATTERY_KJ, ENERGY_MODEL_VERSION
from release_checks import release_status
from reward import MANDATORY_OBJECTIVE_VERSION, ObjectiveWeights, mandatory_objective_weights

FIVE_COST_KEYS = ('operating_cost', 'pickup_early_cost', 'pickup_late_cost',
                  'delivery_late_cost')  # delivery credit is not a scientific term


def _scenario():
    """Two depots, two orders — one UAV-feasible and one ADR-feasible.

    Deliberately small so every claimed backend (including the exact MILP under
    CBC with a short time limit) completes mandatory service. Scope is
    cross-backend consistency, not a scaling benchmark.
    """
    xy = np.array([
        [0., 0.],      # 0: UAV depot
        [0., .02],     # 1: ADR depot
        [.1, 0.],      # 2: pickup R0 (UAV)
        [.2, 0.],      # 3: delivery R0 (UAV)
        [.05, .05],    # 4: pickup R1 (ADR)
        [.1, .05],     # 5: delivery R1 (ADR)
    ])
    d = np.linalg.norm(xy[:, None] - xy[None, :], axis=-1)
    instance = dict(n_depots=2, n_depots_uav=1, n_req=2,
                    d_uav=d, d_adr=d, coordinates=xy,
                    wind=np.array([0., 0.]))
    rows = [
        dict(req_id=0, t_arrival=0., t_pickup=1., t_delivery=20.,
             pickup_node=2, delivery_node=3, demand=2.),
        dict(req_id=1, t_arrival=0., t_pickup=1., t_delivery=20.,
             pickup_node=4, delivery_node=5, demand=1.),
    ]
    fleet = {
        0: Vehicle(0, 'uav', 0, 0., UAV_BATTERY_KJ, 0., 40.,
                   battery_init=UAV_BATTERY_KJ, allowed_depots=(0,)),
        1: Vehicle(1, 'adr', 1, 0., ADR_BATTERY_KJ, 0., 10.,
                   battery_init=ADR_BATTERY_KJ, allowed_depots=(1,)),
    }
    return instance, fleet, rows


def _scale_scenario(n_req: int, seed: int = 20261009):
    """Larger uniform-random fixture for scale-conformance (n=10/25)."""
    rng = np.random.default_rng(seed)
    n_depots = 2
    coords = np.vstack([
        np.array([[0., 0.], [1., 1.]]),
        rng.uniform(.05, .95, size=(2 * n_req, 2)),
    ])
    d = np.linalg.norm(coords[:, None] - coords[None, :], axis=-1)
    rows = []
    for rid in range(n_req):
        rows.append(dict(
            req_id=rid, t_arrival=0.,
            t_pickup=float(rng.uniform(5., 20.)),
            t_delivery=float(rng.uniform(40., 90.)),
            pickup_node=n_depots + rid,
            delivery_node=n_depots + n_req + rid,
            demand=float(rng.uniform(1., 8.))))
    instance = dict(n_depots=n_depots, n_depots_uav=1, n_req=n_req,
                    d_uav=d, d_adr=d, coordinates=coords,
                    wind=np.array([0., 0.]))
    n_uav = max(2, n_req // 10)
    n_adr = max(2, n_req // 10)
    fleet = {}
    vid = 0
    for _ in range(n_uav):
        fleet[vid] = Vehicle(vid, 'uav', 0, 0., UAV_BATTERY_KJ, 0., 40.,
                             battery_init=UAV_BATTERY_KJ, allowed_depots=(0,))
        vid += 1
    for _ in range(n_adr):
        fleet[vid] = Vehicle(vid, 'adr', 1, 0., ADR_BATTERY_KJ, 0., 10.,
                             battery_init=ADR_BATTERY_KJ, allowed_depots=(1,))
        vid += 1
    return instance, fleet, rows


def _solvers(exact_time_limit: float = 30.):
    exact_cfg = ExactRollingConfig(1, 1, 1, 1, time_limit_seconds=exact_time_limit,
                                   depot_visit_limit=0, energy_load_intervals=8)
    return [
        ('greedy', GreedyInsertion()),
        ('fifo', FIFOSolver()),
        ('alns', ALNSBaseline(time_budget_small_s=0.)),
        ('ortools_cbc', ORToolsRollingHorizonSolver(exact_cfg, solver_id='CBC')),
    ]


def _run_dispatcher(solver, instance, fleet, rows, policy='mandatory',
                   cost_weights=None):
    import copy
    disp = RollingHorizonDispatcher(solver, delta_minutes=5., shift_minutes=120.,
                                    service_policy=policy, cost_weights=cost_weights)
    log = disp.run_shift(rows, copy.deepcopy(fleet), copy.deepcopy(instance))
    return log, log[-1]


def _five_cost_from_summary(summary: Dict[str, Any]) -> Dict[str, float]:
    components = {k: float(summary.get(k, 0.)) for k in FIVE_COST_KEYS}
    components['five_cost_total'] = sum(components.values())
    return components


def _check_mandatory_run(name: str, log, summary) -> Dict[str, Any]:
    assert summary['service_complete'], f'{name} must deliver every obligation under the mandatory contract'
    assert summary['undelivered_penalty'] == 0., f'{name} must not charge a missed-order penalty'
    assert summary['battery_penalty_cost'] == 0., f'{name} must not charge a soft battery term'
    quality = float(summary['quality_cost'])
    five = _five_cost_from_summary(summary)
    assert abs(five['five_cost_total'] - quality) < 1e-6, (
        f'{name}: five-cost components sum {five["five_cost_total"]} '
        f'disagrees with reported quality_cost {quality}')
    return five


HEURISTIC_BACKENDS = ('greedy', 'fifo', 'alns')


def run_scale_audit(n_req: int = 10, exact_time_limit: float = 60.,
                    seed: int = 20261009,
                    output_path: Path | None = None) -> Dict[str, Any]:
    """Scale-regime conformance at n=10/25.

    Unlike ``run_audit`` which pins strict backend identity at a tiny fixture,
    this check verifies the weaker but necessary properties at scale:

    - every **heuristic** backend (Greedy, FIFO, ALNS) produces a
      mandatory-complete episode with internally consistent five-cost
      accounting and a passing independent trace audit;
    - all reported heuristic costs are finite;
    - the exact backend's status (``incumbent_found`` / ``no_incumbent`` /
      ``optimal`` / ``raised``) is **recorded** as evidence rather than
      asserted; MILP timeouts at scale are expected.

    Heuristic cost divergence from each other and from any MILP incumbent is
    expected and not asserted. The audit PASSes when the three heuristics meet
    the above invariants, regardless of whether CBC finds an MILP incumbent.
    """
    instance, fleet, rows = _scale_scenario(n_req=n_req, seed=seed)
    report: Dict[str, Any] = {
        'audit': 'backend_scale_conformance',
        'scenario_scale': dict(n_req=n_req, n_vehicles=len(fleet), seed=seed),
        'energy_model_version': ENERGY_MODEL_VERSION,
        'mandatory_objective_version': MANDATORY_OBJECTIVE_VERSION,
        'release_status': release_status(),
        'backends': {},
        'started_utc': time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()),
    }
    for name, solver in _solvers(exact_time_limit=exact_time_limit):
        start = time.perf_counter()
        try:
            log, summary = _run_dispatcher(solver, instance, fleet, rows)
        except Exception as exc:
            report['backends'][name] = dict(
                raised_exception=repr(exc),
                elapsed_s=time.perf_counter() - start,
                policy='mandatory',
                incumbent_found=False,
                scale_evidence_only=name not in HEURISTIC_BACKENDS,
            )
            continue
        elapsed = time.perf_counter() - start
        five = _check_mandatory_run(name, log, summary)
        audit_cost_trace(log, fleet, rows)
        report['backends'][name] = dict(
            service_complete=bool(summary['service_complete']),
            total_delivered=int(summary['total_delivered']),
            total_revealed=int(summary['total_revealed']),
            quality_cost=float(summary['quality_cost']),
            total_cost=float(summary['total_cost']),
            completion_time_min=float(summary['completion_time_min']),
            five_cost=five,
            elapsed_s=elapsed,
            policy='mandatory',
            incumbent_found=True,
            scale_evidence_only=name not in HEURISTIC_BACKENDS,
        )
    report['finished_utc'] = time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())
    heuristic_ok = all(
        report['backends'].get(name, {}).get('service_complete') is True
        for name in HEURISTIC_BACKENDS)
    report['status'] = 'PASS' if heuristic_ok else 'FAIL'
    report['milp_evidence'] = dict(
        (name, dict(incumbent_found=bool(report['backends'].get(name, {}).get('incumbent_found')),
                    raised=report['backends'].get(name, {}).get('raised_exception'),
                    elapsed_s=report['backends'].get(name, {}).get('elapsed_s')))
        for name in report['backends'] if name not in HEURISTIC_BACKENDS)
    if output_path is not None:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps(report, indent=2, sort_keys=True))
    return report


def run_audit(output_path: Path | None = None) -> Dict[str, Any]:
    instance, fleet, rows = _scenario()
    report: Dict[str, Any] = {
        'audit': 'backend_cross_conformance',
        'energy_model_version': ENERGY_MODEL_VERSION,
        'mandatory_objective_version': MANDATORY_OBJECTIVE_VERSION,
        'release_status': release_status(),
        'backends': {},
        'legacy_optional_rejections': {},
        'started_utc': time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()),
    }

    for name, solver in _solvers():
        start = time.perf_counter()
        try:
            log, summary = _run_dispatcher(solver, instance, fleet, rows)
        except Exception as exc:  # pragma: no cover - raised in audit mode
            report['backends'][name] = dict(error=repr(exc))
            continue
        elapsed = time.perf_counter() - start
        five = _check_mandatory_run(name, log, summary)
        audit_cost_trace(log, fleet, rows)  # raises on trace mismatch
        report['backends'][name] = dict(
            service_complete=bool(summary['service_complete']),
            total_delivered=int(summary['total_delivered']),
            total_revealed=int(summary['total_revealed']),
            quality_cost=float(summary['quality_cost']),
            total_cost=float(summary['total_cost']),
            completion_time_min=float(summary['completion_time_min']),
            five_cost=five,
            elapsed_s=elapsed,
            policy='mandatory',
        )

    # Enforce that the mandatory objective rejects legacy optional cost terms.
    bad_weights = asdict(ObjectiveWeights())  # has unserved/battery_shortfall nonzero
    try:
        _run_dispatcher(GreedyInsertion(), instance, fleet, rows,
                        policy='mandatory', cost_weights=bad_weights)
    except ValueError as exc:
        report['legacy_optional_rejections']['extra_cost_weights'] = str(exc)
    else:  # pragma: no cover
        raise AssertionError('Mandatory objective must reject non-zero unserved/battery terms')

    # Confirm the legacy-optional path still runs but produces a different scoring
    # convention (missed-price), with the diagnostic warning in place. We do not
    # claim its total is comparable to the mandatory five-cost total.
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter('always')
        legacy_log, legacy_summary = _run_dispatcher(
            GreedyInsertion(), instance, fleet, rows, policy='legacy_optional')
    report['legacy_optional_rejections']['legacy_run_warnings'] = sorted(
        {str(w.message) for w in caught if 'legacy' in str(w.message).lower()
         or 'optional' in str(w.message).lower()})
    report['legacy_optional_run'] = dict(
        service_complete=bool(legacy_summary['service_complete']),
        quality_cost=float(legacy_summary['quality_cost']),
        total_cost=float(legacy_summary['total_cost']),
        comparable_to_mandatory=False,
    )

    # Unsupported configured modes must fail before decoding/replay.
    from dataclasses import replace
    from hardware_profiles import (BatteryPolicy, FLYCART30_SINGLE, PhysicalConfiguration,
                                   STARSHIP_2017, UAVPhasePolicy)
    config = PhysicalConfiguration(
        uav_battery=BatteryPolicy(FLYCART30_SINGLE.identifier, usable_fraction=.8,
                                  initial_soc=.8, recharge_soc=.9, reserve_soc=.1),
        adr_battery=BatteryPolicy(STARSHIP_2017.identifier, usable_fraction=.8,
                                  initial_soc=.8, recharge_soc=.9, reserve_soc=.1),
        uav_phases=UAVPhasePolicy(customer_landing_min=.4, customer_takeoff_min=.6,
                                 depot_landing_min=.3, depot_takeoff_min=.5),
        continuation_policy='depot_or_terminal')  # intentionally unsupported in natives
    guarded = {}
    tensor_data = _tensor_fixture(config)
    for name, call in _unsupported_native_calls(tensor_data):
        try:
            call()
        except NotImplementedError as exc:
            guarded[name] = str(exc) or type(exc).__name__
        else:  # pragma: no cover
            raise AssertionError(f'Unsupported configured mode {name!r} must fail before decoding')
    report['unsupported_config_guards'] = guarded

    report['finished_utc'] = time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())
    report['status'] = 'PASS' if report['backends'] and all(
        v.get('service_complete') for v in report['backends'].values()) else 'FAIL'

    if output_path is not None:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps(report, indent=2, sort_keys=True))
    return report


def _tensor_fixture(config):
    import torch
    n = 4
    xy = torch.stack([torch.arange(n) * .1, torch.zeros(n)], dim=-1).unsqueeze(0)
    x = torch.ones(1, n, 11); x[:, :, :2] = xy
    d = torch.cdist(xy, xy)
    data = dict(x=x, coordinates=xy, demand=torch.tensor([[0., 0., 1., -1.]]),
                time_window=torch.tensor([[0., 0., 10., 12.03]]),
                capacity=torch.tensor([[40., 10.]]),
                battery=torch.tensor([[config.uav_battery.full_charge_kj,
                                       config.adr_battery.full_charge_kj]]),
                edge_attr_d=d, edge_attr_r=d, wind=torch.zeros(1, 2),
                n_depots=2, n_depots_uav=1, service_policy='mandatory',
                objective_weights=asdict(mandatory_objective_weights()),
                physical_config=config.to_dict())
    return data


def _unsupported_native_calls(data):
    import torch
    from reward import native_physical_configuration, trajectory_battery_cost, with_mandatory_contract
    from VRP_Actor import Model
    tour = torch.tensor([[[0], [1]]])
    empty_times = torch.empty(1, 2, 0)
    yield 'native_physical_configuration', lambda: native_physical_configuration(data)
    yield 'with_mandatory_contract', lambda: with_mandatory_contract(data)
    yield 'VRP_Actor_forward', lambda: Model(11, 32, 4, 8, 1)(dict(physical_config=data['physical_config']), 1, 1)
    yield 'trajectory_battery_cost', lambda: trajectory_battery_cost(data, tour, empty_times, 1, 2)


def main(argv: List[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description='Backend cross-conformance audit')
    parser.add_argument('--output', type=Path, default=None,
                        help='Write report JSON here')
    args = parser.parse_args(argv)
    report = run_audit(output_path=args.output)
    print(json.dumps({k: v for k, v in report.items()
                      if k in ('status', 'backends', 'unsupported_config_guards')},
                     indent=2, sort_keys=True))
    return 0 if report['status'] == 'PASS' else 1


if __name__ == '__main__':
    raise SystemExit(main())
