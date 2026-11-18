"""Paired equal-budget benchmark across exact / heuristic / metaheuristic / learned methods.

Provides paired, equal-budget comparisons with at least one strong
search/optimization baseline at 25 and 60 requests, reporting service and
cost jointly.

Design — fair comparison contract
---------------------------------
Every entry in ``METHOD_REGISTRY`` is a callable
``run(full_instance, *, time_budget_s, seed) -> dict`` that returns a
*normalized result row* with the same schema regardless of method family.
This makes a paired, equal-budget comparison a `for method in methods: ...`
loop — no per-method special-casing in the driver.

Fairness clauses enforced here:

1. **Same instance** — pairing key is ``instance_sha256`` from the frozen
   paired bank. Every method sees byte-identical ``full_instance`` dicts.
2. **Same objective** — the five-cost mandatory objective. The driver does
   not permit per-method weight overrides; use the λ_U sweep separately
   for a λ_U-robustness check.
3. **Same end-to-end wall time budget** — ``time_budget_s`` is the total
   budget including construction, encoding, solving, repair, and
   validation. Each method is responsible for interpreting the budget
   (e.g. passing it as a solver time limit), and the driver measures
   actual wall time with ``perf_counter`` and records whether the method
   exceeded the budget.
4. **Same seed** — a per-instance seed is derived from the instance hash
   so stochastic methods (ALNS random moves, sampling policies) are
   deterministic across reruns.
5. **Service and cost jointly** — the output CSV includes
   ``delivered_count``, ``lost_count``, ``service_complete`` alongside the
   five-cost breakdown.

Classical methods available in the registry (see ``METHOD_REGISTRY``):

- ``gurobi_milp`` — exact MILP (requires licence); reports solver status
  and MIP gap. Scale limitation: n <= ~10 in practice.
- ``ortools_cpsat`` — OR-Tools CP-SAT guided local search. **Not an exact
  MILP**: use the ``ortools_cpsat`` key, not any ``*milp*`` key.
- ``alns`` — adaptive large neighbourhood search metaheuristic.
- ``greedy_insertion`` — classical regret-based insertion heuristic.
- ``fifo`` — first-in-first-out assignment (weakest baseline floor).
- ``offline_alns`` — ALNS with full-horizon arrival information (not
  online-fair; included as a quality-reference upper bound, labelled
  as such in reports).

Learned methods are registered by the caller after loading a trained
checkpoint:

```python
from baselines import KoolAMModel
driver.register('hetgat', make_rl_runner(hetgat_checkpoint))
driver.register('kool_am', make_rl_runner(kool_am_checkpoint))
driver.run(bank_path, output_csv, methods=['hetgat', 'kool_am', 'alns',
                                              'ortools_cpsat', 'greedy_insertion'],
            time_budget_s=8.0)
```

Learned-method wiring is intentionally caller-registered because the
checkpoint path and architecture aren't known at import time.
"""
from __future__ import annotations

import argparse
import copy
import csv
import hashlib
import json
import math
import sys
import time
import traceback
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Sequence

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from artifacts import load_tensor_file

BENCHMARK_FORMAT_VERSION = 'paired-equal-budget-benchmark-1'


# ─── Canonical result row schema ─────────────────────────────────────────────

RESULT_FIELDS = [
    'bank_id', 'instance_id', 'instance_sha256', 'n_req', 'method',
    'time_budget_s', 'wall_time_s', 'exceeded_budget',
    # Service metrics: delivered fraction, lost orders
    'delivered_count', 'lost_count', 'service_complete', 'service_rate',
    # Five-cost decomposition (mandatory objective)
    'cost_total', 'cost_operating_uav', 'cost_operating_adr',
    'cost_pickup_early', 'cost_pickup_late', 'cost_delivery_late',
    # Required columns: energy use, battery, recharge, completion time
    'energy_kj', 'travel_energy_kj', 'service_energy_kj',
    'battery_penalty_cost', 'recharge_events_count',
    'travel_minutes', 'completion_time_minutes',
    # Audit columns: per-method arc-energy accounting (populated when
    # the simulator exposes per-leg state; NaN otherwise)
    'max_arc_energy_kj', 'min_battery_after_kj',
    # Family-specific extras
    'solver_status', 'solver_mip_gap', 'solver_lower_bound',
    'replan_count',
    # Diagnostics
    'status', 'error',
]


def _normalized_row(**kw) -> Dict[str, Any]:
    """Fill in every schema field with a safe default; override via kw."""
    row = {k: '' for k in RESULT_FIELDS}
    row.update({k: v for k, v in kw.items() if k in RESULT_FIELDS})
    return row


# ─── Method registry ─────────────────────────────────────────────────────────

@dataclass
class MethodRegistry:
    """Mutable registry of callable methods.

    Each callable: ``fn(full_instance, *, time_budget_s, seed) -> dict``
    returning a dict following ``RESULT_FIELDS`` (missing fields fine; the
    driver fills defaults). The driver measures end-to-end wall time
    independently — the callable's reported ``wall_time_s`` is ignored.
    """
    _methods: Dict[str, Callable[..., Dict[str, Any]]] = field(default_factory=dict)

    def register(self, name: str, fn: Callable[..., Dict[str, Any]]) -> None:
        if name in self._methods:
            raise ValueError(f'method {name!r} already registered')
        self._methods[name] = fn

    def available(self) -> List[str]:
        return sorted(self._methods)

    def __contains__(self, name: str) -> bool:
        return name in self._methods

    def __getitem__(self, name: str) -> Callable[..., Dict[str, Any]]:
        return self._methods[name]


def _build_default_registry() -> MethodRegistry:
    """Register classical baselines with lazy imports.

    Learned-method wrappers are *not* registered here — the caller must
    construct and register those after loading a trained checkpoint.
    """
    reg = MethodRegistry()

    def _wrap_alns(full_instance, *, time_budget_s: float, seed: int):
        from ce_cpdptw_alns import (ALNSBaseline, SyntheticALNSConfig,
                                      solve_static_instance_with_alns)
        config = SyntheticALNSConfig(time_budget_s=time_budget_s, seed=seed)
        episode_log = solve_static_instance_with_alns(
            full_instance, config, solver=ALNSBaseline(seed=seed))
        return _coerce_episode_log(episode_log, full_instance, method='alns')

    def _wrap_ortools_cpsat(full_instance, *, time_budget_s: float, seed: int):
        from ce_cpdptw_rolling_baselines import (ExactRollingConfig,
                                                   solve_static_instance_with_ortools_rolling)
        cfg = ExactRollingConfig(
            n_uav=int(full_instance.get('n_uav', 2)),
            n_adr=int(full_instance.get('n_adr', 2)),
            n_depots_uav=int(full_instance.get('n_depots_uav', 1)),
            n_depots_adr=int(full_instance.get('n_depots_adr', 1)),
            delta_minutes=float(full_instance.get('delta_minutes', 5.0)),
            time_limit_seconds=float(time_budget_s))
        episode_log = solve_static_instance_with_ortools_rolling(full_instance, cfg)
        return _coerce_episode_log(episode_log, full_instance, method='ortools_cpsat')

    def _wrap_gurobi(full_instance, *, time_budget_s: float, seed: int):
        from ce_cpdptw_rolling_baselines import (ExactRollingConfig,
                                                   solve_static_instance_with_gurobi_rolling)
        cfg = ExactRollingConfig(
            n_uav=int(full_instance.get('n_uav', 2)),
            n_adr=int(full_instance.get('n_adr', 2)),
            n_depots_uav=int(full_instance.get('n_depots_uav', 1)),
            n_depots_adr=int(full_instance.get('n_depots_adr', 1)),
            delta_minutes=float(full_instance.get('delta_minutes', 5.0)),
            time_limit_seconds=float(time_budget_s))
        episode_log = solve_static_instance_with_gurobi_rolling(full_instance, cfg)
        return _coerce_episode_log(episode_log, full_instance, method='gurobi_milp')

    def _wrap_offline_alns(full_instance, *, time_budget_s: float, seed: int):
        # Full-horizon ALNS: not online-fair. Report as upper-bound reference.
        from offline_alns import solve_static_instance_with_offline_alns
        episode_log = solve_static_instance_with_offline_alns(
            full_instance, time_budget_s=time_budget_s, seed=seed)
        return _coerce_episode_log(episode_log, full_instance, method='offline_alns')

    reg.register('alns', _wrap_alns)
    reg.register('ortools_cpsat', _wrap_ortools_cpsat)
    reg.register('gurobi_milp', _wrap_gurobi)
    reg.register('offline_alns', _wrap_offline_alns)
    return reg


# ─── Episode log coercer ─────────────────────────────────────────────────────

def _coerce_episode_log(episode_log, full_instance, *, method: str) -> Dict[str, Any]:
    """Normalize whatever a solver returned to the common RESULT_FIELDS schema.

    Different solvers return different shapes. This function tries the most
    common schemas; add branches as new solvers land.
    """
    if isinstance(episode_log, list):
        # The rolling-horizon dispatchers return a list of per-replan logs;
        # fold them into one episode summary using the same helper the main
        # benchmark script uses.
        try:
            from benchmark_rolling_baselines_with_plots import get_summary
            summary = get_summary(episode_log)
            episode_log = summary
        except Exception:
            episode_log = {}
    if not isinstance(episode_log, dict):
        episode_log = {}

    delivered = int(episode_log.get('total_delivered',
                                      episode_log.get('delivered_count', 0)) or 0)
    n_req = int(full_instance.get('n_req', 0))
    lost = max(0, n_req - delivered)
    service_complete = bool(episode_log.get('service_complete', delivered == n_req))

    def _f(name, default=0.0):
        return float(episode_log.get(name, default) or default)

    # Energy aggregates: dispatch_sim exposes energy_used_kj (total),
    # travel_energy_kj, service_energy_kj. Direct solvers may use
    # cost_energy_kj or energy_kj. Fall back to NaN if nothing provided.
    energy_kj = _f('energy_used_kj',
                    default=_f('energy_kj', default=_f('cost_energy_kj', default=math.nan)))
    travel_energy_kj = _f('travel_energy_kj', default=math.nan)
    service_energy_kj = _f('service_energy_kj', default=math.nan)

    # Per-leg audit if the dispatch_sim log included completed legs.
    max_arc_energy_kj = math.nan
    min_battery_after_kj = math.nan
    if isinstance(episode_log, dict):
        for key in ('completed', 'legs'):
            legs = episode_log.get(key)
            if isinstance(legs, list) and legs:
                arc_energies = [float(leg.get('energy_used', math.nan))
                                 for leg in legs
                                 if isinstance(leg, dict) and leg.get('energy_used') is not None]
                bat_after = [float(leg.get('battery_after', math.nan))
                              for leg in legs
                              if isinstance(leg, dict) and leg.get('battery_after') is not None]
                if arc_energies:
                    max_arc_energy_kj = max(e for e in arc_energies if math.isfinite(e)) \
                        if any(math.isfinite(e) for e in arc_energies) else math.nan
                if bat_after:
                    min_battery_after_kj = min(b for b in bat_after if math.isfinite(b)) \
                        if any(math.isfinite(b) for b in bat_after) else math.nan
                break

    return dict(
        method=method,
        delivered_count=delivered,
        lost_count=lost,
        service_complete=service_complete,
        service_rate=(delivered / n_req) if n_req > 0 else 0.0,
        cost_total=_f('total_cost'),
        cost_operating_uav=_f('operating_cost_uav'),
        cost_operating_adr=_f('operating_cost_adr'),
        cost_pickup_early=_f('pickup_early_cost'),
        cost_pickup_late=_f('pickup_late_cost'),
        cost_delivery_late=_f('delivery_late_cost'),
        energy_kj=energy_kj,
        travel_energy_kj=travel_energy_kj,
        service_energy_kj=service_energy_kj,
        battery_penalty_cost=_f('battery_penalty_cost',
                                 default=_f('battery_penalty', default=math.nan)),
        recharge_events_count=int(episode_log.get('recharge_events', 0) or 0),
        travel_minutes=_f('travel_minutes', default=math.nan),
        completion_time_minutes=_f('completion_time_min',
                                     default=_f('completion_time_minutes', default=math.nan)),
        max_arc_energy_kj=max_arc_energy_kj,
        min_battery_after_kj=min_battery_after_kj,
        solver_status=str(episode_log.get('solver_status', '')),
        solver_mip_gap=_f('solver_mip_gap', default=math.nan),
        solver_lower_bound=_f('solver_lower_bound', default=math.nan),
        replan_count=int(episode_log.get('replan_count', 0) or 0),
    )


# ─── Driver ──────────────────────────────────────────────────────────────────

def _instance_seed(instance_sha256: str, base: int = 20261009) -> int:
    """Deterministic per-instance seed so stochastic methods are reproducible."""
    h = hashlib.sha256(f'{base}:{instance_sha256}'.encode()).hexdigest()
    return int(h[:12], 16) & 0x7FFFFFFF


def run_one(method_name: str, run_fn: Callable[..., Dict[str, Any]],
            full_instance: Dict[str, Any], *,
            instance_sha256: str,
            instance_id: str,
            bank_id: str,
            time_budget_s: float,
            fairness_slack_s: float = 2.0) -> Dict[str, Any]:
    """Run one (instance, method) cell and produce one result row.

    ``fairness_slack_s`` tolerates small overshoot (solver teardown, etc.)
    before flagging the row as ``exceeded_budget=True``. Default 2s.
    """
    seed = _instance_seed(instance_sha256)
    start = time.perf_counter()
    try:
        normalized = run_fn(full_instance, time_budget_s=time_budget_s, seed=seed)
        status = 'ok'
        error = ''
    except Exception as exc:
        normalized = {}
        status = f'error:{type(exc).__name__}'
        error = f'{exc}\n{traceback.format_exc(limit=3)}'
    wall = time.perf_counter() - start

    row = _normalized_row(
        bank_id=bank_id,
        instance_id=instance_id,
        instance_sha256=instance_sha256,
        n_req=int(full_instance.get('n_req', 0)),
        method=method_name,
        time_budget_s=float(time_budget_s),
        wall_time_s=float(wall),
        exceeded_budget=wall > time_budget_s + fairness_slack_s,
        status=status,
        error=error,
    )
    row.update(normalized)
    return row


def _done_pairs(output_csv: Path) -> set:
    """Return (instance_sha256, method) tuples already present in the CSV.

    Used by --resume to skip cells already evaluated. A cell counts as done
    only if its ``status`` column is 'ok' (so errored rows get retried).
    """
    done = set()
    if not output_csv.exists():
        return done
    with output_csv.open('r', newline='', encoding='utf-8') as stream:
        reader = csv.DictReader(stream)
        for row in reader:
            if row.get('status') == 'ok':
                done.add((row.get('instance_sha256', ''), row.get('method', '')))
    return done


def run_benchmark(bank_path: Path,
                   output_csv: Path,
                   methods: Sequence[str],
                   registry: MethodRegistry,
                   time_budget_s: float,
                   instance_limit: Optional[int] = None,
                   resume: bool = False,
                   progress_every: int = 1) -> Dict[str, Any]:
    """Full driver: for each instance in the bank, run each method under
    the shared time budget. Writes one row per (instance, method) cell.

    When ``resume=True`` and ``output_csv`` already exists, skips every
    (instance_sha256, method) cell whose status in the existing CSV is 'ok'
    and APPENDS new rows. Errored rows get retried. Header is written only
    when the CSV does not yet exist.
    """
    import torch
    raw = load_tensor_file(bank_path, map_location='cpu', weights_only=True)
    if not isinstance(raw, dict) or 'instances' not in raw:
        raise ValueError(f'{bank_path} is not a frozen paired bank')
    bank_id = str(raw.get('bank_id', bank_path.stem))
    instances = list(raw['instances'])
    if instance_limit is not None:
        instances = instances[:instance_limit]

    for m in methods:
        if m not in registry:
            raise KeyError(f'method {m!r} not registered; available: {registry.available()}')

    output_csv.parent.mkdir(parents=True, exist_ok=True)

    done = _done_pairs(output_csv) if resume else set()
    file_mode = 'a' if (resume and output_csv.exists()) else 'w'
    write_header = file_mode == 'w'
    if resume and done:
        print(f'[resume] skipping {len(done)} already-done (instance, method) cells',
              flush=True)

    n_rows = 0
    n_skipped = 0
    with output_csv.open(file_mode, newline='', encoding='utf-8') as stream:
        writer = csv.DictWriter(stream, fieldnames=RESULT_FIELDS)
        if write_header:
            writer.writeheader()
            stream.flush()
        for i, inst in enumerate(instances):
            sha = str(inst.get('instance_sha256', f'auto{i:04d}'))
            iid = str(inst.get('instance_id', sha))
            for method in methods:
                if (sha, method) in done:
                    n_skipped += 1
                    continue
                t0 = time.perf_counter()
                row = run_one(method, registry[method], inst,
                                instance_sha256=sha, instance_id=iid,
                                bank_id=bank_id, time_budget_s=time_budget_s)
                writer.writerow(row)
                n_rows += 1
                stream.flush()           # crash-safe at per-cell granularity
                if progress_every and n_rows % progress_every == 0:
                    dt = time.perf_counter() - t0
                    cr = row.get('service_rate', 'n/a')
                    status = row.get('status', '')
                    print(f'[row {n_rows}] inst={i+1}/{len(instances)} '
                          f'method={method} status={status} '
                          f'service_rate={cr} wall={dt:.1f}s', flush=True)

    manifest = dict(
        benchmark_format_version=BENCHMARK_FORMAT_VERSION,
        bank_id=bank_id,
        bank_path=str(bank_path.resolve()),
        output_csv=str(output_csv.resolve()),
        n_instances=len(instances),
        methods=list(methods),
        time_budget_s=float(time_budget_s),
        n_rows_written=n_rows,
        n_skipped_resumed=n_skipped,
        completed_utc=time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()),
    )
    output_csv.with_suffix('.manifest.json').write_text(
        json.dumps(manifest, indent=2, sort_keys=True))
    return manifest


# ─── CLI ─────────────────────────────────────────────────────────────────────

def main(argv: Sequence[str] | None = None) -> int:
    reg = _build_default_registry()
    parser = argparse.ArgumentParser(
        description='Paired equal-budget benchmark')
    parser.add_argument('--bank-path', type=Path, required=True)
    parser.add_argument('--output-csv', type=Path, required=True)
    parser.add_argument('--methods', nargs='+', default=['alns', 'ortools_cpsat'],
                         choices=reg.available(),
                         help='Classical methods to compare. Learned methods '
                              'require a wrapper; use the Python API.')
    parser.add_argument('--time-budget-s', type=float, default=8.0,
                         help='End-to-end wall-time budget per instance per method')
    parser.add_argument('--instance-limit', type=int, default=None,
                         help='Cap number of instances evaluated (pilot runs)')
    parser.add_argument('--resume', action='store_true',
                         help='Skip (instance, method) cells with status=ok already '
                              'present in output-csv; append remaining. Safe to '
                              'invoke repeatedly after crashes or Ctrl+C.')
    parser.add_argument('--progress-every', type=int, default=1,
                         help='Print a progress line every N written rows (default 1 = '
                              'per cell).')
    args = parser.parse_args(argv)
    manifest = run_benchmark(args.bank_path, args.output_csv,
                               methods=args.methods, registry=reg,
                               time_budget_s=args.time_budget_s,
                               instance_limit=args.instance_limit,
                               resume=args.resume,
                               progress_every=args.progress_every)
    print(json.dumps(manifest, indent=2, sort_keys=True))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
