"""RL backend conformance under the shared mandatory contract (pre-flight).

Intended to be run AFTER a trained HetGAT / Kool AM checkpoint lands. Random-
init actors on the classical-audit scenario fail with ``KeyError('demand')``
because the RL path consumes a tensor-first instance dict (``demand``,
``time_window``, ``edge_attr_uav``, ...) while the classical audit's
``_scenario()`` is numpy-only. The parity point is not about input tensor
shapes — it is about every solver going through the same
``RollingHorizonDispatcher.run_shift(..., service_policy='mandatory')``
and the same post-run audit (``_check_mandatory_run`` + ``audit_cost_trace``).

Argument ``--checkpoint`` is required: with a trained checkpoint and a
frozen-bank instance (which carries the tensor fields the RL path needs),
this script runs the same mandatory check as the classical audit.

"""
from __future__ import annotations

import argparse
import copy
import json
import sys
import time
from pathlib import Path
from typing import Any, Dict, Sequence

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from artifacts import load_tensor_file

from release_checks import release_status
from reward import MANDATORY_OBJECTIVE_VERSION
from tools.diagnostics.backend_conformance_audit import (
    FIVE_COST_KEYS, _check_mandatory_run)


def _load_bank_instance(bank_path: Path, instance_index: int = 0) -> Dict[str, Any]:
    """Load one raw numpy-format instance from a frozen paired bank.

    The bank stores coords, d_uav/d_adr, wind, arrivals and owners (classical
    format). Use ``_bank_instance_to_tensor`` to adapt the entry to the
    tensor-first contract required by the RL path.
    """
    import torch
    raw = load_tensor_file(bank_path, map_location='cpu', weights_only=True)
    if not isinstance(raw, dict) or 'instances' not in raw:
        raise ValueError(f'{bank_path} is not a frozen paired bank')
    return raw['instances'][instance_index]


def _bank_instance_to_tensor(bank_raw: Dict[str, Any], instance_index: int = 0) -> Dict[str, Any]:
    """Adapt one frozen-bank entry to the tensor-first RL instance contract.

    The paired bank (``tools/experiments/paired_bank.py``) persists each
    instance's geometry (``coordinates``, ``d_uav``, ``d_adr``, ``wind``),
    its ``arrivals`` list, and ``owners`` — but no tensor fields. The RL
    path (``main.RLSolver``) consumes a tensor-first dict with ``demand``,
    ``time_window``, ``x``, ``edge_attr_d``/``edge_attr_r``,
    ``edge_attr_uav``/``edge_attr_adr``, ``mask_adjacency_*``, ``coordinates``,
    ``wind``, ``capacity``, ``battery``, ``n_depots``, ``n_depots_uav``,
    ``n_req``, ``service_policy``.

    This helper constructs those tensors from the bank's stored geometry and
    arrivals in the exact format produced by ``creat_vrp.create_instance``
    (see lines ~342-455 there), so RL and classical solvers share identical
    instances for paired-equal-budget evaluation. We mirror the generator's
    construction verbatim (same scalings, same clip ranges, same dtypes), but
    pull demand / arrival / pickup / delivery times from the bank's arrivals
    rather than re-sampling. Depot accessibility is Euclidean-based (all
    nodes reachable) which matches the bank's full-connectivity ``d_uav``/
    ``d_adr`` matrices (no NFZ / barrier semantics in the paired bank).

    The RL audit consumes the same instance that greedy / Kool AM / MILP
    consume (shared contract).
    """
    import math
    import numpy as np
    import torch
    from creat_vrp import static_signed_margin
    from energy import adr_energy_matrix, uav_energy_matrix
    from model_config import (ADR_BATTERY_KJ, ADR_CAPACITY_KG, ADR_CRUISE_MPS,
                              ADR_PROFILE, ENERGY_MODEL_VERSION,
                              GENERATOR_VERSION, MODEL_VERSION,
                              SCALE_M_PER_COORD, UAV_BATTERY_KJ,
                              UAV_CAPACITY_KG, UAV_CRUISE_MPS, UAV_PROFILE)

    V_UAV_COORD_PER_MIN = UAV_CRUISE_MPS * 60.0 / SCALE_M_PER_COORD
    V_ADR_COORD_PER_MIN = ADR_CRUISE_MPS * 60.0 / SCALE_M_PER_COORD

    if not isinstance(bank_raw, dict) or 'instances' not in bank_raw:
        raise ValueError('bank_raw is not a frozen paired bank')
    inst = bank_raw['instances'][instance_index]
    spec = bank_raw.get('spec', {})

    def _to_np(x, dtype=np.float32):
        if hasattr(x, 'cpu'):
            x = x.cpu().numpy()
        return np.asarray(x, dtype=dtype)

    n_req = int(inst['n_req'])
    n_depots = int(inst['n_depots'])
    n_depots_uav = int(inst['n_depots_uav'])
    n_total = n_depots + 2 * n_req
    n_uav = int(spec.get('n_uav', 2))
    n_adr = int(spec.get('n_adr', 2))

    coords = _to_np(inst['coordinates'], np.float32)
    d_uav = _to_np(inst['d_uav'], np.float32)
    d_adr = _to_np(inst['d_adr'], np.float32)
    wind_stored = _to_np(inst['wind'], np.float32).reshape(-1)
    # Bank persists wind as (wind_mps_x, 0) — i.e. magnitude along +x axis;
    # that matches ``_wind_vec`` in dispatch_sim.py when stored as [mag, dir=0].
    wind_mag = float(wind_stored[0])
    wind_dir = float(wind_stored[1]) if wind_stored.size > 1 else 0.0

    # Sort arrivals by t_arrival so request ids are reveal-prefix ordered,
    # matching create_instance's convention.
    arrivals_sorted = sorted(inst['arrivals'],
                             key=lambda r: float(r.get('t_arrival', r.get('t_pickup', 0.0))))
    if len(arrivals_sorted) != n_req:
        raise ValueError(f'arrivals count {len(arrivals_sorted)} != n_req {n_req}')

    demand_vals = np.array([float(r['demand']) for r in arrivals_sorted], dtype=np.float32)
    t_arrival = np.array([float(r['t_arrival']) for r in arrivals_sorted], dtype=np.float32)
    t_pickup = np.array([float(r['t_pickup']) for r in arrivals_sorted], dtype=np.float32)
    t_delivery = np.array([float(r['t_delivery']) for r in arrivals_sorted], dtype=np.float32)
    delivery_slack = (t_delivery - t_pickup).clip(0.0).astype(np.float32)

    # Signed demand on the full node layout: +d at pickup, -d at delivery.
    demand = np.zeros(n_total, dtype=np.float32)
    demand[n_depots:n_depots + n_req] = demand_vals
    demand[n_depots + n_req:] = -demand_vals

    # Energy matrices — same signatures as creat_vrp.create_instance.
    wind_vec = np.array([wind_mag * math.cos(wind_dir),
                         wind_mag * math.sin(wind_dir)], dtype=np.float32)
    e_uav = uav_energy_matrix(
        d_uav * SCALE_M_PER_COORD, demand_vals, demand, n_depots, n_req,
        wind_vec, UAV_CRUISE_MPS, coordinates=coords)
    e_adr = adr_energy_matrix(
        d_adr * SCALE_M_PER_COORD, demand_vals, demand, n_depots, n_req,
        ADR_CRUISE_MPS)

    # Pickup time per node (zeros at depots and at delivery nodes).
    tp_full = np.zeros(n_total, dtype=np.float32)
    tp_full[n_depots:n_depots + n_req] = t_pickup
    margin_uav = static_signed_margin(tp_full, d_uav, V_UAV_COORD_PER_MIN)
    margin_adr = static_signed_margin(tp_full, d_adr, V_ADR_COORD_PER_MIN)

    shift_minutes = 120.0
    # ``high`` in create_instance is the coord range upper bound. The paired
    # bank uses a 1x1 unit box (see paired_bank._coords), so DIST_SCALE
    # equals sqrt(2); matches the normalisation in create_instance when
    # coords live in [0,1]^2. The actor uses DIST_SCALE only as a clip
    # denominator, so a conservative value keeps the clip inert.
    coord_hi = float(max(coords.max(), 1.0))
    DIST_SCALE = coord_hi * math.sqrt(2)
    ENERGY_SCALE = 300_000.0

    tp_i = tp_full[:, None]
    tp_j = tp_full[None, :]
    t_gap = np.abs(tp_i - tp_j) / shift_minutes

    edge_attr_uav = np.stack([
        np.clip(margin_uav.reshape(-1) / shift_minutes, -2.0, 2.0),
        np.clip(d_uav.reshape(-1) / DIST_SCALE, 0.0, 2.0),
        np.clip(e_uav.reshape(-1) / ENERGY_SCALE, 0.0, 1.0),
        t_gap.reshape(-1),
    ], axis=-1).astype(np.float32)
    edge_attr_adr = np.stack([
        np.clip(margin_adr.reshape(-1) / shift_minutes, -2.0, 2.0),
        np.clip(d_adr.reshape(-1) / DIST_SCALE, 0.0, 2.0),
        np.clip(e_adr.reshape(-1) / ENERGY_SCALE, 0.0, 1.0),
        t_gap.reshape(-1),
    ], axis=-1).astype(np.float32)

    # Request-mode permission parity fix.
    # The adapter previously hardcoded `acc_uav = acc_adr = ones(n_total)`,
    # ignoring the bank's per-node accessibility arrays AND the per-request
    # `allowed_modes` field. Classical methods read these from the raw bank
    # entry, so the RL path was benching against a strictly more permissive
    # feasible set. Honor:
    #   (1) inst['acc_uav']/inst['acc_adr'] at the node level if present
    #       (NFZ / barrier geometries bake into these in generated banks).
    #   (2) arr['allowed_modes'] per arrival for pickup + delivery nodes
    #       (a UAV-only request should zero out ADR for that (p, d) pair
    #       and vice-versa).
    # Fall through to all-ones only when the bank lacks those fields, so
    # legacy benches on simple Euclidean banks keep their old behavior.
    if 'acc_uav' in inst:
        acc_uav = _to_np(inst['acc_uav'], np.float32).reshape(-1)
        if acc_uav.size != n_total:
            raise ValueError(f'acc_uav size {acc_uav.size} != n_total {n_total}')
    else:
        acc_uav = np.ones(n_total, dtype=np.float32)
    if 'acc_adr' in inst:
        acc_adr = _to_np(inst['acc_adr'], np.float32).reshape(-1)
        if acc_adr.size != n_total:
            raise ValueError(f'acc_adr size {acc_adr.size} != n_total {n_total}')
    else:
        acc_adr = np.ones(n_total, dtype=np.float32)
    # Per-request allowed_modes: zero out mode accessibility on pickup +
    # delivery nodes for requests the mode cannot serve.
    for req_idx, arr in enumerate(arrivals_sorted):
        allowed = arr.get('allowed_modes')
        if not allowed:
            continue
        pickup_node_idx = n_depots + req_idx
        delivery_node_idx = n_depots + n_req + req_idx
        if 'uav' not in allowed:
            acc_uav[pickup_node_idx] = 0.0
            acc_uav[delivery_node_idx] = 0.0
        if 'adr' not in allowed:
            acc_adr[pickup_node_idx] = 0.0
            acc_adr[delivery_node_idx] = 0.0

    node_type = np.zeros((n_total, 3), dtype=np.float32)
    node_type[:n_depots, 0] = 1.0
    node_type[n_depots:n_depots + n_req, 1] = 1.0
    node_type[n_depots + n_req:, 2] = 1.0

    t_pickup_feat = np.zeros(n_total, dtype=np.float32)
    t_pickup_feat[n_depots:n_depots + n_req] = t_pickup
    t_delivery_feat = np.zeros(n_total, dtype=np.float32)
    t_delivery_feat[n_depots + n_req:] = t_delivery

    DEMAND_MAX = max(float(demand_vals.max()) if demand_vals.size else 1.0,
                     float(ADR_CAPACITY_KG))
    delivery_slack_node = np.zeros(n_total, dtype=np.float32)
    delivery_slack_node[n_depots:n_depots + n_req] = delivery_slack / shift_minutes
    delivery_slack_node[n_depots + n_req:] = delivery_slack / shift_minutes

    COORD_SCALE = coord_hi
    TIME_SCALE = shift_minutes + 60.0

    node_feat = np.concatenate([
        coords / COORD_SCALE,
        demand[:, None] / DEMAND_MAX,
        t_pickup_feat[:, None] / TIME_SCALE,
        t_delivery_feat[:, None] / TIME_SCALE,
        node_type,
        acc_uav[:, None],
        acc_adr[:, None],
        delivery_slack_node[:, None],
    ], axis=-1).astype(np.float32)

    # Mode-parity fix. Previously only filtered on the
    # source axis (`acc_* [:, None]`), which lets the vehicle be chosen
    # TO move to an inaccessible node even though it couldn't then leave.
    # Add destination-axis filter (`acc_* [None, :]`) so a UAV-forbidden
    # node (e.g. rooftop-only / mode-restricted) is blocked both ways:
    # no vehicle can arrive there AND no vehicle can leave (ground truth
    # for mode permissions). Backward-compatible: on existing all-ones
    # banks the extra factor is identity.
    mask_adj_uav = ((d_uav < 1e9).astype(np.float32)
                    * acc_uav[:, None] * acc_uav[None, :])
    mask_adj_adr = ((d_adr < 1e9).astype(np.float32)
                    * acc_adr[:, None] * acc_adr[None, :])

    edges_index = np.stack(
        np.indices((n_total, n_total)).reshape(2, -1), axis=0)

    capacity = np.hstack([
        np.ones(n_uav, dtype=np.float32) * float(UAV_CAPACITY_KG),
        np.ones(n_adr, dtype=np.float32) * float(ADR_CAPACITY_KG),
    ])
    battery = np.hstack([
        np.ones(n_uav, dtype=np.float32) * float(UAV_BATTERY_KJ),
        np.ones(n_adr, dtype=np.float32) * float(ADR_BATTERY_KJ),
    ])

    tensor_instance: Dict[str, Any] = {
        'x': torch.tensor(node_feat, dtype=torch.float32),
        'coordinates': torch.tensor(coords, dtype=torch.float32),
        'energy_model_version': ENERGY_MODEL_VERSION,
        'generator_version': GENERATOR_VERSION,
        'model_version': MODEL_VERSION,
        'uav_hardware_profile': UAV_PROFILE,
        'adr_hardware_profile': ADR_PROFILE,
        'battery_unit': 'kJ',
        'edge_index': torch.tensor(edges_index, dtype=torch.long),
        'edge_attr_uav': torch.tensor(edge_attr_uav, dtype=torch.float32),
        'edge_attr_adr': torch.tensor(edge_attr_adr, dtype=torch.float32),
        'mask_adjacency_uav': torch.tensor(mask_adj_uav.reshape(-1, 1),
                                           dtype=torch.float32),
        'mask_adjacency_adr': torch.tensor(mask_adj_adr.reshape(-1, 1),
                                           dtype=torch.float32),
        'demand': torch.tensor(demand, dtype=torch.float32).unsqueeze(-1),
        'capacity': torch.tensor(capacity, dtype=torch.float32),
        'battery': torch.tensor(battery, dtype=torch.float32),
        'edge_attr_d': torch.tensor(d_uav.reshape(-1, 1), dtype=torch.float32),
        'edge_attr_r': torch.tensor(d_adr.reshape(-1, 1), dtype=torch.float32),
        'time_window': torch.tensor(
            np.concatenate([np.zeros(n_depots, dtype=np.float32),
                            t_pickup, t_delivery])[:, None],
            dtype=torch.float32),
        'wind': torch.tensor([wind_mag, wind_dir], dtype=torch.float32),
        'n_depots': n_depots,
        'n_depots_uav': n_depots_uav,
        'n_req': n_req,
        'n_uav': n_uav,
        'n_adr': n_adr,
        'n_depots_adr': int(spec.get('n_depots_adr', n_depots - n_depots_uav)),
        't_arrival': torch.tensor(t_arrival, dtype=torch.float32),
        'arrival_peaks': torch.tensor(np.zeros(0, dtype=np.float32),
                                       dtype=torch.float32),
        'service_policy': 'mandatory',
        # Preserve arrivals / owners so downstream code (contracts, logs,
        # reward functions) can consume the paired-bank ownership split.
        'arrivals': [dict(r) for r in arrivals_sorted],
        'owners': dict(inst.get('owners', {})),
    }
    return tensor_instance


def _build_tensor_instance(n_req: int = 5, seed: int = 20261017) -> Dict[str, Any]:
    """Build a Rung-A tensor instance via ``creat_vrp.create_instance``.

    The RL path consumes a tensor-first instance dict (`demand`,
    `time_window`, `x`, `edge_attr_uav/_adr`, …). Frozen paired banks are
    numpy-only (built for classical solvers), so we construct a fresh
    tensor instance with a fixed seed for reproducibility. The scenario
    size and fleet mirror Rung-A's config.
    """
    import numpy as np
    from creat_vrp import create_instance
    rng = np.random.default_rng(seed)
    result = create_instance(
        n_req=n_req, n_uav=2, n_adr=2,
        n_depots_uav=1, n_depots_adr=1,
        rng=rng,
    )
    instance = result if isinstance(result, dict) else result[0]
    instance['service_policy'] = 'mandatory'
    return instance


def run_rl_audit(checkpoint_path: Path, arch: str, bank_path: Path | None = None,
                  instance_index: int = 0, n_req: int = 5, seed: int = 20261017,
                  output_path: Path | None = None) -> Dict[str, Any]:
    """Load the checkpoint, run through RollingHorizonDispatcher, check parity.

    If ``bank_path`` is None, builds a fresh tensor instance via
    ``create_instance(n_req, seed=seed)``. If ``bank_path`` is set, loads
    one frozen paired-bank entry and adapts it to the tensor-first contract
    via ``_bank_instance_to_tensor`` so the RL path consumes the SAME
    instance the classical baselines consume (shared contract).
    """
    import torch
    from dispatch_sim import RollingHorizonDispatcher, Vehicle
    from main import build_solver
    from audit_cost_trace import audit_cost_trace
    from model_config import (ADR_BATTERY_KJ, ADR_CAPACITY_KG,
                               UAV_BATTERY_KJ, UAV_CAPACITY_KG)

    from benchmark_rolling_baselines_with_plots import _arrivals_from_instance
    rl_solver = build_solver(str(checkpoint_path), arch=arch)
    if bank_path is not None:
        bank_raw = load_tensor_file(bank_path, map_location='cpu', weights_only=True)
        instance = _bank_instance_to_tensor(bank_raw, instance_index=instance_index)
        arrivals = [copy.deepcopy(r) for r in instance['arrivals']]
    else:
        instance = _build_tensor_instance(n_req=n_req, seed=seed)
        # ``create_instance`` returns tensor instance fields but no ``arrivals``
        # list; ``_arrivals_from_instance`` synthesises the arrival stream from
        # the demand / time_window / t_arrival tensors.
        arrivals = _arrivals_from_instance(instance)
    # Build the fleet explicitly from model_config constants — matches the
    # convention in tools/experiments/milp_scaling_study._materialize_fleet,
    # which is the pattern already used by the strong-coalition and MILP paths.
    n_uav = int(instance.get('n_uav', 2))
    n_adr = int(instance.get('n_adr', 2))
    n_depots_uav = int(instance.get('n_depots_uav', 1))
    n_depots_adr = int(instance.get('n_depots_adr', 1))
    fleet = {}
    vid = 0
    for u in range(n_uav):
        fleet[vid] = Vehicle(vid, 'uav', u % n_depots_uav, 0., UAV_BATTERY_KJ,
                              0., UAV_CAPACITY_KG, battery_init=UAV_BATTERY_KJ,
                              allowed_depots=tuple(range(n_depots_uav)))
        vid += 1
    for a in range(n_adr):
        depot = n_depots_uav + a % n_depots_adr
        fleet[vid] = Vehicle(vid, 'adr', depot, 0., ADR_BATTERY_KJ, 0.,
                              ADR_CAPACITY_KG, battery_init=ADR_BATTERY_KJ,
                              allowed_depots=tuple(range(n_depots_uav,
                                                           n_depots_uav + n_depots_adr)))
        vid += 1

    disp = RollingHorizonDispatcher(rl_solver, delta_minutes=5.,
                                     shift_minutes=120.,
                                     service_policy='mandatory')
    start = time.perf_counter()
    log = disp.run_shift(arrivals, fleet, instance)
    elapsed = time.perf_counter() - start
    summary = log[-1]
    five = _check_mandatory_run(f'rl_{arch}', log, summary)
    audit_cost_trace(log, fleet, arrivals)

    report = dict(
        audit='rl_backend_conformance',
        mandatory_objective_version=MANDATORY_OBJECTIVE_VERSION,
        release_status=release_status(),
        checkpoint_path=str(Path(checkpoint_path).resolve()),
        bank_path=str(Path(bank_path).resolve()) if bank_path is not None else None,
        instance_index=instance_index,
        arch=arch,
        service_complete=bool(summary['service_complete']),
        total_delivered=int(summary['total_delivered']),
        total_revealed=int(summary['total_revealed']),
        quality_cost=float(summary['quality_cost']),
        total_cost=float(summary['total_cost']),
        completion_time_min=float(summary['completion_time_min']),
        five_cost=five,
        elapsed_s=elapsed,
        policy='mandatory',
        trace_audit_passed=True,
        five_cost_keys=list(FIVE_COST_KEYS),
        status='PASS' if summary['service_complete'] else 'FAIL',
        generated_utc=time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()),
    )
    if output_path is not None:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps(report, indent=2, sort_keys=True))
    return report


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description='RL backend conformance under the shared mandatory contract')
    parser.add_argument('--checkpoint', type=Path, required=True,
                         help='Trained HetGAT or Kool AM best_actor.pt')
    parser.add_argument('--arch', choices=['hetgat', 'simplegat', 'koolam'],
                         default='hetgat')
    parser.add_argument('--bank-path', type=Path, default=None,
                         help='Optional frozen paired bank .pt (not yet converted '
                              'to tensor format; omit to use create_instance seed)')
    parser.add_argument('--instance-index', type=int, default=0)
    parser.add_argument('--n-req', type=int, default=5)
    parser.add_argument('--seed', type=int, default=20261017)
    parser.add_argument('--output', type=Path, default=None)
    args = parser.parse_args(argv)
    report = run_rl_audit(args.checkpoint, args.arch, args.bank_path,
                           instance_index=args.instance_index,
                           n_req=args.n_req, seed=args.seed,
                           output_path=args.output)
    print(json.dumps(dict(status=report['status'],
                          arch=report['arch'],
                          service_complete=report['service_complete'],
                          five_cost=report['five_cost'],
                          elapsed_s=report['elapsed_s']),
                     indent=2, sort_keys=True))
    return 0 if report['status'] == 'PASS' else 1


if __name__ == '__main__':
    raise SystemExit(main())
