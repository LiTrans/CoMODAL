"""Paired owned-book scenario bank with feasibility certification.

Generates deterministic paired scenarios at 5/10/25/60 requests with disjoint
UAV- and ADR-owned books over shared geometry, timing, and resources. Each
instance is certified for singleton and joint feasibility using the scalar
mandatory dispatcher, and classified as one of:

- ``all_feasible``: both singletons and the joint union serve to completion,
  eligible for finite standalone-savings cost games;
- ``joint_only``: joint feasible but at least one singleton infeasible, kept
  as pooled-operation evidence (not a finite cost game);
- ``infeasible``: joint also infeasible, excluded from the bank.

The frozen bank uses ``experiment_manifest.freeze_bank`` so later runs fail
closed on stale spec/source/content. This module is a bank artifact, not an
optimization or physical validator.
"""
from __future__ import annotations

import argparse
import copy
import dataclasses
import json
import math
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Mapping, Sequence, Tuple

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from coalition_contract import SharingContract
from creat_vrp import REQUEST_PAIRING_VERSION, _create_spatial_template, resample_request_pairs
from dispatch_sim import MandatoryServiceError, RollingHorizonDispatcher, Vehicle
from experiment_manifest import digest_json, freeze_bank, json_value, release_contract
from greedy_insertion import GreedyInsertion
from model_config import ADR_BATTERY_KJ, ADR_CAPACITY_KG, UAV_BATTERY_KJ, UAV_CAPACITY_KG
from release_checks import release_status

# paired-owned-book-bank-2: switch from 1x1 Euclidean geometry
# to the obstacle-aware generator (_create_spatial_template at 5x5 coords
# with NFZ + building clusters for UAV, road network for ADR, Floyd-Warshall
# relaxed). Earlier banks (-1) remain cite-able as the simplified-geometry
# baseline; new benchmarks run on -2 so claims about obstacle handling are
# supported by the evaluation set, not only by training.
BANK_FORMAT_VERSION = 'paired-owned-book-bank-3'
SHIFT_MINUTES = 120.0


@dataclass(frozen=True)
class BankSpec:
    n_req: int
    n_instances: int
    seed: int
    beta: float = 0.0          # payload-urgency correlation (0 removes bias)
    wind_mps: float = 0.0
    uav_fraction_of_book: float = 0.5
    n_uav: int = 2
    n_adr: int = 2
    n_depots_uav: int = 1
    n_depots_adr: int = 1
    eps_shared: float = 0.5    # cross-service budget for coalition evaluation
    # hardness overrides for stress banks. Defaults match
    # the pre-override canonical bank so existing banks regenerate identically.
    # Override for "hard" benchmark banks that actually stress battery/
    # recharge/TW/capacity — see bank_hardness_audit.py.
    coord_high: float = 5.0            # service-area half-side (km); larger -> longer edges
    tw_slack_mean: float = 30.0        # mean delivery-window slack (min)
    tw_slack_std: float = 5.0          # std of slack
    tw_slack_clip_low: float = 15.0    # clamp slack floor (min)
    tw_slack_clip_high: float = 60.0   # clamp slack ceiling (min)
    demand_low: float = 1.0            # per-request demand range (kg)
    demand_high: float = 8.0
    bank_format_version: str = BANK_FORMAT_VERSION

    def to_json(self) -> Dict[str, Any]:
        return dataclasses.asdict(self)


@dataclass
class InstanceCertification:
    instance_sha256: str
    singleton_uav: Dict[str, Any] = field(default_factory=dict)
    singleton_adr: Dict[str, Any] = field(default_factory=dict)
    joint: Dict[str, Any] = field(default_factory=dict)
    classification: str = 'unknown'


def _rng(seed: int) -> np.random.Generator:
    return np.random.default_rng(seed)


def _coords_simplified(rng: np.random.Generator, n_depots: int, n_req: int) -> np.ndarray:
    """Legacy -1 bank layout: depots at corners of a 1x1 box, customers
    uniform inside. Kept for diff/comparison with the earlier simplified bank;
    not called by the -2 generator."""
    coords = np.zeros((n_depots + 2 * n_req, 2), dtype=np.float64)
    corners = np.array([[.0, .0], [1., .0], [.0, 1.], [1., 1.]])
    for d in range(n_depots):
        coords[d] = corners[d % 4]
    coords[n_depots:] = rng.uniform(.05, .95, size=(2 * n_req, 2))
    return coords


def _demand_and_timing(rng: np.random.Generator, n_req: int, beta: float,
                        *, demand_low: float = 1., demand_high: float = 8.,
                        tw_slack_mean: float = 30., tw_slack_std: float = 5.,
                        tw_slack_clip_low: float = 15., tw_slack_clip_high: float = 60.
                        ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Return demand, t_arrival, t_pickup (ready), t_delivery (target).

    Hardness overrides passed from BankSpec default to the canonical values
    used in the earlier banks, so an unparameterised call reproduces
    the historical generator exactly.
    """
    demand = rng.uniform(demand_low, demand_high, size=n_req).astype(np.float64)
    # Spread arrivals through the first ~80% of the shift.
    t_arrival = np.sort(rng.uniform(0., SHIFT_MINUTES * .8, size=n_req)).astype(np.float64)
    pickup_lead = rng.uniform(8., 15., size=n_req).astype(np.float64)
    delivery_slack = rng.normal(tw_slack_mean, tw_slack_std, size=n_req) \
        .clip(tw_slack_clip_low, tw_slack_clip_high).astype(np.float64)
    # Payload-urgency correlation: beta correlates light demand with urgency and heavy with slack.
    if abs(beta) > 1e-9:
        order = np.argsort(demand)
        urgency = np.linspace(-1., 1., n_req) * beta
        pickup_lead = pickup_lead + urgency[order.argsort()] * 3.
        delivery_slack = delivery_slack - urgency[order.argsort()] * 10.
        pickup_lead = pickup_lead.clip(2., 25.)
        delivery_slack = delivery_slack.clip(15., 60.)
    t_pickup = t_arrival + pickup_lead
    t_delivery = t_pickup + delivery_slack
    return demand, t_arrival, t_pickup, t_delivery


def _ownership_split(rng: np.random.Generator, n_req: int, uav_fraction: float) -> Dict[int, str]:
    """Deterministic disjoint R_U / R_A split by randomly chosen index subset."""
    k = int(round(uav_fraction * n_req))
    k = max(1, min(n_req - 1, k)) if n_req >= 2 else n_req  # keep both books nonempty when possible
    uav_ids = set(rng.choice(n_req, size=k, replace=False).tolist()) if k > 0 else set()
    return {rid: ('uav' if rid in uav_ids else 'adr') for rid in range(n_req)}


def _build_instance(spec: BankSpec, index: int) -> Dict[str, Any]:
    """Generate shared geometry + arrivals + ownership for one instance.

    Format v2: geometry comes from ``_create_spatial_template``
    (5x5 coord box, NFZ + building clusters blocking UAV edges, road network
    for ADR, Floyd-Warshall relaxed) rather than the legacy unit-box
    Euclidean layout. Timing and ownership are unchanged from v1 so the
    paired-bank design (disjoint owned books, ``market_group='shared'``,
    allowed_modes=(uav,adr)) is preserved.
    """
    rng = _rng(spec.seed + 1000 * index)
    n_depots = spec.n_depots_uav + spec.n_depots_adr

    # Obstacle-aware geometry (v2). _create_spatial_template allocates the
    # same number of nodes (n_depots + 2 * n_req) in the same canonical
    # order (depots, pickups, deliveries). Its RNG state advances exactly
    # as before so timing/owners sampling below sees a deterministic rng.
    template = _create_spatial_template(
        spec.n_req, spec.n_depots_uav, spec.n_depots_adr, rng,
        lower=0.0, high=float(spec.coord_high), spatial_cluster_k=0,
    )
    demand, t_arrival, t_pickup, t_delivery = _demand_and_timing(
        rng, spec.n_req, spec.beta,
        demand_low=spec.demand_low, demand_high=spec.demand_high,
        tw_slack_mean=spec.tw_slack_mean, tw_slack_std=spec.tw_slack_std,
        tw_slack_clip_low=spec.tw_slack_clip_low,
        tw_slack_clip_high=spec.tw_slack_clip_high,
    )
    owners = _ownership_split(rng, spec.n_req, spec.uav_fraction_of_book)
    template, eligible = resample_request_pairs(
        template, demand, spec.n_depots_uav, spec.n_depots_adr, spec.n_uav, spec.n_adr, rng,
    )
    coords = template['coords'].astype(np.float64)
    d_uav = template['d_uav'].astype(np.float64)
    d_adr = template['d_adr'].astype(np.float64)
    acc_uav, acc_adr = template['acc_uav'], template['acc_adr']
    nfz_cx, nfz_cy, nfz_radius = template['nfz']
    arrivals: List[Dict[str, Any]] = []
    for rid in range(spec.n_req):
        pickup_node = int(n_depots + rid)
        delivery_node = int(n_depots + spec.n_req + rid)
        allowed = tuple(mode for mode, feasible in zip(('uav', 'adr'), eligible[:, rid]) if feasible)
        arrivals.append(dict(req_id=rid,
                             t_arrival=float(t_arrival[rid]),
                             t_pickup=float(t_pickup[rid]),
                             t_delivery=float(t_delivery[rid]),
                             pickup_node=pickup_node,
                             delivery_node=delivery_node,
                             demand=float(demand[rid]),
                             firm_owner=owners[rid],
                             market_group='shared',
                             allowed_modes=allowed))
    wind = np.array([spec.wind_mps, 0.], dtype=np.float64)
    instance_body: Dict[str, Any] = dict(
        n_depots=n_depots, n_depots_uav=spec.n_depots_uav, n_req=spec.n_req,
        d_uav=d_uav, d_adr=d_adr, coordinates=coords, wind=wind,
        acc_uav=acc_uav, acc_adr=acc_adr,
        nfz=np.array([nfz_cx, nfz_cy, nfz_radius], dtype=np.float64),
        arrivals=arrivals, owners=owners,
        bank_format_version=spec.bank_format_version, request_pairing_version=REQUEST_PAIRING_VERSION)
    instance_body['instance_sha256'] = digest_json(
        dict(coords=coords.tolist(),
             d_uav=d_uav.tolist(), d_adr=d_adr.tolist(),
             acc_uav=acc_uav.tolist(), acc_adr=acc_adr.tolist(),
             wind=wind.tolist(), arrivals=arrivals, owners=owners,
             spec=spec.to_json(), request_pairing_version=REQUEST_PAIRING_VERSION))
    return instance_body


def _build_fleet(spec: BankSpec, mode_filter: str | None = None) -> Dict[int, Vehicle]:
    """Build a mode-filtered fleet. ``None`` keeps the full fleet."""
    fleet: Dict[int, Vehicle] = {}
    vid = 0
    if mode_filter in (None, 'uav'):
        for u in range(spec.n_uav):
            fleet[vid] = Vehicle(vid, 'uav', u % spec.n_depots_uav, 0.,
                                 UAV_BATTERY_KJ, 0., UAV_CAPACITY_KG,
                                 battery_init=UAV_BATTERY_KJ,
                                 allowed_depots=tuple(range(spec.n_depots_uav)))
            vid += 1
    if mode_filter in (None, 'adr'):
        for a in range(spec.n_adr):
            depot = spec.n_depots_uav + a % spec.n_depots_adr
            fleet[vid] = Vehicle(vid, 'adr', depot, 0., ADR_BATTERY_KJ, 0., ADR_CAPACITY_KG,
                                 battery_init=ADR_BATTERY_KJ,
                                 allowed_depots=tuple(range(spec.n_depots_uav, spec.n_depots_uav + spec.n_depots_adr)))
            vid += 1
    return fleet


def _trial(spec: BankSpec, instance: Dict[str, Any], scope: str) -> Dict[str, Any]:
    """Run GreedyInsertion once and record feasibility + five-cost on success."""
    arrivals = [copy.deepcopy(row) for row in instance['arrivals']]
    if scope == 'uav_singleton':
        arrivals = [row for row in arrivals if row['firm_owner'] == 'uav']
        fleet = _build_fleet(spec, 'uav')
        members = ('uav',)
    elif scope == 'adr_singleton':
        arrivals = [row for row in arrivals if row['firm_owner'] == 'adr']
        fleet = _build_fleet(spec, 'adr')
        members = ('adr',)
    elif scope == 'joint':
        fleet = _build_fleet(spec, None)
        members = ('uav', 'adr')
    else:
        raise ValueError(f'Unknown scope: {scope}')
    if not arrivals:
        return dict(scope=scope, book_size=0, feasible=True, note='empty_book',
                    quality_cost=0., total_delivered=0)
    incompatible = [row['req_id'] for row in arrivals
                    if not any(mode in row['allowed_modes'] for mode in members)]
    if incompatible:
        return dict(scope=scope, book_size=len(arrivals), feasible=False,
                    reason='no_permitted_mode', incompatible_ids=incompatible)
    contract = SharingContract.from_arrivals(arrivals, members=members,
                                             eps_shared=spec.eps_shared)
    # Keep the full-instance node layout so pickup/delivery indices remain
    # valid; filtered arrivals simply never reveal the orders they omit.
    full_instance = dict(n_depots=instance['n_depots'],
                         n_depots_uav=instance['n_depots_uav'],
                         n_req=int(instance['n_req']),
                         d_uav=instance['d_uav'].copy(),
                         d_adr=instance['d_adr'].copy(),
                         coordinates=instance['coordinates'].copy(),
                         wind=instance['wind'].copy())
    for mode in ('uav', 'adr'):
        access = instance[f'acc_{mode}']
        full_instance[f'mask_adjacency_{mode}'] = (
            (full_instance[f'd_{mode}'] < 1e9) & access[:, None] & access[None, :]
        )
    try:
        logs = RollingHorizonDispatcher(
            GreedyInsertion(), delta_minutes=5., shift_minutes=SHIFT_MINUTES,
            service_policy='mandatory',
        ).run_shift(arrivals, fleet, full_instance, sharing_contract=contract)
    except MandatoryServiceError as exc:
        return dict(scope=scope, book_size=len(arrivals), feasible=False,
                    reason=str(exc.reason), pending_ids=list(exc.pending_ids))
    summary = logs[-1]
    return dict(scope=scope, book_size=len(arrivals),
                feasible=bool(summary['service_complete']),
                quality_cost=float(summary['quality_cost']),
                total_delivered=int(summary['total_delivered']),
                total_revealed=int(summary['total_revealed']),
                completion_time_min=float(summary['completion_time_min']))


def _classify(singleton_uav: Dict[str, Any], singleton_adr: Dict[str, Any],
              joint: Dict[str, Any]) -> str:
    if not joint['feasible']:
        return 'infeasible'
    if singleton_uav['feasible'] and singleton_adr['feasible']:
        return 'all_feasible'
    return 'joint_only'


def generate_bank(spec: BankSpec, output_dir: Path) -> Dict[str, Any]:
    """Build + certify + freeze. Returns bank + certification manifest."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    certifications: List[InstanceCertification] = []

    def factory(index: int) -> Dict[str, Any]:
        instance = _build_instance(spec, index)
        cert = InstanceCertification(instance_sha256=instance['instance_sha256'])
        cert.singleton_uav = _trial(spec, instance, 'uav_singleton')
        cert.singleton_adr = _trial(spec, instance, 'adr_singleton')
        cert.joint = _trial(spec, instance, 'joint')
        cert.classification = _classify(cert.singleton_uav, cert.singleton_adr, cert.joint)
        certifications.append(cert)
        # Flatten so instance_digest sees arrays at the top level and the
        # certification fields are JSON-serializable primitives.
        payload = dict(instance)
        payload['certification'] = json.dumps(dict(
            instance_sha256=cert.instance_sha256,
            classification=cert.classification,
            singleton_uav=cert.singleton_uav,
            singleton_adr=cert.singleton_adr,
            joint=cert.joint), sort_keys=True)
        # Arrivals/owners contain only primitives (floats, ints, strings, tuples).
        return payload

    bank_path = output_dir / f'paired_owned_book_bank_n{spec.n_req}_s{spec.seed}_v3.pt'
    bank = freeze_bank(bank_path, spec.to_json(), factory, spec.n_instances)

    counts = {label: 0 for label in ('all_feasible', 'joint_only', 'infeasible')}
    for entry in bank['instances']:
        cert = json.loads(entry['certification'])
        counts[cert['classification']] += 1

    manifest = dict(
        audit='paired_owned_book_bank',
        certification_protocol='preserve-frozen-mode-permissions-1',
        bank_format_version=BANK_FORMAT_VERSION,
        request_pairing_version=REQUEST_PAIRING_VERSION,
        spec=spec.to_json(),
        bank_path=str(bank_path.resolve()),
        bank_id=bank['bank_id'],
        n_instances=spec.n_instances,
        classification_counts=counts,
        eligible_for_finite_cost_game_count=counts['all_feasible'],
        pooled_operation_only_count=counts['joint_only'],
        infeasible_count=counts['infeasible'],
        release_status=release_status(),
        release_contract=release_contract(),
        generated_utc=time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()),
    )

    manifest_path = bank_path.with_name(bank_path.stem + '_manifest.json')
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True))
    return dict(bank=bank, manifest=manifest, manifest_path=manifest_path, bank_path=bank_path)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description='Certify a paired owned-book bank')
    parser.add_argument('--n-req', type=int, required=True)
    parser.add_argument('--n-instances', type=int, required=True)
    parser.add_argument('--seed', type=int, default=20261009)
    parser.add_argument('--beta', type=float, default=0.0)
    parser.add_argument('--wind-mps', type=float, default=0.0)
    parser.add_argument('--n-uav', type=int, default=2)
    parser.add_argument('--n-adr', type=int, default=2)
    parser.add_argument('--eps-shared', type=float, default=0.5)
    parser.add_argument('--output-dir', type=Path, required=True)
    # hardness overrides for benchmark stress banks.
    parser.add_argument('--coord-high', type=float, default=5.0,
                         help='Service-area half-side in km. 5.0 = canonical; '
                              '15-25 = battery-stress bank.')
    parser.add_argument('--tw-slack-mean', type=float, default=30.0)
    parser.add_argument('--tw-slack-std', type=float, default=5.0)
    parser.add_argument('--tw-slack-clip-low', type=float, default=15.0)
    parser.add_argument('--tw-slack-clip-high', type=float, default=60.0)
    parser.add_argument('--demand-low', type=float, default=1.0)
    parser.add_argument('--demand-high', type=float, default=8.0)
    args = parser.parse_args(argv)
    spec = BankSpec(n_req=args.n_req, n_instances=args.n_instances, seed=args.seed,
                    beta=args.beta, wind_mps=args.wind_mps, n_uav=args.n_uav, n_adr=args.n_adr,
                    eps_shared=args.eps_shared,
                    coord_high=args.coord_high,
                    tw_slack_mean=args.tw_slack_mean, tw_slack_std=args.tw_slack_std,
                    tw_slack_clip_low=args.tw_slack_clip_low,
                    tw_slack_clip_high=args.tw_slack_clip_high,
                    demand_low=args.demand_low, demand_high=args.demand_high)
    result = generate_bank(spec, args.output_dir)
    print(json.dumps({'bank_id': result['bank']['bank_id'],
                      'manifest': str(result['manifest_path']),
                      'classification_counts': result['manifest']['classification_counts']},
                     indent=2, sort_keys=True))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
