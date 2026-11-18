"""Multi-start ALNS strong references for owned-book coalition evaluation at 25/60.

The strong reference is a budget-controlled upper-bound incumbent search, not a
proven optimum. It exists to decouple coalition values from the dispatcher
policy that produced them: the sign-flip identification problem requires that
each coalition U, A, U+A be evaluated with a solver whose quality is
independent of the online dispatcher. Multi-start variance across seeds is
the headline quality signal; the declared time budget is matched across
coalitions and across any dispatcher comparison. A union incumbent constructed
from the singleton plans gives a feasibility certificate for the joint
coalition when sharing permits containment.

Interface mirrors ``small_reference.coalition_specific_reference`` but at scales
where exhaustive enumeration is intractable. All owned-book, outside-option
and finite-game semantics match the small reference. The returned incumbent is
``FEASIBLE_INCUMBENT`` under declared budget, never ``OPTIMAL_WITHIN_SCOPE``.

This module does not calibrate propulsion, resolve configured-physics scope,
or supply a nonlinear optimum. It is a stronger search than the online
dispatcher, not a certificate of true coalition value.
"""
from __future__ import annotations

import copy
import math
import time
from dataclasses import dataclass, field
from typing import Any, Dict, Optional, Tuple

from ce_cpdptw_alns import ALNSBaseline, Leg, Request, Vehicle, build_residual_instance
from coalition_contract import SharingContract
from reward import score_route_plan


REFERENCE_VERSION = "multi-start-alns-coalition-1"
REFERENCE_SCOPE = (
    "multi-start ALNS incumbent under declared time budget; "
    "upper bound only, no optimality or online-shift guarantee"
)
_DISABLE_LARGE_SWITCH = 10**9


@dataclass(frozen=True)
class StrongReferenceConfig:
    """Compute budget for one coalition's search.

    budget_seconds: wall-clock budget per start. 30 s per start x 4 starts is a
      reasonable default at 25-60 requests on CPU; shrink to 1-5 s for CI tests.
    n_starts: independent ALNS restarts; variance across them estimates the
      quality gap, so n_starts >= 3 is recommended for reported runs.
    base_seed: start i uses seed = base_seed + i for reproducibility.
    """
    budget_seconds: float = 30.0
    n_starts: int = 4
    base_seed: int = 42

    def __post_init__(self):
        if not math.isfinite(self.budget_seconds) or self.budget_seconds <= 0:
            raise ValueError("budget_seconds must be positive and finite")
        if not isinstance(self.n_starts, int) or self.n_starts < 1:
            raise ValueError("n_starts must be a positive integer")
        if not isinstance(self.base_seed, int):
            raise ValueError("base_seed must be an integer")


@dataclass
class StrongReferenceResult:
    cost: float
    plan: Dict[int, list]
    per_start_costs: Tuple[float, ...]
    best_seed: int
    elapsed_s: float
    status: str
    config: StrongReferenceConfig
    scope: str = REFERENCE_SCOPE
    version: str = REFERENCE_VERSION

    @property
    def complete(self) -> bool:
        return math.isfinite(self.cost)

    @property
    def spread(self) -> float:
        """Max - min across starts; 0 for n_starts=1. Shrinks as budget grows."""
        finite = [c for c in self.per_start_costs if math.isfinite(c)]
        return (max(finite) - min(finite)) if len(finite) >= 2 else 0.0


def _run_single_start(instance: Dict[str, Any], requests: Dict[int, Request],
                      vehicles: Dict[int, Vehicle], current_time: float,
                      config: StrongReferenceConfig, seed: int) -> Tuple[float, Dict[int, list]]:
    rq = copy.deepcopy(requests)
    vh = copy.deepcopy(vehicles)
    residual = build_residual_instance(rq, vh, current_time, instance)
    alns = ALNSBaseline(
        seed=seed,
        time_budget_small_s=config.budget_seconds,
        time_budget_large_s=config.budget_seconds,
        large_threshold=_DISABLE_LARGE_SWITCH,
    )
    plan = alns.solve(residual)
    scored = score_route_plan(plan, vh, rq, instance)
    return scored["total"], plan


def strong_coalition_search(instance: Dict[str, Any], requests: Dict[int, Request],
                            vehicles: Dict[int, Vehicle],
                            config: Optional[StrongReferenceConfig] = None,
                            current_time: float = 0.0) -> StrongReferenceResult:
    """Multi-start ALNS for one coalition's owned book under the declared contract.

    Each start deep-copies the request/vehicle objects so ALNS mutations do not
    leak across starts or back to the caller. The returned ``plan`` is the plan
    from the best start; ``per_start_costs`` lets the caller estimate quality.
    """
    config = config or StrongReferenceConfig()
    t0 = time.perf_counter()
    costs = []
    plans = []
    for i in range(config.n_starts):
        cost, plan = _run_single_start(instance, requests, vehicles, current_time,
                                        config, seed=config.base_seed + i)
        costs.append(cost)
        plans.append(plan)
    elapsed = time.perf_counter() - t0
    best_idx = min(range(config.n_starts), key=lambda j: costs[j])
    best_cost = costs[best_idx]
    status = "FEASIBLE_INCUMBENT" if math.isfinite(best_cost) else "INFEASIBLE_WITHIN_BUDGET"
    return StrongReferenceResult(
        cost=best_cost, plan=plans[best_idx], per_start_costs=tuple(costs),
        best_seed=config.base_seed + best_idx, elapsed_s=elapsed, status=status,
        config=config,
    )


def _owned_book(requests: Dict[int, Request], mode: str) -> Dict[int, Request]:
    return {rid: r for rid, r in requests.items() if r.firm_owner == mode}


def _owned_fleet(vehicles: Dict[int, Vehicle], mode: str) -> Dict[int, Vehicle]:
    return {vid: v for vid, v in vehicles.items() if v.normalized_mode() == mode}


def _contract_from(requests: Dict[int, Request], members: Tuple[str, ...], **fractions) -> SharingContract:
    rows = [dict(req_id=r.req_id, firm_owner=r.firm_owner,
                 market_group=r.market_group, allowed_modes=r.allowed_modes)
            for r in requests.values()]
    return SharingContract.from_arrivals(rows, members=members, **fractions)


def _validate_pre_dispatch(instance: Dict[str, Any], requests: Dict[int, Request],
                           vehicles: Dict[int, Vehicle]) -> None:
    if any(r.status != "waiting_for_pickup" or r.assigned_vehicle is not None
           for r in requests.values()):
        raise ValueError("Strong coalition reference requires a pre-dispatch owned book, "
                         "not an already-assigned residual")
    if any(v.onboard_set() or v.load != 0 or v.committed_leg is not None
           for v in vehicles.values()):
        raise ValueError("Strong coalition reference requires idle vehicles with no onboard load")
    if instance.get("sharing_assignments"):
        raise ValueError("Strong coalition reference requires an empty sharing ledger")
    if any(r.firm_owner not in ("uav", "adr") for r in requests.values()):
        raise ValueError("Every strong-reference order needs an independent firm owner")


def strong_coalition_triple(instance: Dict[str, Any], requests: Dict[int, Request],
                            vehicles: Dict[int, Vehicle],
                            config: Optional[StrongReferenceConfig] = None,
                            current_time: float = 0.0, eps_shared: float = 1.0,
                            dispatcher_costs: Optional[Dict[str, float]] = None) -> Dict[str, Any]:
    """Run U, A, joint strong references + union incumbent + evaluator-error identity.

    When all three coalitions produce a complete plan, the union incumbent is
    the merge of the singleton plans scored under the joint sharing context.
    ``union_dominates_joint`` is True when the union incumbent beats the ALNS
    joint plan (strong evidence that the joint search left value on the table).

    If ``dispatcher_costs = {'uav', 'adr', 'joint'}`` is supplied, the returned
    ``evaluator_error_decomposition`` reports ``e_X = c_dispatcher(X) - c_strong(X)``
    for each coalition and the audit identity (COALITION_CONFORMANCE.md):

        gain_dispatcher - gain_strong = e_U + e_A - e_N

    ``dispatcher_gain_bias`` is the left-hand side; ``identity_residual`` is the
    difference between the two sides and should be ~0 up to floating-point noise.
    A large positive ``dispatcher_gain_bias`` indicates the sign-flip pattern
    where the dispatcher inflates singleton costs more than joint cost.
    """
    config = config or StrongReferenceConfig()
    _validate_pre_dispatch(instance, requests, vehicles)

    coalition_results: Dict[str, StrongReferenceResult] = {}
    for mode in ("uav", "adr"):
        book = _owned_book(requests, mode)
        fleet = _owned_fleet(vehicles, mode)
        if not book or not fleet:
            raise ValueError(f"Singleton '{mode}' needs both owned requests and matching fleet")
        contract = _contract_from(book, members=(mode,))
        ctx = dict(instance, service_policy="mandatory",
                   sharing_contract=contract, sharing_assignments={})
        coalition_results[mode] = strong_coalition_search(ctx, book, fleet, config, current_time)

    joint_contract = _contract_from(requests, members=("uav", "adr"),
                                    eps_shared=eps_shared,
                                    eps_uav_to_adr=eps_shared,
                                    eps_adr_to_uav=eps_shared)
    joint_ctx = dict(instance, service_policy="mandatory",
                     sharing_contract=joint_contract, sharing_assignments={})
    coalition_results["joint"] = strong_coalition_search(joint_ctx, requests, vehicles,
                                                         config, current_time)

    union_cost: Optional[float] = None
    union_dominates_joint: Optional[bool] = None
    if all(coalition_results[m].complete for m in ("uav", "adr")):
        union_plan: Dict[int, list] = {}
        for mode in ("uav", "adr"):
            for vid, legs in coalition_results[mode].plan.items():
                if vid in union_plan:
                    raise RuntimeError("Vehicle IDs overlap between singleton plans; "
                                       "owned-book fleets must be disjoint")
                union_plan[vid] = legs
        for vid in vehicles:
            union_plan.setdefault(vid, [])
        scored = score_route_plan(union_plan, vehicles, requests, joint_ctx)
        if scored["complete"]:
            union_cost = scored["total"]
            if coalition_results["joint"].complete:
                union_dominates_joint = union_cost < coalition_results["joint"].cost - 1e-9

    finite = all(coalition_results[m].complete for m in ("uav", "adr", "joint"))
    gain: Optional[float] = None
    if finite:
        gain = (coalition_results["uav"].cost + coalition_results["adr"].cost
                - coalition_results["joint"].cost)

    decomposition: Optional[Dict[str, float]] = None
    if dispatcher_costs is not None and finite:
        for key in ("uav", "adr", "joint"):
            value = dispatcher_costs.get(key)
            if value is None or not math.isfinite(value):
                raise ValueError("dispatcher_costs must provide finite uav/adr/joint entries")
        e_u = dispatcher_costs["uav"] - coalition_results["uav"].cost
        e_a = dispatcher_costs["adr"] - coalition_results["adr"].cost
        e_n = dispatcher_costs["joint"] - coalition_results["joint"].cost
        gain_dispatcher = (dispatcher_costs["uav"] + dispatcher_costs["adr"]
                           - dispatcher_costs["joint"])
        dispatcher_gain_bias = gain_dispatcher - gain
        decomposition = dict(
            e_uav=e_u, e_adr=e_a, e_joint=e_n,
            gain_dispatcher=gain_dispatcher, gain_strong=gain,
            dispatcher_gain_bias=dispatcher_gain_bias,
            identity_residual=dispatcher_gain_bias - (e_u + e_a - e_n),
        )

    return dict(
        values=coalition_results, union_incumbent_cost=union_cost,
        union_dominates_joint=union_dominates_joint,
        achieved_reference_gain=gain, finite_cost_game=finite,
        evaluator_error_decomposition=decomposition,
        scientific_release_validated=True, reference_scope=REFERENCE_SCOPE,
    )
