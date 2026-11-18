"""Joint-only coalition settlement interface.

Provides an explicit algebraic statement of which customer obligations belong
to each firm and how their union forms the grand-coalition task.

The existing modules provide the building blocks:

* ``coalition_contract.SharingContract`` — owned books (``owners[rid] -> firm``)
  + market groups (shared / uav / adr) + per-request mode permissions +
  epsilon sharing budgets.
* ``allocation.shapley / epm / pam`` — closed-form two-player cost allocation
  rules with core / efficient / participation checks.
* ``strong_coalition_reference.strong_coalition_triple`` — multi-start ALNS
  reference values for v(U), v(A), v(N).

This module composes those building blocks into one explicit interface that
takes raw owned books, forms the grand-coalition task algebraically, runs a
chosen dispatcher on each of (U, A, N), and returns a settlement with the
allocation + feasibility properties the paper reports.

Design decisions
----------------

* The grand coalition's task is the DISJOINT UNION of the two owned books:
  ``grand_book = {uav_book[rid] | rid in uav_book} ⊔ {adr_book[rid] | rid in adr_book}``
  Request IDs across the two books MUST be disjoint; we validate this.
* v(S) is the five-cost mandatory objective achieved by a dispatcher serving
  the owned book of coalition S with the vehicles of coalition S, under a
  SharingContract whose ``members`` is exactly S. For singletons, the
  cross-mode epsilon budgets collapse to zero because the other mode is
  not a member — the ``allows`` check in SharingContract already handles
  this via the ``mode not in self.members`` guard.
* The dispatcher is injected (``dispatcher_fn``) so this interface works with
  the trained HetGAT coop policy, with classical ALNS, with offline strong
  references, or with any other method the paper reports. The dispatcher is
  called three times (one per coalition) under identical shared-physics
  conditions (``_simulate_leg``, mandatory contract, same banks).
* Allocation rule is a kwarg: Shapley by default, EPM / PAM available.

Not a replacement for ``strong_coalition_reference`` (which computes v(S) via
multi-start search as a reference bound). ``strong_coalition_reference`` is
one choice of ``dispatcher_fn``; this module is the orchestration + settlement
layer sitting above whichever dispatcher the paper wants to report.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Dict, Iterable, Mapping, Optional, Tuple

import math

from allocation import (char_fn, core_exists, efficient, epm, pam,
                         participation_flags, shapley)
from coalition_contract import MODES, SharingContract
from dispatch_sim import Request, Vehicle


SETTLEMENT_VERSION = 'joint-only-settlement-1'
ALLOCATION_RULES = ('shapley', 'epm', 'pam')


DispatcherFn = Callable[
    [Mapping[str, Any], Mapping[int, Request], Mapping[int, Vehicle]], float
]


@dataclass(frozen=True)
class CoalitionBook:
    """One firm's owned-book side of the game.

    ``requests`` maps req_id -> Request with ``firm_owner`` already set to the
    firm's mode. ``vehicles`` are the firm's own fleet. ``mode`` is the firm
    identifier ('uav' or 'adr'); the caller is responsible for consistency.
    """
    mode: str
    requests: Mapping[int, Request]
    vehicles: Mapping[int, Vehicle]

    def __post_init__(self):
        if self.mode not in MODES:
            raise ValueError(f'invalid firm mode: {self.mode!r}')
        for rid, r in self.requests.items():
            if int(rid) != int(r.req_id):
                raise ValueError('CoalitionBook request-id map inconsistent')
            if getattr(r, 'firm_owner', None) != self.mode:
                raise ValueError(
                    f'request {rid} firm_owner={r.firm_owner!r} does not match '
                    f'book mode {self.mode!r}')
        for vid, v in self.vehicles.items():
            if int(vid) != int(v.vehicle_id):
                raise ValueError('CoalitionBook vehicle-id map inconsistent')


@dataclass(frozen=True)
class Settlement:
    """One joint-settlement outcome: three valuations + allocation + flags."""
    settlement_version: str
    allocation_rule: str
    v_uav: float
    v_adr: float
    v_joint: float
    coalition_gain: float             # v(U) + v(A) - v(N); positive if coop helps
    allocation_uav: float             # firm's share under the chosen rule
    allocation_adr: float
    core_exists: bool
    efficient: bool
    participation_uav: bool
    participation_adr: bool
    extra: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return dict(
            settlement_version=self.settlement_version,
            allocation_rule=self.allocation_rule,
            v_uav=float(self.v_uav),
            v_adr=float(self.v_adr),
            v_joint=float(self.v_joint),
            coalition_gain=float(self.coalition_gain),
            allocation_uav=float(self.allocation_uav),
            allocation_adr=float(self.allocation_adr),
            core_exists=bool(self.core_exists),
            efficient=bool(self.efficient),
            participation_uav=bool(self.participation_uav),
            participation_adr=bool(self.participation_adr),
            extra=dict(self.extra),
        )


def grand_coalition_book(uav_book: CoalitionBook, adr_book: CoalitionBook
                           ) -> Tuple[Dict[int, Request], Dict[int, Vehicle]]:
    """Form the grand-coalition task as the disjoint union of owned books.

    Raises ValueError on overlapping request IDs or vehicle IDs. The returned
    maps are new dicts; the input books are not mutated. This is the explicit
    algebraic step forming the grand coalition from owned books.
    """
    if not isinstance(uav_book, CoalitionBook) or uav_book.mode != 'uav':
        raise ValueError('uav_book must be a CoalitionBook with mode=uav')
    if not isinstance(adr_book, CoalitionBook) or adr_book.mode != 'adr':
        raise ValueError('adr_book must be a CoalitionBook with mode=adr')
    overlap_r = set(uav_book.requests) & set(adr_book.requests)
    if overlap_r:
        raise ValueError(
            f'Owned books must be request-id disjoint; overlapping ids: {overlap_r}')
    overlap_v = set(uav_book.vehicles) & set(adr_book.vehicles)
    if overlap_v:
        raise ValueError(
            f'Fleets must be vehicle-id disjoint; overlapping ids: {overlap_v}')
    requests: Dict[int, Request] = {}
    requests.update(uav_book.requests)
    requests.update(adr_book.requests)
    vehicles: Dict[int, Vehicle] = {}
    vehicles.update(uav_book.vehicles)
    vehicles.update(adr_book.vehicles)
    return requests, vehicles


def _contract_for(members: Tuple[str, ...],
                   contract_requests: Mapping[int, Request],
                   eps_shared: float, eps_uav_to_adr: float, eps_adr_to_uav: float
                   ) -> SharingContract:
    """SharingContract for a specific coalition.

    ``contract_requests`` is the FULL union book (U ∪ A) regardless of which
    coalition is being valued. This matches how downstream RL and ALNS code
    check the contract: ``main.RLSolver.solve`` builds a tensor sharing
    context over every request id in the instance (``n_req`` ids), so the
    contract must cover all of them.

    What varies by coalition is ``members`` — the set of modes allowed to
    serve ANY request. For a singleton coalition members=('uav',), the
    ``allows()`` check in SharingContract refuses to assign any request to
    the 'adr' mode (and vice versa), so the semantic "firm serves only its
    own book under its own fleet" is enforced via (a) the dispatcher's
    arrival stream revealing only the firm's requests + (b) members
    disallowing cross-mode service.
    """
    rows = [dict(req_id=r.req_id, firm_owner=r.firm_owner,
                 market_group=getattr(r, 'market_group', 'shared'),
                 allowed_modes=tuple(getattr(r, 'allowed_modes', MODES)))
            for r in contract_requests.values()]
    return SharingContract.from_arrivals(
        rows, members=members,
        eps_shared=eps_shared,
        eps_uav_to_adr=eps_uav_to_adr,
        eps_adr_to_uav=eps_adr_to_uav)


def _valuation(instance_fn: Callable[[], Dict[str, Any]],
                members: Tuple[str, ...],
                book_requests: Mapping[int, Request],
                book_vehicles: Mapping[int, Vehicle],
                full_requests: Mapping[int, Request],
                dispatcher_fn: DispatcherFn,
                eps_shared: float, eps_uav_to_adr: float, eps_adr_to_uav: float
                ) -> float:
    """Compute v(S) by running ``dispatcher_fn`` on the owned book under a
    SharingContract whose members is S.

    ``book_requests`` is this coalition's subset (passed to the dispatcher to
    drive the arrival stream). ``full_requests`` is the grand-coalition book
    (used only to build the contract so it covers every request id the
    downstream tensor path expects — see ``_contract_for`` docstring).
    ``instance_fn`` is a factory so each coalition call starts from a fresh
    instance dict (no cross-coalition state bleed through
    objective_weights / sharing_assignments).
    """
    inst = dict(instance_fn())
    inst['sharing_contract'] = _contract_for(
        members, full_requests, eps_shared, eps_uav_to_adr, eps_adr_to_uav)
    inst['sharing_assignments'] = {}
    return float(dispatcher_fn(inst, book_requests, book_vehicles))


def joint_settlement(instance_fn: Callable[[], Dict[str, Any]],
                      uav_book: CoalitionBook,
                      adr_book: CoalitionBook,
                      dispatcher_fn: DispatcherFn,
                      *,
                      allocation_rule: str = 'shapley',
                      eps_shared: float = 1.0,
                      eps_uav_to_adr: float = 0.0,
                      eps_adr_to_uav: float = 0.0) -> Settlement:
    """Run joint-only settlement on one (U, A) instance under one dispatcher.

    Steps:
      1. v(U) = dispatcher serving UAV's owned book with UAV fleet only;
         contract members = {'uav'}, no cross-mode budget.
      2. v(A) = mirror for ADR.
      3. v(N) = dispatcher serving the union book with the union fleet;
         contract members = {'uav', 'adr'}, epsilon sharing active.
      4. Allocation: apply ``allocation_rule`` to (v(U), v(A), v(N)).
      5. Core / efficient / participation flags.

    Returns a ``Settlement`` with all fields the paper reports.
    """
    if allocation_rule not in ALLOCATION_RULES:
        raise ValueError(
            f'allocation_rule must be one of {ALLOCATION_RULES}, got {allocation_rule!r}')

    grand_requests, grand_vehicles = grand_coalition_book(uav_book, adr_book)

    v_uav = _valuation(instance_fn, ('uav',), uav_book.requests,
                        uav_book.vehicles, grand_requests, dispatcher_fn,
                        eps_shared=0.0, eps_uav_to_adr=0.0, eps_adr_to_uav=0.0)
    v_adr = _valuation(instance_fn, ('adr',), adr_book.requests,
                        adr_book.vehicles, grand_requests, dispatcher_fn,
                        eps_shared=0.0, eps_uav_to_adr=0.0, eps_adr_to_uav=0.0)
    v_joint = _valuation(instance_fn, MODES, grand_requests, grand_vehicles,
                          grand_requests, dispatcher_fn,
                          eps_shared=eps_shared,
                          eps_uav_to_adr=eps_uav_to_adr,
                          eps_adr_to_uav=eps_adr_to_uav)

    coalition_gain = v_uav + v_adr - v_joint
    allocators = dict(shapley=shapley, epm=epm, pam=pam)
    alloc_u, alloc_a = allocators[allocation_rule](v_uav, v_adr, v_joint)

    return Settlement(
        settlement_version=SETTLEMENT_VERSION,
        allocation_rule=allocation_rule,
        v_uav=v_uav, v_adr=v_adr, v_joint=v_joint,
        coalition_gain=coalition_gain,
        allocation_uav=alloc_u,
        allocation_adr=alloc_a,
        core_exists=core_exists(alloc_u, alloc_a, v_uav, v_adr),
        efficient=efficient(alloc_u, alloc_a, v_joint),
        participation_uav=bool(participation_flags(
            alloc_u, alloc_a, v_uav, v_adr).get('uav', False)),
        participation_adr=bool(participation_flags(
            alloc_u, alloc_a, v_uav, v_adr).get('adr', False)),
        extra=dict(
            characteristic_function=char_fn(v_uav, v_adr, v_joint),
            n_uav_requests=len(uav_book.requests),
            n_adr_requests=len(adr_book.requests),
            n_uav_vehicles=len(uav_book.vehicles),
            n_adr_vehicles=len(adr_book.vehicles),
            eps_shared=eps_shared,
            eps_uav_to_adr=eps_uav_to_adr,
            eps_adr_to_uav=eps_adr_to_uav,
        ),
    )
