"""Replay matched books from the release protocol."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
from pathlib import Path
from time import perf_counter
from unittest.mock import patch


ROOT = Path(__file__).resolve().parents[1]


def digest(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def check_file(path: Path, expected: str) -> None:
    if digest(path) != expected:
        raise ValueError(f"SHA-256 mismatch: {path}")


def replay(args: argparse.Namespace) -> None:
    os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
    import torch
    from artifacts import load_tensor_file
    from dispatch_sim import MandatoryServiceError, RollingHorizonDispatcher
    from guarded_commitment import GuardedCommitment
    from routing_benchmark import _build_classical_registry, _make_hetgat_coop_runner

    torch.set_num_threads(1)
    protocol = json.loads((ROOT / "data/routing_protocol.json").read_text(encoding="utf-8"))
    settings = protocol["cases"][str(args.requests)]
    bank_path = ROOT / settings["bank"]
    check_file(bank_path, settings["bank_sha256"])
    bank = load_tensor_file(bank_path)
    if bank["bank_id"] != settings["bank_id"]:
        raise ValueError("Unexpected bank identity")
    indices = args.book if args.book else [int(next(iter(settings["book_hashes"])))]
    if len(set(indices)) != len(indices) or any(str(index) not in settings["book_hashes"] for index in indices):
        raise ValueError("Each book must be a distinct index from the release protocol")
    model = ROOT / settings["model"]
    if args.method != "alns":
        check_file(model, settings["model_sha256"])

    uav_pack, adr_pack = protocol["packs_kj"][args.pack]
    fleet_uav, fleet_adr = settings["fleet"]
    options = dict(max_recharges_per_vehicle=settings["recharge_cap"], recharge_cost_includes_dwell=True,
                   allow_delayed_departure=True, capacity_kj_uav=uav_pack, capacity_kj_adr=adr_pack)
    if args.method == "alns":
        runner = _build_classical_registry(fleet_uav, fleet_adr, depot_sharing=False, **options)["alns_strong"]
    else:
        runner = _make_hetgat_coop_runner(model, fleet_uav, fleet_adr, continuation_check="onboard",
                                         reserve_frac_uav=.25, reserve_frac_adr=.20, **options)
    original = RollingHorizonDispatcher.run_shift
    rows = []
    for index in indices:
        book = bank["instances"][index]
        if book["instance_sha256"] != settings["book_hashes"][str(index)]:
            raise ValueError(f"Unexpected book identity for index {index}")
        captured = {}

        def capture(dispatcher, arrivals, fleet, instance, *positional, **named):
            solver = dispatcher.solver
            try:
                if args.method == "guarded":
                    guard = GuardedCommitment(solver, 3, 3)
                    with patch.object(solver, "solve", guard.solve):
                        log = original(dispatcher, arrivals, fleet, instance, *positional, **named)
                else:
                    log = original(dispatcher, arrivals, fleet, instance, *positional, **named)
            except MandatoryServiceError as error:
                captured.update(log=error.episode_log, status=error.reason)
                raise
            captured.update(log=log, status="complete")
            return log

        torch.manual_seed(protocol["seed"])
        started = perf_counter()
        with patch.object(RollingHorizonDispatcher, "run_shift", capture):
            runner(book, time_budget_s=args.alns_seconds or settings["alns_seconds"], seed=protocol["seed"])
        logs = captured["log"]
        legs = [leg for tick in logs for leg in tick.get("completed", ())]
        delivered = len({leg["request_id"] for leg in legs if leg["leg_type"] == "delivery"})
        complete = captured["status"] == "complete"
        if complete != (delivered == args.requests):
            raise RuntimeError(f"Service ledger disagrees with terminal status for book {index}")
        rows.append(dict(requests=args.requests, pack=args.pack, method=args.method, book_index=index,
                         book_sha256=book["instance_sha256"], status=captured["status"], delivered=delivered,
                         complete=complete, cost=logs[-1]["quality_cost"] if complete else "",
                         recharges=sum(leg["leg_type"] == "depot" for leg in legs),
                         energy_kj=sum(leg["energy_used"] for leg in legs), elapsed_s=perf_counter() - started))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(f"Saved {len(rows)} rows to {args.output}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--requests", type=int, choices=(25, 60), required=True)
    parser.add_argument("--pack", choices=("nominal", "cap50"), required=True)
    parser.add_argument("--method", choices=("alns", "greedy", "guarded"), required=True)
    parser.add_argument("--book", type=int, nargs="+", help="Book indices from the release protocol")
    parser.add_argument("--alns-seconds", type=float, choices=(.25, 2., 8.), help="Strong ALNS target per replan")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.alns_seconds is not None and args.method != "alns":
        parser.error("--alns-seconds applies to strong ALNS")
    replay(args)


if __name__ == "__main__":
    main()
