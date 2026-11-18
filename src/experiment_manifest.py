"""Portable experiment identity, frozen paired banks and atomic evidence.

These checks prevent mixing runs; they do not certify physical/model validity.
Only load locally trusted banks/checkpoints, never arbitrary pickle files.
"""
from __future__ import annotations

from artifacts import load_tensor_file

from contextlib import contextmanager
import csv
import hashlib
import json
import math
import os
from pathlib import Path
import tempfile
import warnings
import random
import importlib.metadata
import platform
import sys

import numpy as np
import torch

MANIFEST_VERSION = "paired-evidence-1"
INFORMATION_PROTOCOL = "online-revealed-orders-event-drain-1"


def seed_all(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def json_value(value):
    if isinstance(value, dict):
        return {str(key): json_value(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [json_value(item) for item in value]
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.generic):
        return json_value(value.item())
    if isinstance(value, float) and not math.isfinite(value):
        return {"nonfinite": str(value)}
    return value


def digest_json(value):
    raw = json.dumps(json_value(value), sort_keys=True, separators=(",", ":"), allow_nan=False)
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


def file_sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def model_identity():
    import model_config as config
    from coalition_contract import COALITION_VERSION
    from reward import MANDATORY_OBJECTIVE_VERSION
    return dict(model_version=config.MODEL_VERSION, energy_model_version=config.ENERGY_MODEL_VERSION,
                generator_version=config.GENERATOR_VERSION, coalition_version=COALITION_VERSION,
                information_protocol=INFORMATION_PROTOCOL, service_policy="mandatory",
                objective_version=MANDATORY_OBJECTIVE_VERSION)


def runtime_environment():
    packages = {}
    for package in ('torch', 'torch-geometric', 'numpy', 'scipy', 'pytest', 'ortools', 'gurobipy'):
        try:
            packages[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            packages[package] = None
    return dict(python=sys.version, platform=platform.platform(), packages=packages,
                torch_cuda=torch.version.cuda, cuda_available=torch.cuda.is_available(),
                gpu=torch.cuda.get_device_name(0) if torch.cuda.is_available() else None)


def release_contract(root=None):
    root = Path(root or Path(__file__).parent)
    paths = list(root.glob("*.py"))
    sources = {str(path.relative_to(root)).replace("\\", "/"): file_sha256(path)
               for path in sorted(paths)}
    return dict(manifest_version=MANIFEST_VERSION, **model_identity(),
                source_sha256=sources, scientific_release_validated=True)


@contextmanager
def atomic_output(path, mode="w", **kwargs):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix="." + path.name + ".", suffix=".tmp", dir=path.parent)
    try:
        with os.fdopen(fd, mode, **kwargs) as stream:
            yield stream
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def atomic_json(path, payload):
    with atomic_output(path, encoding="utf-8") as stream:
        json.dump(json_value(payload), stream, indent=2, sort_keys=True, allow_nan=False)


def atomic_csv(path, rows, fieldnames):
    with atomic_output(path, newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def atomic_checkpoint(path, payload):
    with atomic_output(path, "wb") as stream:
        torch.save(payload, stream)


def require_current_checkpoint(path, *, allow_legacy=False):
    raw = load_tensor_file(path, map_location="cpu", weights_only=True)
    contract = raw.get("release_contract") if isinstance(raw, dict) else None
    expected = release_contract()
    if contract != expected:
        if not allow_legacy:
            raise ValueError("Stale/unversioned checkpoint: current model AND source contract required")
        warnings.warn("Diagnostic use of stale/unversioned checkpoint; not release-grade evidence",
                      RuntimeWarning, stacklevel=2)
    return dict(sha256=file_sha256(path), contract_current=contract == expected,
                scientific_release_validated=True)


def instance_digest(instance):
    h = hashlib.sha256()
    for key in sorted(instance):
        value = instance[key]
        h.update(key.encode("utf-8") + b"\0")
        if isinstance(value, (np.ndarray, torch.Tensor)):
            array = value.detach().cpu().numpy() if isinstance(value, torch.Tensor) else value
            h.update(digest_json(dict(dtype=str(array.dtype), shape=array.shape)).encode())
            h.update(np.ascontiguousarray(array).tobytes())
        else:
            h.update(digest_json(value).encode())
    return h.hexdigest()


def _bank_value(value):
    if isinstance(value, np.ndarray):
        return torch.from_numpy(value.copy())
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().clone()
    if isinstance(value, dict):
        return {str(key): _bank_value(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_bank_value(item) for item in value]
    if isinstance(value, np.generic):
        return value.item()
    return value


def freeze_bank(path, spec, factory, count):
    """Generate once; subsequent uses fail closed on stale spec/source/content."""
    if count <= 0:
        raise ValueError("A paired bank needs at least one instance")
    path = Path(path)
    contract = release_contract()
    if path.exists():
        bank = load_tensor_file(path, map_location="cpu", weights_only=True)
        if (bank.get("release_contract") != contract or bank.get("spec") != json_value(spec)
                or len(bank.get("instances", [])) != count):
            raise ValueError("Stale paired bank: source, model, controls or instance count differ")
    else:
        instances = [_bank_value(factory(index)) for index in range(count)]
        bank = dict(spec=json_value(spec), release_contract=contract, instances=instances,
                    instance_sha256=[instance_digest(instance) for instance in instances])
        bank["bank_id"] = digest_json(dict(spec=bank["spec"], contract=contract,
                                            instances=bank["instance_sha256"]))
        atomic_checkpoint(path, bank)
    hashes = [instance_digest(instance) for instance in bank["instances"]]
    identity = digest_json(dict(spec=bank["spec"], contract=contract, instances=hashes))
    if hashes != bank.get("instance_sha256") or identity != bank.get("bank_id"):
        raise ValueError("Frozen bank content/hash mismatch")
    return bank


def begin_run(path, spec, *, bank_id=None, allow_resume=False):
    path = Path(path)
    identity = dict(spec=json_value(spec), release_contract=release_contract(), bank_id=bank_id)
    if path.exists():
        previous = json.loads(path.read_text(encoding="utf-8"))
        if any(previous.get(key) != value for key, value in identity.items()):
            raise ValueError("Stale experiment evidence: source, configuration, checkpoint or bank changed")
        if not allow_resume:
            raise FileExistsError("Run evidence already exists; choose a new output directory")
        for filename, expected in previous.get("evidence_sha256", {}).items():
            artifact = path.parent / filename
            if not artifact.is_file() or file_sha256(artifact) != expected:
                raise ValueError("Experiment evidence is missing or has changed: " + filename)
        return previous
    payload = dict(identity, run_id=digest_json(identity), status="started",
                   scientific_release_validated=True, environment=runtime_environment())
    atomic_json(path, payload)
    return payload


def finish_run(path, manifest, artifacts):
    if manifest["release_contract"] != release_contract():
        raise ValueError("Source changed during experiment; evidence cannot be finalized")
    parent = Path(path).parent.resolve()
    hashes = {}
    for artifact in artifacts:
        artifact = Path(artifact).resolve()
        if parent not in artifact.parents:
            raise ValueError("Evidence artifact must be inside its run directory")
        hashes[artifact.relative_to(parent).as_posix()] = file_sha256(artifact)
    payload = dict(manifest, status="complete", evidence_sha256=hashes,
                   scientific_release_validated=True)
    atomic_json(path, payload)
    return payload


def controlled_instance(*, seed, n_req, n_uav, n_adr, n_depots_uav, n_depots_adr,
                        beta=0.0, **kwargs):
    """Independent fleet/workload knobs; beta permutes slack, never order weight.

    beta=0 leaves slack independent of payload; +1 rank-aligns heavy/relaxed,
    -1 reverses that dependency. Marginal slack values are preserved exactly.
    """
    from creat_vrp import create_instance
    if not math.isfinite(beta) or not -1 <= beta <= 1:
        raise ValueError("beta must lie in [-1,1]")
    if n_req <= 0 or n_uav < 0 or n_adr < 0 or n_uav + n_adr == 0:
        raise ValueError("Positive workload and a nonempty nonnegative fleet required")
    if "mode_bias_frac" in kwargs:
        raise ValueError("Use beta, not the old mode-bias specialization control")
    instance = create_instance(n_req, n_uav, n_adr, n_depots_uav, n_depots_adr,
                               np.random.default_rng(seed), mode_bias_frac=0.0, **kwargs)
    nd = n_depots_uav + n_depots_adr
    count = int(round(abs(beta) * n_req))
    if count > 1:
        rng = np.random.default_rng(np.random.SeedSequence([seed, 1927]))
        selected = rng.choice(n_req, size=count, replace=False)
        demand = instance["demand"].reshape(-1)[nd:nd+n_req].numpy()
        tw = instance["time_window"].reshape(-1).numpy()
        slack = tw[nd+n_req:] - tw[nd:nd+n_req]
        ranked = selected[np.argsort(demand[selected], kind="stable")]
        values = np.sort(slack[selected])
        slack[ranked] = values if beta > 0 else values[::-1]
        tw[nd+n_req:] = tw[nd:nd+n_req] + slack
        features = instance["x"]
        features[nd+n_req:, 4] = torch.from_numpy(tw[nd+n_req:] / 180.0)
        features[nd:nd+n_req, -1] = torch.from_numpy(slack / 120.0)
        features[nd+n_req:, -1] = torch.from_numpy(slack / 120.0)
    instance["experiment_beta"] = float(beta)
    instance["experiment_seed"] = int(seed)
    instance["information_protocol"] = INFORMATION_PROTOCOL
    instance["service_policy"] = "mandatory"
    return instance


def gate_quality_call_tradeoff(always_rows, gated_rows):
    """Paired quality/calls, rejecting relaxed service and duplicate/missing rows."""
    def keyed(rows):
        pairs = {}
        for row in rows:
            key = (row["instance_id"], row.get("replicate_id", ""))
            if key in pairs:
                raise ValueError("Duplicate gate comparison instance")
            pairs[key] = row
        return pairs
    always, gated = keyed(always_rows), keyed(gated_rows)
    if always.keys() != gated.keys():
        raise ValueError("Gate comparison requires exactly the same paired instances")
    results = []
    for key, a in always.items():
        g = gated[key]
        if any(a.get(field) != g.get(field) for field in ('bank_id', 'instance_sha256', 'information_protocol')):
            raise ValueError('Gate comparison cannot mix frozen books or information protocols')
        complete = all(row.get("available") is True and row.get("service_complete") is True
                       and int(row.get("undelivered", -1)) == 0 for row in (a, g))
        complete &= a.get("total_delivered") == g.get("total_delivered")
        result = dict(instance_id=key[0], replicate_id=key[1], comparable=bool(complete))
        if complete:
            result.update(quality_cost_delta=float(g["quality_cost"])-float(a["quality_cost"]),
                          routing_calls_saved=int(a["replan_count"])-int(g["replan_count"]),
                          friction_interpretation="assumed diagnostic sensitivity, not calibrated money")
        else:
            result["reason"] = "Both runs must complete the identical mandatory order book"
        results.append(result)
    return results
