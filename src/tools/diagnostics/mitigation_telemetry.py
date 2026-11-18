"""Per-epoch telemetry for the route-policy mitigations.

Each helper is a pure function that returns a scalar the trainer can write
to its CSV. The point is to let us ANSWER whether a mitigation is actually
contributing, not just whether it correlates with higher val_complete_rate.

Signals currently collected:

  * ``critic_pearson_corr(V_preds, returns)`` -- Pearson correlation between
    the critic's V(s) prediction and the actual episode return. If this is
    near 0 the critic is behaving like a frozen constant and the advantage
    A = R - V reduces to A = R - const, i.e. REINFORCE. If the mitigation
    is working this should climb above ~0.3 within a few epochs.

  * ``dyn_proj_grad_l2(model, policy)`` -- L2 norm of the gradient flowing
    into the per-vehicle dyn projection layers (``dyn_proj_uav`` +
    ``dyn_proj_adr``). If the policy is actually consuming the extended
    10-dim features this stays non-trivial; if the extra features are
    "dead weight" this collapses toward the 3-dim-equivalent magnitude.

  * ``depot_visit_fraction(actions, n_depots)`` -- fraction of decoder
    actions that are depot nodes (indices in [0, n_depots)). The battery-
    slack-k mitigation claims to nudge the policy toward more mid-shift
    recharge visits; this metric lets us check elasticity vs k.

  * ``pbrs_telescope_equal(step_shaping, episodic_shaping)`` -- invariance
    check (Ng/Harada/Russell 1999) that under gamma=1 whole-episode
    REINFORCE the per-step PBRS sum equals the episodic shaping. If this
    fails the --delivery-shaping-mode plumbing is broken.

All helpers accept and return plain floats / ints / tensors, so the
trainer can call them without pulling in extra dependencies. See the
instrumentation hook in VRP_Rollout_train.py for wiring.
"""
from __future__ import annotations

from typing import Iterable, Optional

import math

import torch


def critic_pearson_corr(V_preds: Iterable[float],
                         returns: Iterable[float]) -> float:
    """Pearson correlation between predicted V(s) and realised return.

    Returns 0.0 if either sequence has fewer than 2 samples or zero
    variance (so a frozen-constant critic surfaces as corr=0, not NaN).
    """
    v = [float(x) for x in V_preds]
    r = [float(x) for x in returns]
    n = min(len(v), len(r))
    if n < 2:
        return 0.0
    v = v[:n]; r = r[:n]
    mv = sum(v) / n
    mr = sum(r) / n
    sv = sum((x - mv) ** 2 for x in v)
    sr = sum((x - mr) ** 2 for x in r)
    if sv <= 0.0 or sr <= 0.0:
        return 0.0
    cov = sum((v[i] - mv) * (r[i] - mr) for i in range(n))
    denom = math.sqrt(sv * sr)
    if denom <= 0.0:
        return 0.0
    return cov / denom


def dyn_proj_grad_l2(model: torch.nn.Module,
                      policy: str = 'route') -> float:
    """L2 norm across the per-vehicle dyn projection layer gradients.

    Walks the model (handles DDP / DataParallel wrapping) and sums the
    gradient L2 of ``dyn_proj_uav.weight`` and ``dyn_proj_adr.weight``
    if present. Returns 0.0 if the layers are not yet wired (coop policy
    or basic dyn without critic).
    """
    inner = _unwrap(model)
    total_sq = 0.0
    found = False
    for name in ('dyn_proj_uav', 'dyn_proj_adr'):
        mod = _find_submodule(inner, name)
        if mod is None:
            continue
        w = getattr(mod, 'weight', None)
        if w is None or w.grad is None:
            continue
        total_sq += float(w.grad.detach().pow(2).sum().item())
        found = True
    if not found:
        return 0.0
    return math.sqrt(total_sq)


def depot_visit_fraction(actions: torch.Tensor, n_depots: int) -> float:
    """Fraction of actions that are depot node indices.

    ``actions`` is the usual ``tour_idx[:, :, 1:]`` tensor (B, A, T) of
    node indices chosen by the decoder. ``n_depots`` is the number of
    depot slots at the front of the flat node table; everything < that
    is a depot, everything >= that is a customer.
    """
    if actions.numel() == 0 or n_depots <= 0:
        return 0.0
    is_depot = (actions >= 0) & (actions < n_depots)
    return float(is_depot.float().mean().item())


def pbrs_telescope_equal(step_shaping: Iterable[float],
                          episodic_shaping: float,
                          tol: float = 1e-5) -> bool:
    """Check that per-step PBRS telescopes to the episodic form.

    Under gamma=1 whole-episode REINFORCE, per-step PBRS of the form
    (gamma * Phi(s') - Phi(s)) summed over an episode starting at
    Phi(s_0) = 0 and ending at a terminal state equals Phi(s_T)
    (Ng/Harada/Russell 1999). The trainer's episodic shaping is
    k * delivered_count = -k * (Phi(s_T) - Phi(s_0)) by convention,
    so sum(step) and episodic shaping must agree up to float tol.
    """
    s = sum(float(x) for x in step_shaping)
    return abs(s - float(episodic_shaping)) <= tol


# ---------------------------------------------------------------------------
# Helpers for model-unwrapping (DDP / DataParallel).

def _unwrap(model: torch.nn.Module) -> torch.nn.Module:
    if hasattr(model, 'module'):
        return model.module  # type: ignore[attr-defined]
    return model


def _find_submodule(root: torch.nn.Module, name: str) -> Optional[torch.nn.Module]:
    # The route-policy actor nests the decoder's dyn projection under
    # the CoopDecoder; we search depth-first so this works regardless of
    # whether the user wraps the actor in Model, DDP, or DataParallel.
    for mod_name, mod in root.named_modules():
        if mod_name.endswith(name):
            return mod
    return None
