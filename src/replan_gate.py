from __future__ import annotations

from artifacts import load_tensor_file

from dataclasses import dataclass
from typing import Any, Dict, List, Optional

import numpy as np
import torch
import torch.nn as nn


GATE_FEATURE_NAMES = [
    'time_frac',
    'delta_frac',
    'since_replan_frac',
    'new_since_replan_frac',
    'arrival_rate_frac',
    'active_frac',
    'waiting_frac',
    'onboard_frac',
    'idle_vehicle_frac',
    'committed_vehicle_frac',
    'mean_load_frac',
    'min_battery_frac',
    'min_pickup_slack_frac',
    'min_delivery_slack_frac',
    'overdue_frac',
    'last_solver_time_frac',
]
GATE_FEATURE_DIM = len(GATE_FEATURE_NAMES)

ARRIVAL_HISTORY_NAMES = [
    'new_since_replan_frac',
    'arrival_rate_frac',
    'active_frac',
    'waiting_frac',
    'onboard_frac',
    'idle_vehicle_frac',
    'min_pickup_slack_frac',
    'min_delivery_slack_frac',
]
ARRIVAL_HISTORY_DIM = len(ARRIVAL_HISTORY_NAMES)

BASELINE_CONTEXT_NAMES = [
    'remaining_time_frac',
    'n_req_log_frac',
    'n_vehicle_log_frac',
    'active_count_log_frac',
    'ticks_since_replan_frac',
]
BASELINE_CONTEXT_DIM = len(BASELINE_CONTEXT_NAMES)


@dataclass
class GateDecision:
    replan: bool
    prob_replan: float = 1.0
    log_prob: Optional[torch.Tensor] = None
    reason: str = ''


def _clamp01(value: float) -> float:
    return float(np.clip(value, 0.0, 1.0))


def _slack_feature(values: List[float], current_time: float, shift_minutes: float) -> float:
    if not values:
        return 1.0
    slack = min(values) - current_time
    return float(np.clip(slack / max(shift_minutes, 1e-9), -1.0, 1.0))


def build_replan_features(
    residual: Dict[str, Any],
    *,
    delta_minutes: float,
    shift_minutes: float,
    last_replan_time: Optional[float] = None,
    new_requests_since_replan: int = 0,
    ticks_since_replan: int = 0,
    last_solver_time_s: float = 0.0,
) -> Dict[str, Any]:
    current_time = float(residual.get('current_time', 0.0))
    full_instance = residual.get('full_instance', {}) or {}
    n_req = max(int(full_instance.get('n_req', 1) or 1), 1)
    requests = residual.get('active_requests', {}) or {}
    vehicles = residual.get('vehicles', {}) or {}

    since_replan = 0.0 if last_replan_time is None else max(0.0, current_time - float(last_replan_time))
    waiting = [r for r in requests.values() if getattr(r, 'status', '') == 'waiting_for_pickup']
    onboard = [
        r for r in requests.values()
        if getattr(r, 'status', '') in ('pickup_committed', 'onboard', 'delivery_committed')
    ]
    active = list(requests.values())

    pickup_targets = [float(r.t_pickup) for r in waiting]
    delivery_targets = [
        float(r.t_delivery) for r in active
        if getattr(r, 'status', '') != 'delivered'
    ]
    overdue = [
        r for r in active
        if (getattr(r, 'status', '') == 'waiting_for_pickup' and current_time > float(r.t_pickup))
        or (getattr(r, 'status', '') in ('pickup_committed', 'onboard', 'delivery_committed')
            and current_time > float(r.t_delivery))
    ]

    n_vehicles = max(len(vehicles), 1)
    idle = [v for v in vehicles.values() if getattr(v, 'committed_leg', None) is None]
    committed_frac = 1.0 - len(idle) / n_vehicles

    load_fracs = []
    battery_fracs = []
    for v in vehicles.values():
        cap = max(float(getattr(v, 'capacity', 1.0) or 1.0), 1e-9)
        load_fracs.append(_clamp01(float(getattr(v, 'load', 0.0) or 0.0) / cap))
        b0 = float(getattr(v, 'battery_init', None) or getattr(v, 'battery', 1.0) or 1.0)
        battery_fracs.append(_clamp01(float(getattr(v, 'battery', b0) or 0.0) / max(b0, 1e-9)))

    arrival_rate = float(new_requests_since_replan) / max(since_replan, float(delta_minutes), 1e-9)
    values = {
        'time_frac': _clamp01(current_time / max(shift_minutes, 1e-9)),
        'delta_frac': _clamp01(float(delta_minutes) / max(shift_minutes, 1e-9)),
        'since_replan_frac': _clamp01(since_replan / max(shift_minutes, 1e-9)),
        'new_since_replan_frac': _clamp01(float(new_requests_since_replan) / max(float(n_req), 1.0)),
        'arrival_rate_frac': _clamp01(arrival_rate / max(float(n_req) / max(shift_minutes, 1e-9), 1e-9)),
        'active_frac': _clamp01(float(len(active)) / max(float(n_req), 1.0)),
        'waiting_frac': _clamp01(float(len(waiting)) / max(float(n_req), 1.0)),
        'onboard_frac': _clamp01(float(len(onboard)) / max(float(n_req), 1.0)),
        'idle_vehicle_frac': _clamp01(float(len(idle)) / float(n_vehicles)),
        'committed_vehicle_frac': _clamp01(committed_frac),
        'mean_load_frac': float(np.mean(load_fracs)) if load_fracs else 0.0,
        'min_battery_frac': float(np.min(battery_fracs)) if battery_fracs else 1.0,
        'min_pickup_slack_frac': _slack_feature(pickup_targets, current_time, shift_minutes),
        'min_delivery_slack_frac': _slack_feature(delivery_targets, current_time, shift_minutes),
        'overdue_frac': _clamp01(float(len(overdue)) / max(float(len(active)), 1.0)),
        'last_solver_time_frac': _clamp01(float(last_solver_time_s) / 10.0),
    }
    vector = np.asarray([values[name] for name in GATE_FEATURE_NAMES], dtype=np.float32)
    history = np.asarray([values[name] for name in ARRIVAL_HISTORY_NAMES], dtype=np.float32)
    max_ticks = max(int(np.ceil(shift_minutes / max(float(delta_minutes), 1e-9))), 1)
    baseline_context_values = {
        'remaining_time_frac': _clamp01(1.0 - values['time_frac']),
        'n_req_log_frac': _clamp01(float(np.log1p(n_req) / np.log1p(100.0))),
        'n_vehicle_log_frac': _clamp01(float(np.log1p(len(vehicles)) / np.log1p(50.0))),
        'active_count_log_frac': _clamp01(float(np.log1p(len(active)) / np.log1p(max(n_req, 1)))),
        'ticks_since_replan_frac': _clamp01(float(ticks_since_replan) / float(max_ticks)),
    }
    baseline_context = np.asarray(
        [baseline_context_values[name] for name in BASELINE_CONTEXT_NAMES],
        dtype=np.float32,
    )
    return {
        'names': GATE_FEATURE_NAMES,
        'values': values,
        'vector': vector,
        'arrival_history': history.reshape(1, ARRIVAL_HISTORY_DIM),
        'baseline_context_names': BASELINE_CONTEXT_NAMES,
        'baseline_context': baseline_context,
        'ticks_since_replan': int(ticks_since_replan),
    }


class ReplanGate(nn.Module):
    def __init__(
        self,
        input_dim: int = GATE_FEATURE_DIM,
        hidden_dim: int = 64,
        *,
        history_dim: int = ARRIVAL_HISTORY_DIM,
        history_hidden_dim: Optional[int] = None,
        use_history: bool = True,
    ):
        super().__init__()
        self.use_history = bool(use_history)
        self.history_dim = int(history_dim)
        self.history_hidden_dim = int(history_hidden_dim or hidden_dim)
        if self.use_history:
            self.history_gru = nn.GRU(
                self.history_dim,
                self.history_hidden_dim,
                batch_first=True,
            )
            net_input_dim = input_dim + self.history_hidden_dim
        else:
            self.history_gru = None
            net_input_dim = input_dim
        self.net = nn.Sequential(
            nn.Linear(net_input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 2),
        )

    def forward(self, features: torch.Tensor, history: Optional[torch.Tensor] = None) -> torch.Tensor:
        x = features.float()
        if self.use_history:
            if history is None:
                hist_emb = x.new_zeros((x.shape[0], self.history_hidden_dim))
            else:
                h = history.float()
                if h.dim() == 2:
                    h = h.unsqueeze(0)
                _, h_n = self.history_gru(h)
                hist_emb = h_n[-1]
                if hist_emb.shape[0] != x.shape[0]:
                    hist_emb = hist_emb.expand(x.shape[0], -1)
            x = torch.cat([x, hist_emb], dim=1)
        return self.net(x)


class TorchReplanGateController:
    def __init__(
        self,
        model: ReplanGate,
        *,
        device: Optional[torch.device] = None,
        stochastic: bool = False,
        min_hold_ticks: int = 0,
        max_hold_ticks: Optional[int] = None,
        safety_margin_min: float = 0.0,
        shift_minutes: float = 120.0,
    ):
        self.model = model
        self.device = device or next(model.parameters()).device
        self.stochastic = stochastic
        self.min_hold_ticks = int(min_hold_ticks)
        self.max_hold_ticks = max_hold_ticks
        # Pickup-deadline safety net: when the minimum pickup-slack across
        # all active requests is within safety_margin_min, override the
        # gate and force a replan.  This bounds the delivered-rate loss the
        # learned gate can incur without changing its expected behaviour on
        # the bulk of decisions.  Set to 0.0 to disable.
        self.safety_margin_min = float(safety_margin_min)
        self.shift_minutes = float(shift_minutes)
        self.safety_overrides = 0
        self.log_probs: List[torch.Tensor] = []
        self.entropies: List[torch.Tensor] = []
        self.decision_features: List[torch.Tensor] = []
        self.decision_contexts: List[torch.Tensor] = []
        self._history_rows: List[torch.Tensor] = []

    def reset(self) -> None:
        self.log_probs.clear()
        self.entropies.clear()
        self.decision_features.clear()
        self.decision_contexts.clear()
        self._history_rows.clear()
        self.safety_overrides = 0

    def _safety_override(self, features: Dict[str, Any]) -> bool:
        """Return True if a pickup-deadline is within safety_margin_min."""
        if self.safety_margin_min <= 0.0:
            return False
        values = features.get('values') or {}
        min_pickup_slack_frac = float(values.get('min_pickup_slack_frac', 1.0))
        # The slack feature is in [0, 1] as (slack_minutes / shift_minutes);
        # convert back to minutes.  Negative values can occur if any active
        # pickup is already overdue — definitely a replan trigger.
        slack_min = min_pickup_slack_frac * self.shift_minutes
        return slack_min <= self.safety_margin_min

    def decide(self, features: Dict[str, Any], residual: Optional[Dict[str, Any]] = None) -> GateDecision:
        x = torch.as_tensor(features['vector'], dtype=torch.float32, device=self.device).view(1, -1)
        context = torch.as_tensor(
            features.get('baseline_context', np.zeros(BASELINE_CONTEXT_DIM, dtype=np.float32)),
            dtype=torch.float32,
            device=self.device,
        ).view(-1, BASELINE_CONTEXT_DIM)[-1]
        history_value = features.get('arrival_history')
        if history_value is None:
            hist_row = torch.zeros(ARRIVAL_HISTORY_DIM, dtype=torch.float32, device=self.device)
        else:
            hist_row = torch.as_tensor(
                history_value,
                dtype=torch.float32,
                device=self.device,
            ).view(-1, ARRIVAL_HISTORY_DIM)[-1]
        self._history_rows.append(hist_row)
        history = torch.stack(self._history_rows, dim=0).unsqueeze(0)
        logits = self.model(x, history)
        ticks = int(features.get('ticks_since_replan', 0))
        if ticks < self.min_hold_ticks:
            logits = logits.clone()
            logits[:, 1] = -1e9
        if self.max_hold_ticks is not None and ticks >= int(self.max_hold_ticks):
            logits = logits.clone()
            logits[:, 0] = -1e9

        dist = torch.distributions.Categorical(logits=logits)
        action = dist.sample() if self.stochastic else logits.argmax(dim=1)
        log_prob = dist.log_prob(action)
        entropy = dist.entropy()
        if self.stochastic:
            self.log_probs.append(log_prob.squeeze(0))
            self.entropies.append(entropy.squeeze(0))
            self.decision_features.append(x.squeeze(0).detach())
            self.decision_contexts.append(context.detach())
        prob_replan = torch.softmax(logits, dim=1)[0, 1].detach().item()
        gate_replan = bool(action.item() == 1)
        # Safety net: deadline-imminent override (only active at inference;
        # disabled during training where safety_margin_min defaults to 0).
        forced_by_safety = False
        if not gate_replan and self._safety_override(features):
            gate_replan = True
            forced_by_safety = True
            self.safety_overrides += 1
        return GateDecision(
            replan=gate_replan,
            prob_replan=float(prob_replan),
            log_prob=log_prob.squeeze(0),
            reason='safety_net' if forced_by_safety else 'learned',
        )


class HeuristicReplanGate:
    def __init__(
        self,
        *,
        min_hold_ticks: int = 1,
        max_hold_ticks: int = 6,
        new_request_frac: float = 0.15,
        min_delivery_slack_frac: float = 0.08,
        idle_vehicle_frac: float = 0.25,
    ):
        self.min_hold_ticks = int(min_hold_ticks)
        self.max_hold_ticks = int(max_hold_ticks)
        self.new_request_frac = float(new_request_frac)
        self.min_delivery_slack_frac = float(min_delivery_slack_frac)
        self.idle_vehicle_frac = float(idle_vehicle_frac)

    def reset(self) -> None:
        return None

    def decide(self, features: Dict[str, Any], residual: Optional[Dict[str, Any]] = None) -> GateDecision:
        values = features['values']
        ticks = int(features.get('ticks_since_replan', 0))
        if ticks < self.min_hold_ticks:
            return GateDecision(False, prob_replan=0.0, reason='min_hold')
        if ticks >= self.max_hold_ticks:
            return GateDecision(True, prob_replan=1.0, reason='max_hold')
        if values['new_since_replan_frac'] >= self.new_request_frac and values['idle_vehicle_frac'] >= self.idle_vehicle_frac:
            return GateDecision(True, prob_replan=1.0, reason='new_arrivals_idle')
        if values['min_delivery_slack_frac'] <= self.min_delivery_slack_frac and values['active_frac'] > 0.0:
            return GateDecision(True, prob_replan=1.0, reason='urgent')
        if values['overdue_frac'] > 0.0:
            return GateDecision(True, prob_replan=1.0, reason='overdue')
        return GateDecision(False, prob_replan=0.0, reason='hold')


# ---------------------------------------------------------------------------
# Baseline gates: single-rule comparators for the learned gate under the
# κ-sensitivity sweep. Each gate implements the ReplanGate controller
# interface (reset + decide) so tools/experiments/gate_friction_sweep.py
# can drop them in alongside TorchReplanGateController without changes.
# ---------------------------------------------------------------------------


class AlwaysReplanGate:
    """Baseline: replan on every dispatch tick. Upper bound on compute
    cost; lower bound on service loss. Paper's current κ=0 recommendation at
    n=5 collapses to this gate."""

    def reset(self) -> None:
        return None

    def decide(self, features: Dict[str, Any], residual: Optional[Dict[str, Any]] = None) -> GateDecision:
        return GateDecision(True, prob_replan=1.0, reason='always')


class PeriodicReplanGate:
    """Baseline: replan every ``period_ticks`` dispatch ticks. The
    ``period_ticks=5`` default matches the dispatcher's Δ=5 min cycle so one
    tick == 5 min. Calls removed = (period-1)/period relative to AlwaysReplan.
    """

    def __init__(self, *, period_ticks: int = 5):
        if period_ticks < 1:
            raise ValueError('period_ticks must be >= 1')
        self.period_ticks = int(period_ticks)
        self._tick = 0

    def reset(self) -> None:
        self._tick = 0

    def decide(self, features: Dict[str, Any], residual: Optional[Dict[str, Any]] = None) -> GateDecision:
        # Prefer the dispatcher-reported tick counter when present; falls back
        # to internal counter otherwise so the gate works in test harnesses.
        ticks = int(features.get('ticks_since_replan', self._tick))
        self._tick += 1
        if ticks == 0 or (ticks % self.period_ticks) == 0:
            return GateDecision(True, prob_replan=1.0, reason='periodic')
        return GateDecision(False, prob_replan=0.0, reason='periodic_hold')


class SafetyOnlyGate:
    """Baseline: hold unless pickup slack falls below
    ``safety_margin_min`` (minutes of wall-clock buffer before the
    earliest pickup TW opens/closes). Lower bound on compute cost; upper
    bound on service loss since it never replans preemptively. Pairs
    naturally with the learned gate's inference-time safety net
    (``TorchReplanGateController.safety_margin_min``)."""

    def __init__(self, *, safety_margin_min: float = 2.0, shift_minutes: float = 120.0):
        if safety_margin_min < 0.0:
            raise ValueError('safety_margin_min must be >= 0')
        self.safety_margin_min = float(safety_margin_min)
        self.shift_minutes = float(shift_minutes)

    def reset(self) -> None:
        return None

    def decide(self, features: Dict[str, Any], residual: Optional[Dict[str, Any]] = None) -> GateDecision:
        values = features.get('values') or {}
        # On tick 0 we must replan — no prior plan exists.
        if int(features.get('ticks_since_replan', 0)) == 0:
            return GateDecision(True, prob_replan=1.0, reason='initial')
        min_pickup_slack_frac = float(values.get('min_pickup_slack_frac', 1.0))
        slack_min = min_pickup_slack_frac * self.shift_minutes
        if slack_min <= self.safety_margin_min:
            return GateDecision(True, prob_replan=1.0, reason='safety_slack')
        # Also replan on any overdue request (service is already lost
        # otherwise; this keeps the baseline from degenerating to pure hold).
        if float(values.get('overdue_frac', 0.0)) > 0.0:
            return GateDecision(True, prob_replan=1.0, reason='overdue')
        return GateDecision(False, prob_replan=0.0, reason='safety_hold')


def build_baseline_gate(name: str, **kwargs) -> Any:
    """Named factory for the baseline gates. The ``gate_friction_sweep``
    driver uses this to instantiate from a string config."""
    name = str(name).lower().strip()
    if name in ('always', 'always_replan', 'alwaysreplan'):
        return AlwaysReplanGate()
    if name in ('periodic', 'periodic_k', 'periodic_replan'):
        return PeriodicReplanGate(period_ticks=int(kwargs.get('period_ticks', 5)))
    if name in ('safety', 'safety_only', 'safetyonly'):
        return SafetyOnlyGate(
            safety_margin_min=float(kwargs.get('safety_margin_min', 2.0)),
            shift_minutes=float(kwargs.get('shift_minutes', 120.0)),
        )
    raise ValueError(f'Unknown baseline gate: {name!r}. '
                     f'Expected one of: always, periodic, safety.')


def load_gate_controller(
    path: str,
    *,
    device: Optional[torch.device] = None,
    stochastic: bool = False,
    safety_margin_min: float = 0.0,
    shift_minutes: float = 120.0,
) -> TorchReplanGateController:
    device = device or torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    raw = load_tensor_file(path, map_location=device, weights_only=True)
    state = raw.get('model_state_dict', raw) if isinstance(raw, dict) else raw
    saved_args = raw.get('args', {}) if isinstance(raw, dict) else {}
    hidden_dim = 64
    use_history = True
    history_hidden_dim = None
    first = state.get('net.0.weight') if isinstance(state, dict) else None
    if first is not None:
        hidden_dim = int(first.shape[0])
        if int(first.shape[1]) == GATE_FEATURE_DIM:
            use_history = False
        elif int(first.shape[1]) > GATE_FEATURE_DIM:
            history_hidden_dim = int(first.shape[1]) - GATE_FEATURE_DIM
    gate = ReplanGate(
        hidden_dim=hidden_dim,
        history_hidden_dim=history_hidden_dim,
        use_history=use_history,
    ).to(device)
    gate.load_state_dict(state)
    gate.eval()
    min_hold_ticks = int(saved_args.get('min_hold_ticks', 0) or 0)
    max_hold_raw = saved_args.get('max_hold_ticks', None)
    max_hold_ticks = None if max_hold_raw is None else int(max_hold_raw)
    return TorchReplanGateController(
        gate,
        device=device,
        stochastic=stochastic,
        min_hold_ticks=min_hold_ticks,
        max_hold_ticks=max_hold_ticks,
        safety_margin_min=safety_margin_min,
        shift_minutes=shift_minutes,
    )
