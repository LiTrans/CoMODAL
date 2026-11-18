from __future__ import annotations

import argparse
import csv
import os
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from coalition import make_fleet
from creat_vrp import create_instance
from dispatch_sim import RollingHorizonDispatcher
from main import build_solver
from experiment_manifest import (atomic_checkpoint, atomic_csv, begin_run, controlled_instance,
    finish_run, freeze_bank, release_contract, require_current_checkpoint, seed_all)
from replan_gate import (
    BASELINE_CONTEXT_DIM,
    GATE_FEATURE_DIM,
    ReplanGate,
    TorchReplanGateController,
)


def arrivals_from_instance(full_instance: Dict[str, Any]) -> List[Dict[str, Any]]:
    n_depots = int(full_instance['n_depots'])
    n_req = int(full_instance['n_req'])
    tw = full_instance['time_window']
    demand = full_instance['demand']
    tw_arr = tw.cpu().numpy().reshape(-1) if hasattr(tw, 'cpu') else np.asarray(tw).reshape(-1)
    demand_arr = demand.cpu().numpy().reshape(-1) if hasattr(demand, 'cpu') else np.asarray(demand).reshape(-1)
    if 't_arrival' in full_instance:
        raw = full_instance['t_arrival']
        t_arr = raw.cpu().numpy().reshape(-1) if hasattr(raw, 'cpu') else np.asarray(raw).reshape(-1)
    else:
        t_arr = np.zeros(n_req, dtype=np.float32)
    arrivals = []
    for rid in range(n_req):
        pickup = n_depots + rid
        delivery = n_depots + n_req + rid
        arrivals.append({
            'req_id': rid,
            't_arrival': float(t_arr[rid]),
            't_pickup': float(tw_arr[pickup]),
            't_delivery': float(tw_arr[delivery]),
            'pickup_node': pickup,
            'delivery_node': delivery,
            'demand': float(abs(demand_arr[pickup])),
        })
    return sorted(arrivals, key=lambda row: row['t_arrival'])


def get_summary(log: List[Dict[str, Any]]) -> Dict[str, Any]:
    for row in reversed(log):
        if row.get('summary'):
            return row
    raise RuntimeError('dispatcher log has no summary row')


def _range_value(values, fallback, cast):
    if values is None:
        return (cast(fallback), cast(fallback))
    if len(values) != 2:
        raise ValueError('range arguments must contain exactly two values')
    lo, hi = cast(values[0]), cast(values[1])
    return (min(lo, hi), max(lo, hi))


def _sample_int(rng: np.random.Generator, bounds: Tuple[int, int]) -> int:
    lo, hi = bounds
    return int(rng.integers(lo, hi + 1))


def _sample_float(rng: np.random.Generator, bounds: Tuple[float, float]) -> float:
    lo, hi = bounds
    if abs(hi - lo) <= 1e-12:
        return float(lo)
    return float(rng.uniform(lo, hi))


def sample_episode_config(args: argparse.Namespace, rng: np.random.Generator) -> Dict[str, Any]:
    if args.fixed_regime:
        n_req = int(args.n_req)
        n_uav = int(args.n_uav)
        n_adr = int(args.n_adr)
        delta_minutes = float(args.delta_minutes)
    else:
        n_req = _sample_int(rng, _range_value(args.n_req_range, args.n_req, int))
        n_uav = _sample_int(rng, _range_value(args.n_uav_range, args.n_uav, int))
        n_adr = _sample_int(rng, _range_value(args.n_adr_range, args.n_adr, int))
        delta_minutes = _sample_float(rng, _range_value(args.delta_range, args.delta_minutes, float))

    n_depots_uav = max(1, min(int(args.n_depots_uav), max(n_uav, 1)))
    n_depots_adr = max(1, min(int(args.n_depots_adr), max(n_adr, 1)))
    return {
        'n_req': n_req,
        'n_uav': n_uav,
        'n_adr': n_adr,
        'n_depots_uav': n_depots_uav,
        'n_depots_adr': n_depots_adr,
        'delta_minutes': delta_minutes,
    }


def decision_cost_to_go(
    log: List[Dict[str, Any]],
    undelivered_penalty_scale: float = 1.0,
) -> np.ndarray:
    """Mandatory five-term quality plus explicitly assumed compute friction."""
    summary = get_summary(log)
    if (summary.get('service_policy') != 'mandatory' or summary.get('service_complete') is not True
            or int(summary.get('undelivered', -1)) != 0):
        raise ValueError('Gate reward requires complete mandatory service, not a missed-order price')
    if (undelivered_penalty_scale != 1.0 or float(summary.get('undelivered_penalty', 0)) != 0
            or float(summary.get('battery_penalty_cost', 0)) != 0):
        raise ValueError('Optional-service reward shaping is not allowed for mandatory gate training')
    step_rows = [row for row in log if not row.get('summary')]
    if not step_rows:
        return np.zeros(0, dtype=np.float32)

    step_costs = np.asarray([
        float(row.get('operating_cost_step', 0.0) or 0.0)
        + float(row.get('penalty_cost_step', 0.0) or 0.0)
        + float(row.get('battery_penalty_step', 0.0) or 0.0)
        + float(row.get('compute_cost_step', 0.0) or 0.0)
        for row in step_rows
    ], dtype=np.float32)
    step_costs[-1] += float(
        float(undelivered_penalty_scale)
        * float(summary.get('undelivered_penalty', 0.0) or 0.0)
    )
    cost_to_go_by_step = np.cumsum(step_costs[::-1], dtype=np.float32)[::-1]
    decision_indices = [
        idx for idx, row in enumerate(step_rows)
        if row.get('gate_prob_replan') is not None
    ]
    return cost_to_go_by_step[decision_indices].astype(np.float32, copy=False)


class GateValueBaseline(nn.Module):
    def __init__(
        self,
        input_dim: int = GATE_FEATURE_DIM + BASELINE_CONTEXT_DIM,
        hidden_dim: int = 64,
    ):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x.float()).squeeze(-1)


def train(args: argparse.Namespace) -> None:
    from release_checks import require_training_release
    require_training_release(
        allow_unvalidated=getattr(args, 'allow_unvalidated_model', False),
        acknowledge_stylized_scope=getattr(args, 'acknowledge_stylized_scope', False))
    require_current_checkpoint(args.rl_model, allow_legacy=getattr(args, 'allow_legacy_checkpoint', False))
    if args.undelivered_penalty_scale != 1.0:
        raise ValueError('Mandatory service cannot be replaced by missed-order reward shaping')
    friction = [args.compute_cost, args.compute_time_weight, args.compute_node_weight]
    if not np.isfinite(friction).all() or min(friction) < 0:
        raise ValueError('Compute friction must be finite and nonnegative')
    seed_all(args.seed)
    device = torch.device('cuda' if torch.cuda.is_available() and not args.cpu else 'cpu')
    solver = build_solver(args.rl_model, arch=args.arch)
    gate = ReplanGate(hidden_dim=args.hidden_dim).to(device)
    if abs(float(args.init_hold_bias)) > 1e-9:
        # Output head is gate.net[-1] = Linear(hidden_dim, 2); index 0 = hold logit.
        with torch.no_grad():
            gate.net[-1].bias.copy_(
                torch.tensor([float(args.init_hold_bias), 0.0], device=device)
            )
        print(f"[init] Set output bias to [hold={args.init_hold_bias:+.3f}, replan=0.000]")
    value_baseline = GateValueBaseline(hidden_dim=args.hidden_dim).to(device)
    controller = TorchReplanGateController(
        gate,
        device=device,
        stochastic=True,
        min_hold_ticks=args.min_hold_ticks,
        max_hold_ticks=args.max_hold_ticks,
    )
    opt = torch.optim.Adam(
        list(gate.parameters()) + list(value_baseline.parameters()),
        lr=args.lr,
    )

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    csv_path = out_dir / 'gate_training.csv'
    if csv_path.exists() or (out_dir / 'replan_gate.pt').exists():
        raise FileExistsError('Do not overwrite gate evidence/checkpoints; use a new output directory')
    configs = []
    for episode in range(args.episodes):
        cfg_rng = np.random.default_rng(np.random.SeedSequence([args.seed, episode, 1]))
        cfg = sample_episode_config(args, cfg_rng)
        cfg['temporal_peaks'] = bool(cfg_rng.uniform() < args.temporal_peaks_prob)
        configs.append(cfg)
    def factory(episode):
        cfg = configs[episode]
        return controlled_instance(seed=args.seed + episode, beta=getattr(args, 'beta', 0.0),
            **{key: cfg[key] for key in ('n_req', 'n_uav', 'n_adr', 'n_depots_uav', 'n_depots_adr')},
            temporal_peaks=cfg['temporal_peaks'], temporal_peaks_intensity=args.temporal_peaks_intensity)
    bank = freeze_bank(out_dir / 'training_bank.pt', dict(configs=configs, seed=args.seed,
        beta=getattr(args, 'beta', 0.0), temporal_peaks_intensity=args.temporal_peaks_intensity),
        factory, args.episodes)
    from experiment_manifest import file_sha256
    spec = {key: value for key, value in vars(args).items() if key not in ('output_dir', 'rl_model')}
    spec['router_sha256'] = file_sha256(args.rl_model)
    spec['friction_interpretation'] = 'assumed diagnostic sensitivity, not calibrated money'
    manifest_path = out_dir / 'experiment_manifest.json'
    manifest = begin_run(manifest_path, spec, bank_id=bank['bank_id'])

    with open(csv_path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=[
            'episode', 'total_cost', 'quality_cost', 'compute_cost',
            'replan_count', 'hold_count', 'loss', 'policy_loss',
            'value_loss', 'advantage',
            'n_decisions', 'n_req', 'n_uav', 'n_adr', 'delta_minutes',
            'temporal_peaks',
        ])
        writer.writeheader()

        for episode in range(args.episodes):
            seed_all(args.seed + episode)
            cfg = configs[episode]
            # Per-episode coin flip on bursty vs uniform arrivals.  Without this,
            # arrivals are Poisson (uniform-rate) and the gate's arrival-rate
            # features carry no information that beats a constant policy.
            use_peaks = cfg['temporal_peaks']
            import copy
            inst = copy.deepcopy(bank['instances'][episode])
            inst['n_depots'] = cfg['n_depots_uav'] + cfg['n_depots_adr']
            inst['n_req'] = cfg['n_req']
            fleet = make_fleet(
                n_uav=cfg['n_uav'], n_adr=cfg['n_adr'],
                n_depots_uav=cfg['n_depots_uav'],
                n_depots_adr=cfg['n_depots_adr'],
                full_instance=inst,
            )
            dispatcher = RollingHorizonDispatcher(
                solver,
                delta_minutes=cfg['delta_minutes'],
                shift_minutes=args.shift_minutes,
                replan_gate=controller,
                compute_cost_per_replan=args.compute_cost,
                compute_cost_time_weight=args.compute_time_weight,
                compute_cost_node_weight=args.compute_node_weight,
                service_policy='mandatory',
            )

            opt.zero_grad(set_to_none=True)
            log = dispatcher.run_shift(arrivals_from_instance(inst), fleet, inst)
            summary = get_summary(log)
            total_cost = float(summary['total_cost'])
            returns = decision_cost_to_go(
                log,
                undelivered_penalty_scale=float(args.undelivered_penalty_scale),
            )
            advantage = 0.0
            policy_loss_value = 0.0
            value_loss_value = 0.0
            n_decisions = 0

            if controller.log_probs:
                log_probs = torch.stack(controller.log_probs)
                lengths = (len(log_probs), len(controller.decision_features),
                           len(controller.decision_contexts), int(returns.size))
                if len(set(lengths)) != 1:
                    raise ValueError('Gate decisions/returns do not align; refusing silent truncation')
                n = lengths[0]
                n_decisions = int(n)
                if n > 0:
                    log_probs = log_probs[:n]
                    states = torch.cat([
                        torch.stack(controller.decision_features[:n]),
                        torch.stack(controller.decision_contexts[:n]),
                    ], dim=1)
                    returns_t = torch.as_tensor(
                        returns[:n],
                        dtype=torch.float32,
                        device=device,
                    ) / max(float(args.return_scale), 1e-9)
                    values = value_baseline(states)
                    adv_t = returns_t - values.detach()
                    entropy = torch.stack(controller.entropies[:n]).mean() if controller.entropies else torch.zeros((), device=device)
                    policy_loss = (adv_t * log_probs).mean()
                    policy_loss = policy_loss - args.entropy_coef * entropy
                    value_loss = F.mse_loss(values, returns_t)
                    loss = policy_loss + args.value_loss_coef * value_loss
                    advantage = float((adv_t.detach() * float(args.return_scale)).mean().cpu().item())
                    policy_loss_value = float(policy_loss.detach().cpu().item())
                    value_loss_value = float(value_loss.detach().cpu().item())
                else:
                    loss = torch.zeros((), dtype=torch.float32, device=device, requires_grad=True)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(
                    list(gate.parameters()) + list(value_baseline.parameters()),
                    args.grad_clip,
                )
                opt.step()
                loss_value = float(loss.detach().cpu().item())
            else:
                loss_value = 0.0

            row = {
                'episode': episode,
                'total_cost': total_cost,
                'quality_cost': float(summary.get('quality_cost', total_cost)),
                'compute_cost': float(summary.get('compute_cost', 0.0)),
                'replan_count': int(summary.get('replan_count', 0)),
                'hold_count': int(summary.get('hold_count', 0)),
                'loss': loss_value,
                'policy_loss': policy_loss_value,
                'value_loss': value_loss_value,
                'advantage': float(advantage),
                'n_decisions': n_decisions,
                'n_req': cfg['n_req'],
                'n_uav': cfg['n_uav'],
                'n_adr': cfg['n_adr'],
                'delta_minutes': cfg['delta_minutes'],
                'temporal_peaks': int(use_peaks),
            }
            writer.writerow(row)
            f.flush()
            if (episode + 1) % args.save_every == 0 or episode == args.episodes - 1:
                atomic_checkpoint(out_dir / 'replan_gate.pt', {
                    'model_state_dict': gate.state_dict(),
                    'value_state_dict': value_baseline.state_dict(),
                    'episode': episode,
                    'args': vars(args),
                    'optimizer_state_dict': opt.state_dict(),
                    'torch_rng_state': torch.get_rng_state(),
                    'release_contract': release_contract(),
                    'bank_id': bank['bank_id'],
                    'run_id': manifest['run_id'],
                    'scientific_release_validated': True,
                })
            print(
                f"episode={episode} total={total_cost:.3f} "
                f"quality={row['quality_cost']:.3f} compute={row['compute_cost']:.3f} "
                f"replans={row['replan_count']} holds={row['hold_count']} "
                f"decisions={row['n_decisions']} n_req={cfg['n_req']} "
                f"peaks={int(use_peaks)} "
                f"policy={policy_loss_value:.3f} value={value_loss_value:.3f}",
                flush=True,
            )
    finish_run(manifest_path, manifest, [csv_path, out_dir / 'replan_gate.pt', out_dir / 'training_bank.pt'])


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument('--allow-unvalidated-model', action='store_true',
                        help='Diagnostic training flag retained for backward compatibility.')
    parser.add_argument('--acknowledge-stylized-scope', action='store_true',
                        help='Diagnostic training flag retained for backward compatibility.')
    parser.add_argument('--rl-model', required=True)
    parser.add_argument('--allow-legacy-checkpoint', action='store_true', help='Diagnostic stale checkpoint use only.')
    parser.add_argument('--beta', type=float, default=0.0,
                        help='Independent payload/slack rank dependence [-1,1]; 0 is the neutral control.')
    parser.add_argument('--arch', default='hetgat', choices=['hetgat', 'simplegat'])
    parser.add_argument('--episodes', type=int, default=200)
    parser.add_argument('--seed', type=int, default=1234)
    parser.add_argument('--output-dir', default='gate_training_results')
    parser.add_argument('--cpu', action='store_true')
    parser.add_argument('--n-req', type=int, default=25)
    parser.add_argument('--n-uav', type=int, default=3)
    parser.add_argument('--n-adr', type=int, default=3)
    parser.add_argument('--fixed-regime', action='store_true',
                        help='Use --n-req/--n-uav/--n-adr/--delta-minutes instead of sampled training regimes.')
    parser.add_argument('--n-req-range', type=int, nargs=2, default=[15, 60])
    parser.add_argument('--n-uav-range', type=int, nargs=2, default=[2, 8])
    parser.add_argument('--n-adr-range', type=int, nargs=2, default=[2, 8])
    parser.add_argument('--delta-range', type=float, nargs=2, default=[2.0, 15.0])
    parser.add_argument('--n-depots-uav', type=int, default=2)
    parser.add_argument('--n-depots-adr', type=int, default=2)
    parser.add_argument('--delta-minutes', type=float, default=5.0)
    parser.add_argument('--shift-minutes', type=float, default=120.0)
    parser.add_argument('--compute-cost', type=float, default=0.0,
                        help='Assumed fixed friction per replan; not calibrated money. Default zero control.')
    parser.add_argument('--compute-time-weight', type=float, default=0.0,
                        help='Wall-clock solve-time cost weight; keep 0 for reproducible training.')
    parser.add_argument('--compute-node-weight', type=float, default=0.0,
                        help='Assumed active-node friction, not calibrated money. Default zero control.')
    parser.add_argument('--hidden-dim', type=int, default=64)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--entropy-coef', type=float, default=0.01)
    parser.add_argument('--value-loss-coef', type=float, default=0.5)
    parser.add_argument('--return-scale', type=float, default=100.0)
    parser.add_argument('--baseline-decay', type=float, default=0.9,
                        help='Deprecated; kept for compatibility with older training commands.')
    parser.add_argument('--grad-clip', type=float, default=1.0)
    parser.add_argument('--min-hold-ticks', type=int, default=0)
    parser.add_argument('--max-hold-ticks', type=int, default=8)
    parser.add_argument('--save-every', type=int, default=25)
    parser.add_argument('--init-hold-bias', type=float, default=0.0,
                        help='Initial bias added to the hold logit at the gate output head. '
                             'Positive values bias the gate toward holding at init, which makes '
                             'it easier to learn "when to replan" instead of "when to hold."')
    parser.add_argument('--temporal-peaks-prob', type=float, default=0.0,
                        help='Per-episode probability of generating bursty (peaked) arrivals '
                             'instead of homogeneous Poisson.  Under uniform arrivals the gate '
                             'has no informative arrival-rate signal, so the cost-minimising '
                             'policy degenerates to a constant.  Setting this to e.g. 0.5 mixes '
                             'in temporal_peaks=True episodes so the gate learns when to '
                             'replan during demand bursts.')
    parser.add_argument('--undelivered-penalty-scale', type=float, default=1.0,
                        help='Legacy option; must remain 1. Mandatory delivery is a constraint, not reward shaping.')
    parser.add_argument('--temporal-peaks-intensity', type=float, default=1.0,
                        help='Sharpness of the bursty arrivals.  1.0 = mild bursts (SDapproximately 3.3min, '
                             '2-4 peaks, 40%% burst share).  2.0 = sharp (SD=2min, 3-6 peaks, '
                             '55%% burst share).  3.0 = extreme (SD=1.25min, 4-7 peaks, 70%% '
                             'burst share, plus a 15-25min quiet window).  Higher intensities '
                             'create sharper hold-vs-replan decisions that the gate must time '
                             'precisely; intensity≤1 reproduces the original behaviour.')
    return parser


if __name__ == '__main__':
    train(build_parser().parse_args())
