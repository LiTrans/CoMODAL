
from artifacts import load_tensor_file
import csv
import os
import time
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.optim.lr_scheduler import LambdaLR
import contextlib
from collections import OrderedDict, namedtuple
from itertools import product

from VRP_Actor import Model
from VRP_CoopActorCritic import CoopActorCritic
from creat_vrp import creat_data, reward1, build_dynamic_masked_batch, build_rolling_residual_batch
from rolloutBaseline1 import RolloutBaseline
from reward import (with_mandatory_contract, route_checkpoint_contract,
                     validate_route_checkpoint, IncompleteTensorRouteError)
from rl_value_baseline import RollingStateCritic, attach_window_context

try:
    from torch.utils.tensorboard import SummaryWriter
    _TB_AVAILABLE = True
except ImportError:
    _TB_AVAILABLE = False

IS_DIST    = 'RANK' in os.environ and int(os.environ.get('WORLD_SIZE', 1)) > 1
RANK       = int(os.environ.get('RANK', 0))
LOCAL_RANK = int(os.environ.get('LOCAL_RANK', 0))
WORLD_SIZE = int(os.environ.get('WORLD_SIZE', 1))

if IS_DIST:
    if not torch.cuda.is_available():
        raise RuntimeError(
            f'Distributed launch requested (WORLD_SIZE={WORLD_SIZE}) '
            f'but no CUDA GPUs are visible on this rank (LOCAL_RANK={LOCAL_RANK}). '
            f'Check that the cuda module is loaded and the job was allocated GPUs.'
        )
    torch.cuda.set_device(LOCAL_RANK)
    dist.init_process_group(backend='nccl', device_id=torch.device(f'cuda:{LOCAL_RANK}'))

def _get_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device(f'cuda:{LOCAL_RANK}')
    try:
        import torch_directml
        _dml = torch_directml.device()
        _src = torch.zeros(5, 4, device=_dml)
        _idx = torch.zeros(5, dtype=torch.long, device=_dml)
        torch.zeros(3, 4, device=_dml).scatter_add_(
            0, _idx.unsqueeze(1).expand_as(_src), _src
        )
        if RANK == 0:
            print('[device] DirectML active — Windows GPU via DirectX 12.')
            print('         Note: num_workers must be 0 and pin_memory=False on DirectML.')
        return _dml
    except Exception:
        if RANK == 0:
            print('[device] DirectML scatter_add_ probe failed — falling back to CPU.')
            print('         Training is slower on CPU but fully correct.')
    return torch.device('cpu')

device = _get_device()
_is_directml = 'privateuseone' in str(device)

if RANK == 0:
    n_gpus = torch.cuda.device_count() if torch.cuda.is_available() else 0
    print(f'[device] {device}  |  world={WORLD_SIZE}  |  CUDA GPUs={n_gpus}')

def _inner(model: nn.Module) -> nn.Module:
    return model.module if hasattr(model, 'module') else model

RUNG = 'A'

RUNG_ORDER = ['A', 'B', 'C', 'D']

CACHE_DIR = 'cache'
os.makedirs(CACHE_DIR, exist_ok=True)

RUNG_CONFIG = {
    'A': dict(n_req=5,  n_uav=2,  n_adr=2, n_depots_uav=1, n_depots_adr=1,
              train_size=128_000,  batch_size=512, val_size=10_240, streaming=True,
              grad_accum=1, conv_layers=3),
    'B': dict(n_req=10, n_uav=4,  n_adr=3, n_depots_uav=1, n_depots_adr=1,
              train_size=256_000,  batch_size=512, val_size=10_240, streaming=True,
              grad_accum=1, conv_layers=3),
    'C': dict(n_req=25, n_uav=5,  n_adr=4, n_depots_uav=2, n_depots_adr=2,
              train_size=320_000,  batch_size=256, val_size=5_120,  streaming=True,
              grad_accum=2, conv_layers=3),
    'D': dict(n_req=60, n_uav=10, n_adr=8, n_depots_uav=3, n_depots_adr=3,
              train_size=160_000,  batch_size=64,  val_size=3_200,  streaming=True,
              grad_accum=16, conv_layers=3),
}

import argparse as _argparse
_cli = _argparse.ArgumentParser()
_cli.add_argument('--allow-unvalidated-model', action='store_true',
                  help='Legacy diagnostic training only; does not certify a scientific release.')
_cli.add_argument('--acknowledge-stylized-scope', action='store_true',
                  help='Diagnostic training flag retained for backward compatibility.')
_cli.add_argument('--rung',   default=None, choices=['A', 'B', 'C', 'D'])
_cli.add_argument('--epochs', type=int, default=None)
_cli.add_argument('--route-selector', default='sequential',
                  choices=['sequential', 'legacy_parallel'],
                  help='sequential = agents decoded in order with conditional log-probs; '
                       'legacy_parallel = Gumbel-auction baseline (zero gradient through contested pickups).')
_cli.add_argument('--continuation-check', default='off',
                  choices=['off', 'l1', 'l2', 'l3', 'onboard'],
                  help='Continuation certificate level. off = legacy; l1 = depot-escape + last-pickup; '
                       'l2 = l1 + check every pending pickup; l3 = chain feasibility via depot-recharge.')
_cli.add_argument('--seed', type=int, default=0,
                  help='Master training seed (torch CPU+CUDA, numpy, random). Separate from eval seeds.')
_cli.add_argument('--init-actor', default=None,
                  help='Warmstart checkpoint path; model_state_dict loaded before training. '
                       'Takes precedence over --warmstart.')
_cli.add_argument('--max-decode-steps', type=int, default=None,
                  help='Cap the autoregressive decoder loop length per episode. '
                       'Default formula is 4*num_nodes + 4*n_agents.')
_cli.add_argument('--train-size', type=int, default=None,
                  help='Override train_size per epoch (supersedes RUNG_CONFIG default).')
_cli.add_argument('--val-size', type=int, default=None,
                  help='Override val_size per rung (supersedes RUNG_CONFIG default).')
_cli.add_argument('--smoke',  action='store_true',
                  help='CPU smoke-test: tiny data, 5 epochs, validates full pipeline')
_cli.add_argument('--overfit', action='store_true',
                  help='Overfit diagnostic: train on a single fixed batch.')
_cli.add_argument('--rloo-k', type=int, default=0,
                  help='RLOO trajectories per instance (0 = rollout baseline). Recommended: 4-8.')
_cli.add_argument('--variant', default='',
                  help='Suffix appended to checkpoint folder and CSV log name.')
_cli.add_argument('--warmstart', action='store_true',
                  help='Warm-start from the previous rung best_actor.pt (curriculum).')
_cli.add_argument('--num-workers', type=int, default=None,
                  help='Explicit DataLoader worker count. None = SLURM-CPUS heuristic, 0 = off.')
_cli.add_argument('--delivery-shaping-k', type=float, default=50.0,
                  help='PBRS delivery-shaping coefficient (phi(s) = -k * unserved_count). '
                       'Training-only; policy-invariant under Ng, Harada and Russell 1999.')
_cli.add_argument('--arch', default='hetgat',
                  choices=['hetgat', 'simplegat', 'koolam',
                           'hetero_attn_pdp', 'gann'],
                  help='Encoder architecture. hetgat = full 4D edge features (default). '
                       'simplegat/koolam/hetero_attn_pdp/gann are ablation baselines sharing CoopDecoder.')
_cli.add_argument('--available-fleet', default='both', choices=['both', 'uav', 'adr'],
                  help='Singleton-coalition training: "uav"/"adr" zero the complementary fleet in available_mask.')
_cli.add_argument('--edge-features', default='on', choices=['on', 'off'],
                  help='"off" zeros encoder edge features before the GAT convs.')
_cli.add_argument('--decoder-biases', default='on', choices=['on', 'off'],
                  help='"off" removes dist / urgency / live-margin biases from decoder logits.')
_cli.add_argument('--margin-bias', default='off', choices=['on', 'off'],
                  help='"on" adds per-node post-visit battery-margin bias to the decoder logit.')
_cli.add_argument('--critic', default='off', choices=['on', 'off'],
                  help='Adds a value head to route Model for actor-critic advantages.')
_cli.add_argument('--policy', default='route', choices=['route', 'coop'],
                  help='route = node-route pointer actor; coop = explicit vehicle-request assignment actor-critic.')
_cli.add_argument('--shadow-val-size', type=int, default=256,
                  help='Instances per shadow validation seed.')
_cli.add_argument('--val-seeds', type=int, nargs='+',
                  default=list(range(50_000, 50_000 + 10 * 256, 256)),
                  help='Validation seeds for multi-seed shadow plot (non-overlapping by shadow-val-size).')
_cli.add_argument('--hard-val-seeds', type=int, nargs='+',
                  default=list(range(60_000, 60_000 + 10 * 256, 256)),
                  help='Seeds for tight-TW hard validation sets.')
_cli.add_argument('--entropy-coef', type=float, default=0.01,
                  help='Entropy regularisation coefficient.')
_cli.add_argument('--adv-min-std', type=float, default=0.5,
                  help='Minimum std floor in advantage normalisation.')
_cli.add_argument('--lr', type=float, default=1e-4,
                  help='Base learning rate before DDP sqrt-scaling and warmup.')
_cli.add_argument('--lr-decay', type=float, default=0.97,
                  help='Multiplicative LR decay per epoch after warmup.')
_cli.add_argument('--warmup-epochs', type=int, default=5,
                  help='Linear LR warmup length in epochs.')
_cli.add_argument('--baseline-refresh-after-stale', type=int, default=5,
                  help='Force rollout-baseline refresh after this many consecutive rejections; 0 disables.')
_cli.add_argument('--temperature', type=float, default=1.2,
                  help='Initial sampling temperature, linearly annealed to --temperature-end.')
_cli.add_argument('--temperature-end', type=float, default=1.0,
                  help='Final sampling temperature.')
_cli.add_argument('--battery-slack-k', type=float, default=0.0,
                  help='PBRS mid-shift recharge shaping coefficient. Training-only.')
_cli.add_argument('--delivery-shaping-mode', default='episodic',
                  choices=['episodic', 'pbrs_step'],
                  help='"episodic" = end-of-rollout; "pbrs_step" = per-step (equivalent under gamma=1).')
_cli.add_argument('--battery-mask', default='off', choices=['on', 'off'],
                  help='Hard-mask customer nodes whose post-visit battery would drop below reserve_frac. '
                       'Depots always legal.')
_cli.add_argument('--battery-mask-reserve-frac', type=float, default=0.10,
                  help='Reserve fraction (both fleets) for battery hard-mask.')
_cli.add_argument('--battery-mask-reserve-frac-uav', type=float, default=None,
                  help='Per-fleet UAV override (default inherits --battery-mask-reserve-frac).')
_cli.add_argument('--battery-mask-reserve-frac-adr', type=float, default=None,
                  help='Per-fleet ADR override (default inherits --battery-mask-reserve-frac).')
_cli.add_argument('--battery-mask-lookahead', default='off', choices=['on', 'off'],
                  help='Extend hard-mask margin via pickup->delivery->depot chain.')
_cli.add_argument('--battery-mandatory-recharge', default='off', choices=['on', 'off'],
                  help='At low battery, allow only depots and the pending-delivery node.')
_cli.add_argument('--battery-mandatory-recharge-buffer', type=float, default=0.10,
                  help='Soft buffer above reserve_frac at which mandatory-recharge fires.')
_cli.add_argument('--paper-strict-depot-mask', choices=['on', 'off'], default='on',
                  help='Section 4.4 depot-as-fallback rule.')
_cli.add_argument('--paper-strict-depot-margin', type=float, default=0.10,
                  help='Battery fraction margin above reserve at which paper-strict depot mask stops firing.')
_cli.add_argument('--no-encoder-attn-mask', action='store_true',
                  help='Disable mask_adjacency softmax filter inside HetGatConv/SimpleGatConv.')
_cli.add_argument('--chain-lookahead-in-mask', choices=['on', 'off'], default='on',
                  help='Decoder 2-leg chain-lookahead (redundant with vrpUpdate._safe_continuation_mask).')
_cli.add_argument('--dyn-features', default='basic',
                  choices=['basic', 'extended'],
                  help='CoopDecoder per-vehicle state width. basic = 3 scalars (Section 4.2); extended = 10.')
_cli.add_argument('--dynamic', action=_argparse.BooleanOptionalAction, default=True,
                  help='On-demand training: random fraction of orders revealed per batch.')
_cli.add_argument('--mix-hard-frac', type=float, default=0.3,
                  help='Fraction of streaming instances with non-uniform temporal/spatial/demand pressure.')
_cli.add_argument('--mode-bias', type=float, default=0.2,
                  help='Fraction of orders biased toward vehicle specialisation.')
_cli.add_argument('--uav-reserve-frac', type=float, default=None,
                  help='Override uav_battery.reserve_soc.')
_cli.add_argument('--adr-reserve-frac', type=float, default=None,
                  help='Override adr_battery.reserve_soc.')
_cli.add_argument('--max-recharges-per-vehicle', type=int, default=None,
                  help='Per-vehicle depot-recharge cap during training. None = unlimited.')
_cli.add_argument('--oracle-reward-path', type=str, default=None,
                  help='Path to oracle JSON; episodes use oracle instances + fractional-excess reward term.')
_cli.add_argument('--oracle-reward-k', type=float, default=2.0,
                  help='Coefficient on the oracle-ratio penalty term.')
_cli.add_argument('--map-pool-size', type=int, default=128,
                  help='Reusable synthetic map templates for streaming training. 0 = rebuild every sample.')
_cli.add_argument('--residual', action=_argparse.BooleanOptionalAction, default=False,
                  help='Residual rolling-horizon training with mid-shift start_time/battery. Requires --dynamic.')
_cli.add_argument('--residual-frac', type=float, default=0.7,
                  help='Fraction of dynamic batches trained as residual rolling-horizon states.')
_cli.add_argument('--replan-gate', action=_argparse.BooleanOptionalAction, default=False,
                  help='Replan-gate aware training. Requires --residual.')
_cli.add_argument('--gate-cost', type=float, default=0.05,
                  help='Per-replan-event cost subtracted from reward in gate training.')
_cli.add_argument('--gate-t-max', type=float, default=115.0,
                  help='Max t_current for residual batches in gate training.')
_cli.add_argument('--rolling-critic', action=_argparse.BooleanOptionalAction, default=False,
                  help='Use learned RollingStateCritic instead of greedy rollout baseline.')
_cli.add_argument('--critic-coef', type=float, default=0.5,
                  help='Weight on critic MSE loss.')
_cli.add_argument('--dispatch-window-min', type=float, default=2.0,
                  help='Min dispatch-window length (minutes) for rolling state features.')
_cli.add_argument('--dispatch-window-max', type=float, default=30.0,
                  help='Max dispatch-window length (minutes) for rolling state features.')
_cli.add_argument('--generalized', action=_argparse.BooleanOptionalAction, default=False,
                  help='Train coop policy on mixed-size synthetic batches.')
_cli.add_argument('--n-req-range', type=int, nargs=2, default=(5, 60),
                  metavar=('MIN', 'MAX'),
                  help='Inclusive request-count range for --generalized training.')
_cli.add_argument('--n-uav-range', type=int, nargs=2, default=(1, 10),
                  metavar=('MIN', 'MAX'),
                  help='Inclusive UAV-count range for --generalized training.')
_cli.add_argument('--n-adr-range', type=int, nargs=2, default=(1, 8),
                  metavar=('MIN', 'MAX'),
                  help='Inclusive ADR-count range for --generalized training.')
_cli.add_argument('--availability-range', type=float, nargs=2, default=(0.3, 1.0),
                  metavar=('MIN', 'MAX'),
                  help='Fraction of vehicles available in synthetic rolling states for '
                       '--generalized training.')
_cli_args, _ = _cli.parse_known_args()
if _cli_args.rung is not None:
    RUNG = _cli_args.rung
_CLI_MAX_EPOCHS  = _cli_args.epochs
_CLI_MAX_DECODE_STEPS = _cli_args.max_decode_steps
_CONTINUATION_CHECK = _cli_args.continuation_check
_ROUTE_SELECTOR     = _cli_args.route_selector
_SMOKE           = _cli_args.smoke

# Apply master seed before any model / optimizer / sampler is constructed.
import random as _py_random
import numpy as _np
torch.manual_seed(int(_cli_args.seed))
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(int(_cli_args.seed))
_np.random.seed(int(_cli_args.seed))
_py_random.seed(int(_cli_args.seed))
print(f'[seed] master training seed = {int(_cli_args.seed)}')
_OVERFIT         = _cli_args.overfit
_RLOO_K          = _cli_args.rloo_k
_VARIANT         = _cli_args.variant
_WARMSTART       = _cli_args.warmstart
_INIT_ACTOR      = _cli_args.init_actor
_DELIVERY_SHAPING_K = max(0.0, float(_cli_args.delivery_shaping_k))
_CLI_NUM_WORKERS = _cli_args.num_workers
_ARCH            = _cli_args.arch
_POLICY          = _cli_args.policy
_AVAILABLE_FLEET = _cli_args.available_fleet
_SHADOW_VAL_SIZE = _cli_args.shadow_val_size
_VAL_SEEDS       = _cli_args.val_seeds
_HARD_VAL_SEEDS  = _cli_args.hard_val_seeds
_EDGE_DIM        = 1 if _ARCH == 'simplegat' else 4
_ENTROPY_COEF    = _cli_args.entropy_coef
_BASE_LR         = _cli_args.lr
_LR_DECAY        = _cli_args.lr_decay
_WARMUP_EPOCHS   = _cli_args.warmup_epochs
_BL_REFRESH_STALE = max(0, int(_cli_args.baseline_refresh_after_stale))
_EDGE_FEATURES   = (_cli_args.edge_features == 'on')
_DECODER_BIASES  = (_cli_args.decoder_biases == 'on')
_MARGIN_BIAS     = (_cli_args.margin_bias == 'on')
_CRITIC          = (_cli_args.critic == 'on')
_T_START         = _cli_args.temperature
_T_END           = _cli_args.temperature_end
_BATTERY_SLACK_K = float(_cli_args.battery_slack_k)
_DELIVERY_SHAPING_MODE = _cli_args.delivery_shaping_mode
_DYN_FEATURES    = _cli_args.dyn_features
_BATTERY_MASK    = (_cli_args.battery_mask == 'on')
_BATTERY_MASK_RESERVE_FRAC = float(_cli_args.battery_mask_reserve_frac)
_BATTERY_MASK_RESERVE_FRAC_UAV = (
    float(_cli_args.battery_mask_reserve_frac_uav)
    if _cli_args.battery_mask_reserve_frac_uav is not None else None
)
_BATTERY_MASK_RESERVE_FRAC_ADR = (
    float(_cli_args.battery_mask_reserve_frac_adr)
    if _cli_args.battery_mask_reserve_frac_adr is not None else None
)
_BATTERY_MASK_LOOKAHEAD = (_cli_args.battery_mask_lookahead == 'on')
_BATTERY_MANDATORY_RECHARGE = (_cli_args.battery_mandatory_recharge == 'on')
_BATTERY_MANDATORY_RECHARGE_BUFFER = float(_cli_args.battery_mandatory_recharge_buffer)
_PAPER_STRICT_DEPOT_MASK = (_cli_args.paper_strict_depot_mask == 'on')
_PAPER_STRICT_DEPOT_MARGIN = float(_cli_args.paper_strict_depot_margin)
_ENCODER_ATTN_MASK = not bool(_cli_args.no_encoder_attn_mask)
_CHAIN_LOOKAHEAD_IN_MASK = (_cli_args.chain_lookahead_in_mask == 'on')
_DYNAMIC         = _cli_args.dynamic
_MIX_HARD_FRAC   = _cli_args.mix_hard_frac
_MODE_BIAS       = _cli_args.mode_bias
_UAV_RESERVE_FRAC = _cli_args.uav_reserve_frac
_ADR_RESERVE_FRAC = _cli_args.adr_reserve_frac
_MAX_RECHARGES_PER_VEH = getattr(_cli_args, 'max_recharges_per_vehicle', None)
_MAP_POOL_SIZE   = max(0, _cli_args.map_pool_size)

# Oracle-ratio reward shaping. Loaded once at startup; the
# training loop uses _ORACLE_DB entries as a fixed training set (implicit
# overfit) and adds a fractional-excess term to the per-episode reward.
_ORACLE_REWARD_PATH = getattr(_cli_args, 'oracle_reward_path', None)
_ORACLE_REWARD_K    = float(getattr(_cli_args, 'oracle_reward_k', 2.0))
_ORACLE_DB          = None
if _ORACLE_REWARD_PATH:
    import json as _json_oracle
    with open(_ORACLE_REWARD_PATH) as _f_oracle:
        _ORACLE_DB = _json_oracle.load(_f_oracle)
    print(f'[oracle] loaded {len(_ORACLE_DB)} instances from {_ORACLE_REWARD_PATH}')
    print(f'[oracle] fractional-excess penalty K = {_ORACLE_REWARD_K}')
    _OVERFIT = True   # oracle-reward mode forces overfit on the oracle set


def _build_physical_config_override():
    """Build the physical_config dict used for reserve-30 (T2D) ablation.

    Returns None if neither reserve-frac knob is set. Otherwise returns a
    dict passing ``native_physical_configuration``'s terminal-safety and
    accounting-gap gates: ``legacy_leg_only`` continuation, non-None depot
    phase durations, and the user's reserve-soc overrides.
    """
    if _UAV_RESERVE_FRAC is None and _ADR_RESERVE_FRAC is None:
        return None
    from hardware_profiles import (BatteryPolicy, FLYCART30_SINGLE,
                                     PhysicalConfiguration, STARSHIP_2017,
                                     UAVPhasePolicy)
    uav_reserve = float(_UAV_RESERVE_FRAC) if _UAV_RESERVE_FRAC is not None else 0.0
    adr_reserve = float(_ADR_RESERVE_FRAC) if _ADR_RESERVE_FRAC is not None else 0.0
    phases = UAVPhasePolicy(depot_landing_min=1.0, depot_takeoff_min=1.0)
    config = PhysicalConfiguration(
        uav_battery=BatteryPolicy(FLYCART30_SINGLE.identifier,
                                   reserve_soc=uav_reserve),
        adr_battery=BatteryPolicy(STARSHIP_2017.identifier,
                                   reserve_soc=adr_reserve),
        uav_phases=phases,
        continuation_policy='legacy_leg_only',
    )
    return config.to_dict()


_PHYSICAL_CONFIG_OVERRIDE = _build_physical_config_override()
_ADV_MIN_STD     = _cli_args.adv_min_std
_RESIDUAL        = _cli_args.residual
_RESIDUAL_FRAC   = max(0.0, min(1.0, _cli_args.residual_frac))
_REPLAN_GATE     = getattr(_cli_args, 'replan_gate', False)
_GATE_COST       = float(getattr(_cli_args, 'gate_cost', 0.05))
_GATE_T_MAX      = float(getattr(_cli_args, 'gate_t_max', 115.0))
_USE_CRITIC      = _cli_args.rolling_critic
_CRITIC_COEF     = _cli_args.critic_coef
_WIN_MIN         = max(0.5, float(_cli_args.dispatch_window_min))
_WIN_MAX         = max(_WIN_MIN, float(_cli_args.dispatch_window_max))
_GENERALIZED     = _cli_args.generalized
_N_REQ_RANGE     = tuple(int(x) for x in _cli_args.n_req_range)
_N_UAV_RANGE     = tuple(int(x) for x in _cli_args.n_uav_range)
_N_ADR_RANGE     = tuple(int(x) for x in _cli_args.n_adr_range)
_AVAIL_RANGE     = tuple(float(x) for x in _cli_args.availability_range)

if _GENERALIZED and _POLICY != 'coop':
    raise ValueError('--generalized currently requires --policy coop.')
if _GENERALIZED and _OVERFIT:
    raise ValueError('--generalized and --overfit are mutually exclusive.')

if _ARCH == 'simplegat' and not _VARIANT:
    _VARIANT = 'simplegat'
if not _DYNAMIC and not _VARIANT:
    _VARIANT = 'static'

if _SMOKE:
    if _cli_args.rung is None:
        RUNG = 'A'
    RUNG_CONFIG[RUNG].update(
        train_size=256, batch_size=32, val_size=64, streaming=False
    )
    _SHADOW_VAL_SIZE = min(_SHADOW_VAL_SIZE, 32)
    if _CLI_MAX_EPOCHS is None:
        _CLI_MAX_EPOCHS = 5

# Explicit --train-size / --val-size override.
# Applied AFTER --smoke/--overfit to allow mid-smoke recipes.
if _cli_args.train_size is not None:
    RUNG_CONFIG[RUNG]['train_size'] = int(_cli_args.train_size)
if _cli_args.val_size is not None:
    RUNG_CONFIG[RUNG]['val_size'] = int(_cli_args.val_size)

if _OVERFIT:
    # Streaming can emit variable-length tensors; force a small cached dataset.
    if _cli_args.rung is None:
        RUNG = 'A'
    RUNG_CONFIG[RUNG].update(
        train_size=512, batch_size=512, val_size=64, streaming=False
    )
    _SHADOW_VAL_SIZE = min(_SHADOW_VAL_SIZE, 32)

cfg          = RUNG_CONFIG[RUNG]
n_req        = cfg['n_req']
n_uav        = cfg['n_uav']
n_adr        = cfg['n_adr']
n_depots_uav = cfg['n_depots_uav']
n_depots_adr = cfg['n_depots_adr']
n_depots     = n_depots_uav + n_depots_adr
steps        = n_depots + n_req * 2

max_grad_norm = 2.0

_QUICK_MAX_EPOCHS = None

BASELINE_EVAL_FREQ = int(os.environ.get('BL_EVAL_FREQ', '1'))


def adv_normalize(adv: torch.Tensor) -> torch.Tensor:
    if IS_DIST:
        mean = adv.mean()
        dist.all_reduce(mean, op=dist.ReduceOp.AVG)
        var = ((adv - mean) ** 2).mean()
        dist.all_reduce(var, op=dist.ReduceOp.AVG)
        std = var.sqrt()
    else:
        mean, std = adv.mean(), adv.std()
    if torch.isnan(std):
        return adv - mean
    return (adv - mean) / std.clamp(min=_ADV_MIN_STD)

def _prep_batch(batch: dict) -> dict:
    # Inject physical_config override before with_mandatory_contract so
    # reward and simulator both see the elevated reserve_soc.
    if _PHYSICAL_CONFIG_OVERRIDE is not None and 'physical_config' not in batch:
        batch = dict(batch)
        batch['physical_config'] = _PHYSICAL_CONFIG_OVERRIDE
    batch = with_mandatory_contract(batch)
    if _CLI_MAX_DECODE_STEPS is not None:
        batch = dict(batch) if not isinstance(batch, dict) else batch
        batch['max_decode_steps'] = int(_CLI_MAX_DECODE_STEPS)
    if _MAX_RECHARGES_PER_VEH is not None:
        batch = dict(batch)
        batch.setdefault('max_recharges_per_vehicle', int(_MAX_RECHARGES_PER_VEH))
    if _MAX_RECHARGES_PER_VEH is not None and 'remaining_recharges' not in batch:
        _dem = batch.get('demand')
        if _dem is not None:
            import torch as _t
            _dem_t = _t.as_tensor(_dem)
            _B = _dem_t.shape[0] if _dem_t.ndim >= 2 else 1
            _A = int(n_uav) + int(n_adr)
            batch['remaining_recharges'] = _t.full(
                (_B, _A), int(_MAX_RECHARGES_PER_VEH), dtype=_t.long,
                device=_dem_t.device)
    if _EDGE_DIM == 1:
        batch = dict(batch)
        batch['edge_attr_uav'] = batch['edge_attr_uav'][..., 1:2]
        batch['edge_attr_adr'] = batch['edge_attr_adr'][..., 1:2]
    if _AVAILABLE_FLEET != 'both':
        batch = _apply_fleet_restriction(dict(batch))
    return batch


def _apply_fleet_restriction(batch: dict) -> dict:
    """Zero the complementary fleet's available_mask for singleton training;
    create the mask if missing (build_dynamic_masked_batch omits it)."""
    import torch as _torch
    mask = batch.get('available_mask')
    if not isinstance(mask, _torch.Tensor):
        n_agents = n_uav + n_adr
        B = batch['x'].shape[0] if 'x' in batch else 1
        device = batch['x'].device if 'x' in batch else _torch.device('cpu')
        mask = _torch.ones(B, n_agents, device=device)
    else:
        mask = mask.clone()
    if _AVAILABLE_FLEET == 'uav':
        mask[..., n_uav:] = 0.0
    elif _AVAILABLE_FLEET == 'adr':
        mask[..., :n_uav] = 0.0
    batch['available_mask'] = mask
    return batch

def _active_request_mask_from_initial(initial_visited, batch_size,
                                      n_req_i=None, n_depots_i=None):
    n_req_i = n_req if n_req_i is None else int(n_req_i)
    n_depots_i = n_depots if n_depots_i is None else int(n_depots_i)
    if initial_visited is None:
        return torch.ones(batch_size, n_req_i, dtype=torch.bool, device=device)
    iv = initial_visited.view(batch_size, n_depots_i + 2 * n_req_i)
    return iv[:, n_depots_i + n_req_i:n_depots_i + 2 * n_req_i] <= 0.5

def _policy_forward(model, batch, greedy, T=1.0, training=False,
                    initial_visited=None, start_node=None, start_time=None,
                    start_battery=None, start_capacity=None,
                    n_uav_i=None, n_adr_i=None):
    n_uav_i = n_uav if n_uav_i is None else int(n_uav_i)
    n_adr_i = n_adr if n_adr_i is None else int(n_adr_i)
    batch = with_mandatory_contract(batch)
    # Train on partial-prefix trajectories (complete_episode would reject early).
    if training:
        batch = dict(batch)
        batch['route_scope'] = 'rolling_prefix'

    if _POLICY == 'coop':
        # CoopActorCritic reads start_* from batch, returns dict with value+window_logits.
        _b = batch
        if any(v is not None for v in (start_node, start_time, start_battery, start_capacity)):
            _b = dict(batch)
            if start_node is not None:
                _b['start_node'] = start_node
            if start_time is not None:
                _b['start_time'] = start_time
            if start_battery is not None:
                _b['start_battery'] = start_battery
            if start_capacity is not None:
                _b['start_capacity'] = start_capacity
        out = model(_b, n_uav_i, n_adr_i, greedy=greedy, initial_visited=initial_visited)
        tour_idx = out['tour_idx']
        log_p = out['log_p']
        time_tensor = out['time']
        value = out['value']
        window_logits = out['window_logits']
        # Coop decoder has no per-step entropy; return zero placeholder.
        _B, _A = log_p.shape[0], log_p.shape[1]
        entropy = torch.zeros(_B, _A, device=log_p.device)
        return tour_idx, log_p, time_tensor, value, window_logits, entropy

    _call = _inner(model)
    # Read .critic off unwrapped model (DDP doesn't forward attribute access).
    _has_critic = bool(getattr(_call, 'critic', False))
    # Training routes through DDP-wrapped model for gradient reduction; eval is unwrapped.
    _parallel_select = (training and _ROUTE_SELECTOR == 'legacy_parallel')
    _fwd = model if training else _call
    if training:
        _out = _fwd(
            batch, n_uav_i, n_adr_i, greedy=greedy, T=T,
            checkpoint_encoder=training, training=training,
            parallel_select=_parallel_select,
            initial_visited=initial_visited,
            start_node=start_node,
            start_time=start_time,
            start_battery=start_battery,
            start_capacity=start_capacity,
            return_entropy=True,
        )
        if _has_critic:
            tour_idx, log_p, time_tensor, entropy, value = _out
        else:
            tour_idx, log_p, time_tensor, entropy = _out
            value = None
    else:
        _out = _fwd(
            batch, n_uav_i, n_adr_i, greedy=greedy, T=T,
            checkpoint_encoder=training, training=training,
            parallel_select=_parallel_select,
            initial_visited=initial_visited,
            start_node=start_node,
            start_time=start_time,
            start_battery=start_battery,
            start_capacity=start_capacity,
        )
        if _has_critic:
            tour_idx, log_p, time_tensor, value = _out
        else:
            tour_idx, log_p, time_tensor = _out
            value = None
        entropy = None
    return tour_idx, log_p, time_tensor, value, None, entropy

class _SlicedLoader:
    def __init__(self, loader):
        self._loader = loader

    def __iter__(self):
        for batch in self._loader:
            yield _prep_batch(batch)

    def __len__(self):
        return len(self._loader)

def _ordered_range(bounds, lo_floor=1):
    lo, hi = int(bounds[0]), int(bounds[1])
    lo = max(lo_floor, lo)
    hi = max(lo, hi)
    return lo, hi

def _sample_generalized_batch(batch_size_i: int, epoch: int, batch_idx: int):
    seed = 3_100_000 + RANK * 100_000 + epoch * 10_000 + batch_idx
    rng = np.random.default_rng(seed)
    n_req_lo, n_req_hi = _ordered_range(_N_REQ_RANGE)
    n_uav_lo, n_uav_hi = _ordered_range(_N_UAV_RANGE)
    n_adr_lo, n_adr_hi = _ordered_range(_N_ADR_RANGE)
    n_req_i = int(rng.integers(n_req_lo, n_req_hi + 1))
    n_uav_i = int(rng.integers(n_uav_lo, n_uav_hi + 1))
    n_adr_i = int(rng.integers(n_adr_lo, n_adr_hi + 1))
    n_depots_uav_i = max(1, min(3, (n_uav_i + 3) // 4))
    n_depots_adr_i = max(1, min(3, (n_adr_i + 3) // 4))
    n_depots_i = n_depots_uav_i + n_depots_adr_i

    window_choices = np.array([1.0, 2.0, 5.0, 10.0, 15.0, 30.0, 60.0], dtype=np.float32)
    valid_choices = window_choices[(window_choices >= _WIN_MIN) & (window_choices <= _WIN_MAX)]
    if len(valid_choices) == 0:
        valid_choices = np.array([float(np.clip(5.0, _WIN_MIN, _WIN_MAX))], dtype=np.float32)
    window_length = float(rng.choice(valid_choices))

    pressure = float(n_req_i / max(window_length, 1.0))
    hard_prob = float(np.clip(0.15 + pressure / 20.0, 0.15, 0.85))
    tw_mean = float(np.clip(35.0 - 4.0 * pressure, 8.0, 35.0))
    tw_lo = float(np.clip(0.5 * tw_mean, 4.0, 20.0))
    tw_hi = float(np.clip(2.0 * tw_mean, tw_lo + 4.0, 80.0))
    availability_lo = max(0.05, min(_AVAIL_RANGE[0], _AVAIL_RANGE[1]))
    availability_hi = max(availability_lo, max(_AVAIL_RANGE[0], _AVAIL_RANGE[1]))
    availability_frac = float(rng.uniform(availability_lo, availability_hi))

    loader = creat_data(
        n_req_i, n_uav_i, n_adr_i, n_depots_uav_i, n_depots_adr_i,
        batch_size_i,
        batch_size=batch_size_i,
        num_workers=0,
        use_cache=False,
        streaming=True,
        seed=seed,
        shuffle=False,
        tw_slack_mean=tw_mean,
        tw_slack_std=max(2.0, tw_mean * 0.2),
        tw_slack_clip=(tw_lo, tw_hi),
        mix_hard_frac=hard_prob,
        mode_bias_frac=_MODE_BIAS,
        map_pool_size=_MAP_POOL_SIZE,
    )
    raw = next(iter(loader))
    batch = _prep_batch({k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in raw.items()})

    t_current = float(rng.uniform(5.0, 115.0))
    committed_frac = float(np.clip(1.0 - availability_frac, 0.0, 0.95))
    residual = _RESIDUAL and _DYNAMIC
    if residual:
        (batch, initial_visited, start_node, start_time, start_battery, start_capacity,
         n_vis_i) = build_rolling_residual_batch(
            batch, n_req_i, n_depots_i, n_uav_i, n_adr_i,
            t_current=t_current,
            committed_frac=committed_frac,
            dynamic_features=True,
            n_depots_uav=n_depots_uav_i,
            rng=rng,
        )
        n_vis_i = n_vis_i.float().clamp(min=1).to(device)
    elif _DYNAMIC:
        batch, initial_visited, n_vis_i = build_dynamic_masked_batch(
            batch, n_req_i, n_depots_i, t_current=t_current, dynamic_features=True,
        )
        n_vis_i = n_vis_i.float().clamp(min=1).to(device)
        start_node = start_time = start_battery = start_capacity = None
    else:
        initial_visited = None
        start_node = start_time = start_battery = start_capacity = None
        n_vis_i = float(n_req_i)
        t_current = 0.0

    batch = attach_window_context(
        batch, initial_visited, n_vis_i, n_req_i, n_uav_i, n_adr_i,
        t_current=t_current, window_length=window_length,
    )
    meta = dict(
        n_req=n_req_i,
        n_uav=n_uav_i,
        n_adr=n_adr_i,
        n_depots=n_depots_i,
        n_depots_uav=n_depots_uav_i,
        steps=n_depots_i + 2 * n_req_i,
        window_length=window_length,
        t_current=t_current,
        availability_frac=availability_frac,
    )
    state = (initial_visited, start_node, start_time, start_battery, start_capacity, n_vis_i)
    return batch, meta, state

# t=120 (end-of-shift, full-reveal) is the breakdown snapshot so val_complete_rate
# reflects true full-shift completion (earlier snapshots missed late arrivals).
_VAL_T_SNAPSHOTS = [80.0, 110.0, 120.0]
_VAL_T_BREAKDOWN = 120.0

def rollout(model: nn.Module, dataset, breakdown: bool = False,
            t_current: float = _VAL_T_BREAKDOWN, residual: bool = False,
            hard_mode: bool = False):
    """Validation decode. hard_mode=True forces complete_episode so failures are penalized."""
    if residual:
        raise NotImplementedError('Synthetic locked-vehicle prefixes are not complete-event mandatory validation')
    _inner(model).eval()
    all_cost, all_travel, all_tw, all_undeliv, all_dc, all_battery = [], [], [], [], [], []
    all_complete_mask: List[torch.Tensor] = []
    _HARD_FAIL_PENALTY = 100.0

    for batch_i, b in enumerate(dataset):
        batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in b.items()}
        batch = with_mandatory_contract(batch)
        initial_visited_val = None
        _n_vis_val = float(n_req)
        if residual:
            window_length = min(_WIN_MAX, max(_WIN_MIN, _VAL_T_BREAKDOWN / 4.0))
            torch_state = torch.random.get_rng_state()
            torch.manual_seed(90_000 + int(t_current * 10) + batch_i)
            rng_res = np.random.default_rng(90_000 + int(t_current * 10) + batch_i)
            try:
                (batch, initial_visited_val,
                 _start_node, _start_time, _start_battery, _start_capacity,
                 _n_vis_val) = build_rolling_residual_batch(
                    batch, n_req, n_depots, n_uav, n_adr,
                    t_current=t_current, dynamic_features=True,
                    n_depots_uav=n_depots_uav, rng=rng_res)
            finally:
                torch.random.set_rng_state(torch_state)
            _n_vis_val = _n_vis_val.float().clamp(min=1).to(device)
        elif _DYNAMIC:
            window_length = min(_WIN_MAX, max(_WIN_MIN, _VAL_T_BREAKDOWN / 4.0))
            batch, initial_visited_val, _n_vis_val = build_dynamic_masked_batch(
                batch, n_req, n_depots, t_current=t_current, dynamic_features=True)
            _n_vis_val = _n_vis_val.float().clamp(min=1).to(device)  # [B] float
        else:
            window_length = _WIN_MAX
        batch = attach_window_context(
            batch, initial_visited_val, _n_vis_val, n_req, n_uav, n_adr,
            t_current=t_current, window_length=window_length,
        )
        batch = dict(batch)
        batch['route_scope'] = 'complete_episode' if hard_mode else 'rolling_prefix'
        _hard_failed_batch = False
        with torch.no_grad():
            try:
                tour_idx, _, time_tensor, _, _, _ = _policy_forward(
                    model, batch, greedy=True, initial_visited=initial_visited_val,
                )
            except IncompleteTensorRouteError:
                # One or more instances couldn't produce a complete episode;
                # apply a deterministic batch-level penalty.
                _hard_failed_batch = True
                batch_size = int(batch['time_window'].shape[0])
                _fake_cost = torch.full((batch_size,), _HARD_FAIL_PENALTY)
                all_cost.append(_fake_cost)
                all_complete_mask.append(torch.zeros(batch_size))
                if breakdown:
                    _nan = torch.full((batch_size,), float('nan'))
                    all_travel.append(_nan.clone())
                    all_tw.append(_nan.clone())
                    all_undeliv.append(_nan.clone())
                    all_dc.append(_nan.clone())
                    all_battery.append(_nan.clone())
                continue
            if breakdown:
                rew, comps = reward1(
                    batch['time_window'], tour_idx.detach(),
                    batch['edge_attr_d'], batch['edge_attr_r'],
                    time_tensor, n_uav,
                    num_depots=n_depots,
                    return_breakdown=True,
                    initial_visited=initial_visited_val,
                    available_mask=batch.get('available_mask', None),
                    physics_batch=batch,
                    allow_partial=not hard_mode,
                )
                all_travel.append((comps['travel'].sum(dim=1) / _n_vis_val).cpu())
                all_tw.append((comps['tw_penalty'].sum(dim=1) / _n_vis_val).cpu())
                all_undeliv.append((comps['undeliv'].sum(dim=1) / _n_vis_val).cpu())
                all_dc.append((comps['delivery_credit'].sum(dim=1) / _n_vis_val).cpu())
                all_battery.append((comps['battery'].sum(dim=1) / _n_vis_val).cpu())
            else:
                rew, comps = reward1(
                    batch['time_window'], tour_idx.detach(),
                    batch['edge_attr_d'], batch['edge_attr_r'],
                    time_tensor, n_uav,
                    num_depots=n_depots,
                    initial_visited=initial_visited_val,
                    available_mask=batch.get('available_mask', None),
                    physics_batch=batch,
                    allow_partial=not hard_mode, return_breakdown=True,
                )
            all_cost.append((rew.sum(dim=1) / _n_vis_val).cpu())
            _ua = comps.get('unassigned_count') if isinstance(comps, dict) else None
            if _ua is not None and torch.is_tensor(_ua):
                all_complete_mask.append((_ua == 0).float().cpu())

    costs = torch.cat(all_cost)
    # Expose complete_rate as a side-channel attribute on the costs tensor.
    if all_complete_mask:
        complete = torch.cat(all_complete_mask)
        costs._complete_rate = float(complete.mean().item())  # type: ignore[attr-defined]
    else:
        costs._complete_rate = float('nan')  # type: ignore[attr-defined]
    if not breakdown:
        return costs
    return costs, torch.cat(all_travel), torch.cat(all_tw), torch.cat(all_undeliv), torch.cat(all_dc), torch.cat(all_battery)

def save_checkpoint(epoch, model, optimizer, scheduler, folder, costs, critic=None):
    if RANK != 0:
        return
    torch.save({
        'epoch':               epoch,
        'model_state_dict':    _inner(model).state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict(),
        'critic_state_dict':    critic.state_dict() if critic is not None else None,
        'costs':               costs,
        'rng_state':           torch.get_rng_state(),
        'route_contract':      route_checkpoint_contract(),
    }, os.path.join(folder, 'checkpoint.pth'))
    print(f'Checkpoint saved — epoch {epoch}', flush=True)

def load_checkpoint(model, optimizer, scheduler, path, critic=None):
    ckpt = load_tensor_file(path, map_location=device, weights_only=True)
    validate_route_checkpoint(ckpt)
    _inner(model).load_state_dict(ckpt['model_state_dict'])
    try:
        optimizer.load_state_dict(ckpt['optimizer_state_dict'])
        scheduler.load_state_dict(ckpt['scheduler_state_dict'])
    except ValueError:
        if RANK == 0:
            print('[checkpoint] Optimizer state incompatible with current parameter set; '
                  'model weights restored, optimizer restarted.', flush=True)
    if critic is not None and ckpt.get('critic_state_dict') is not None:
        critic.load_state_dict(ckpt['critic_state_dict'])
    if ckpt.get('rng_state') is not None and not _is_directml:
        torch.set_rng_state(ckpt['rng_state'].cpu().byte())
    return ckpt['epoch'] + 1, ckpt['costs']

def train():
    if _RESIDUAL or _REPLAN_GATE:
        raise NotImplementedError('Supported mandatory training: route (default) or coop. '
                                  'Synthetic residual and gate training still need complete-event rollouts.')
    from release_checks import require_training_release
    require_training_release(
        allow_unvalidated=_cli_args.allow_unvalidated_model,
        acknowledge_stylized_scope=getattr(_cli_args, 'acknowledge_stylized_scope', False))
    params = OrderedDict(
        lr=[_BASE_LR],
        hidden_node_dim=[128],
        hidden_edge_dim=[16],
        entropy_coef=[_ENTROPY_COEF],
    )
    Run  = namedtuple('Run', params.keys())
    runs = [Run(*v) for v in product(*params.values())]

    _cfg = dict(RUNG_CONFIG[RUNG])
    _shadow_val_size = _SHADOW_VAL_SIZE
    if _QUICK_MAX_EPOCHS is not None:
        _cfg['train_size'] = 12800
        _cfg['val_size']   = 1280
        _cfg['batch_size'] = 64
        _cfg['streaming']  = False
        _shadow_val_size = min(_shadow_val_size, 64)

    batch_size  = _cfg['batch_size']
    train_size  = _cfg['train_size']
    val_size    = _cfg['val_size']
    streaming   = _cfg['streaming']
    grad_accum  = _cfg.get('grad_accum', 1)
    conv_layers = _cfg.get('conv_layers', 3)

    # Diagnostic modes force single-process loading to avoid worker CUDA re-import.
    _diagnostic_force_zero = _is_directml or not torch.cuda.is_available() or _SMOKE or _OVERFIT
    if _CLI_NUM_WORKERS is not None:
        n = max(0, int(_CLI_NUM_WORKERS))
        train_workers = n
        val_workers = min(n, 4) if n > 0 else 0
    elif _diagnostic_force_zero:
        train_workers = val_workers = 0
    else:
        _total_cpus   = int(os.environ.get('SLURM_CPUS_PER_TASK', 8))
        _cpus_per_rank = max(1, _total_cpus // max(WORLD_SIZE, 1))
        train_workers = min(8, max(2, _cpus_per_rank - 1))
        val_workers   = min(4, max(1, _cpus_per_rank // 3))

    _run_tag = f'{RUNG}-{_VARIANT}' if _VARIANT else RUNG
    folder = f'CE-PDPTW-{_run_tag}-HetGAT'
    if RANK == 0:
        os.makedirs(folder, exist_ok=True)
        os.makedirs('logs', exist_ok=True)
    if IS_DIST:
        dist.barrier()

    csv_log_path = os.path.join('logs', f'training_{_run_tag}.csv')
    _csv_file = None
    _csv_writer = None
    _tb_writer = None
    if RANK == 0:
        _csv_fields = [
            'epoch', 'rung', 'arch', 'rloo_k', 'entropy_coef', 'lr_decay',
            'train_reward_mean', 'train_loss_mean', 'critic_loss_mean',
            'val_cost', 'val_cost_std',
            'val_travel_cost', 'val_tw_penalty', 'val_undeliv_cost', 'val_deliv_credit', 'val_battery_cost',
            'val_hard_cost', 'val_hard_tw_penalty',
            'val_residual_cost', 'val_residual_tw_penalty',
            'val_select_cost',
            # Fraction of validation instances that would deliver ALL requests
            # under a true mandatory-completion check. Added after
            # Tier 4 revealed that val_cost (partial-prefix scoring) can
            # improve while the policy still can't complete real routes.
            'val_complete_rate',
            'grad_norm_mean', 'grad_norm_max',
            'entropy_mean',
            'baseline_updated', 'mode_uav_frac',
            'uav_demand_ratio', 'uav_dist_ratio',
            'lr', 'elapsed_s',
            # Mitigation telemetry (see
            # tools/diagnostics/mitigation_telemetry.py). These answer
            # "is this mitigation actually contributing?", not "did
            # val_complete_rate go up when I enabled it". All default to
            # 0.0 when the relevant mitigation is off.
            'critic_corr', 'dyn_grad_l2', 'depot_visit_frac',
            # Reward-factor inspector (Probe 1). Per-term means
            # across the epoch's training batches. Reveal which cost term
            # dominates the gradient. All in reward_fw's units (whatever
            # per-vehicle, per-step normalisation the trainer uses).
            'train_travel_mean', 'train_pickup_early_mean',
            'train_pickup_late_mean', 'train_delivery_late_mean',
            'train_tw_total_mean', 'train_shaping_mean',
            'train_battery_slack_mean',
            # Oracle-ratio penalty mean.
            # > 0  → policy worse than oracle (loss added)
            # == 0 → tied with oracle (neutral)
            # < 0  → policy beat oracle (bonus → amplified gradient)
            'train_oracle_excess_mean',
            # Paper-strict depot-mask + encoder-attn-mask diagnostics.
            'dyn_depot_fire_frac',   # fraction of agent-decode-slots where overlay fired
            'depot_pick_mean_tour',  # mean tour-step at which depot was picked
            'legal_cust_at_depot',   # mean # legal customer options when depot picked
        ]
        _csv_file = open(csv_log_path, 'w', newline='')
        _csv_writer = csv.DictWriter(_csv_file, fieldnames=_csv_fields)
        _csv_writer.writeheader()
        if _TB_AVAILABLE:
            _tb_writer = SummaryWriter(log_dir=os.path.join('logs', f'tb_{RUNG}'))

    for lr, h_node, h_edge, ent_coef in runs:
        effective_lr = lr * np.sqrt(WORLD_SIZE) if IS_DIST else lr

        if RANK == 0:
            print(f'\nConfig: lr={lr} (eff={effective_lr:.2e}), bs={batch_size}×{WORLD_SIZE}, '
                  f'h={h_node}, conv={conv_layers}, rung={RUNG}', flush=True)

        cache_train = os.path.join(CACHE_DIR, f'train_{RUNG}.pkl')
        cache_valid = os.path.join(CACHE_DIR, f'valid_{RUNG}.pkl')

        train_loader = None
        if not _GENERALIZED:
            train_loader = creat_data(
                n_req, n_uav, n_adr, n_depots_uav, n_depots_adr,
                train_size,
                batch_size=batch_size,
                streaming=streaming,
                use_cache=(not streaming),
                cache_file=cache_train,
                seed=42,
                num_workers=train_workers,
                distributed=IS_DIST,
                rank=RANK,
                world_size=WORLD_SIZE,
                mix_hard_frac=_MIX_HARD_FRAC if streaming else 0.0,
                mode_bias_frac=_MODE_BIAS if streaming else 0.0,
                map_pool_size=_MAP_POOL_SIZE if streaming else 0,
            )

        valid_loader = creat_data(
            n_req, n_uav, n_adr, n_depots_uav, n_depots_adr,
            val_size,
            batch_size=batch_size,
            streaming=False,
            use_cache=True,
            cache_file=cache_valid,
            seed=1000,
            shuffle=False,
            num_workers=val_workers,
            distributed=False,
        )

        # Shadow val: one small non-overlapping dataset per seed (primary val stays separate).
        _val_loaders_ms = []
        for _vs in _VAL_SEEDS:
            _shadow_vl = creat_data(
                n_req, n_uav, n_adr, n_depots_uav, n_depots_adr,
                _shadow_val_size,
                batch_size=batch_size,
                streaming=False,
                use_cache=True,
                cache_file=os.path.join(CACHE_DIR, f'shadow_{RUNG}_s{_vs}_n{_shadow_val_size}.pkl'),
                seed=_vs,
                shuffle=False,
                num_workers=val_workers,
                distributed=False,
            )
            _val_loaders_ms.append(_SlicedLoader(_shadow_vl))

        _val_loaders_hard = []
        for _hs in _HARD_VAL_SEEDS:
            _vl_h = _SlicedLoader(creat_data(
                n_req, n_uav, n_adr, n_depots_uav, n_depots_adr,
                _shadow_val_size,
                batch_size=batch_size,
                streaming=False,
                use_cache=True,
                cache_file=os.path.join(CACHE_DIR, f'shadow_{RUNG}_hard_s{_hs}_n{_shadow_val_size}.pkl'),
                seed=_hs,
                shuffle=False,
                num_workers=val_workers,
                distributed=False,
                tw_slack_mean=10.0,
                tw_slack_std=2.0,
                tw_slack_clip=(5.0, 20.0),
                temporal_peaks=True,
                spatial_cluster_k=3,
                demand_heavy_tail=True,
            ))
            _val_loaders_hard.append(_vl_h)

        if RANK == 0:
            print('Data ready.', flush=True)

        _overfit_batch = None
        _overfit_iv    = None   # initial_visited for the fixed overfit batch
        _overfit_n_vis = None   # [B] revealed-order counts (tensor when _DYNAMIC)
        if _OVERFIT:
            if _ORACLE_DB is not None:
                # Build overfit batch from oracle instances, attach oracle_cost.
                from tools.diagnostics.rl_backend_conformance import _bank_instance_to_tensor as _b2t

                _bank_cache = {}
                _inst_dicts = []
                _oracle_costs = []
                for _sha, _rec in _ORACLE_DB.items():
                    _bp = _rec['bank_path']
                    if _bp not in _bank_cache:
                        _bank_cache[_bp] = load_tensor_file(_bp, map_location='cpu',
                                                       weights_only=True)
                    _bank = _bank_cache[_bp]
                    _idx = int(_rec['instance_idx'])
                    _inst_dicts.append(_b2t(_bank, _idx))
                    _oracle_costs.append(float(_rec['cost']))

                if not _inst_dicts:
                    raise RuntimeError('oracle JSON produced zero usable instances')

                _first = _inst_dicts[0]
                _oracle_n_req = int(_first['n_req'])
                if _oracle_n_req != int(n_req):
                    raise ValueError(
                        f'Oracle n_req={_oracle_n_req} but trainer RUNG {RUNG} has '
                        f'n_req={n_req}. Rerun with --rung matching oracle.')

                def _stack_or_keep(_values):
                    import torch as _t
                    if all(isinstance(v, _t.Tensor) for v in _values):
                        try:
                            return _t.stack(_values, dim=0)
                        except RuntimeError:
                            return _values
                    return _values[0]

                _batch_keys = set().union(*[d.keys() for d in _inst_dicts])
                _overfit_batch = {}
                for _k in _batch_keys:
                    if all(_k in d for d in _inst_dicts):
                        _overfit_batch[_k] = _stack_or_keep([d[_k] for d in _inst_dicts])
                    else:
                        _overfit_batch[_k] = _inst_dicts[0][_k]

                _overfit_batch = {k: (v.to(device) if isinstance(v, torch.Tensor) else v)
                                  for k, v in _overfit_batch.items()}
                _overfit_batch['oracle_cost'] = torch.tensor(
                    _oracle_costs, dtype=torch.float32, device=device)
                _overfit_batch = _prep_batch(_overfit_batch)
                if RANK == 0:
                    print(f'[oracle] overfit batch built from {len(_inst_dicts)} '
                          f'oracle instances. oracle_costs={_oracle_costs}')
            else:
                _raw = next(iter(train_loader))
                _overfit_batch = _prep_batch({k: v.to(device) if isinstance(v, torch.Tensor) else v
                                              for k, v in _raw.items()})
            if _DYNAMIC:
                # Apply dynamic masking once at t=60 for reproducibility.
                _overfit_batch, _overfit_iv, _ovf_vis = build_dynamic_masked_batch(
                    _overfit_batch, n_req, n_depots, t_current=60.0, dynamic_features=True)
                _overfit_n_vis = _ovf_vis.float().clamp(min=1).to(device)
            if RANK == 0:
                print(
                    f'[OVERFIT] Batch fixed: {_overfit_batch["x"].shape[0]} instances '
                    f'(feat_dim={_overfit_batch["x"].shape[-1]}). '
                    f'train_reward_mean should decrease each epoch if architecture works.',
                    flush=True,
                )

        if _POLICY == 'coop':
            actor = CoopActorCritic(
                input_node_dim=15 if _DYNAMIC else 11,
                hidden_dim=h_node,
                n_heads=4,
                n_layers=2,
            ).to(device)
        else:
            if _ARCH == 'koolam':
                from baselines import KoolAMModel
                actor = KoolAMModel(
                    input_node_dim=15 if _DYNAMIC else 11,
                    hidden_node_dim=h_node,
                    input_edge_dim=_EDGE_DIM,
                    hidden_edge_dim=h_edge,
                    conv_layers=conv_layers,
                    edge_features=_EDGE_FEATURES,
                    decoder_biases=_DECODER_BIASES,
                ).to(device)
            elif _ARCH == 'hetero_attn_pdp':
                from baselines import HAPDPModel
                actor = HAPDPModel(
                    input_node_dim=15 if _DYNAMIC else 11,
                    hidden_node_dim=h_node,
                    input_edge_dim=_EDGE_DIM,
                    hidden_edge_dim=h_edge,
                    conv_layers=conv_layers,
                    edge_features=_EDGE_FEATURES,
                    decoder_biases=_DECODER_BIASES,
                ).to(device)
            elif _ARCH == 'gann':
                from baselines import GANNModel
                actor = GANNModel(
                    input_node_dim=15 if _DYNAMIC else 11,
                    hidden_node_dim=h_node,
                    input_edge_dim=_EDGE_DIM,
                    hidden_edge_dim=h_edge,
                    conv_layers=conv_layers,
                    edge_features=_EDGE_FEATURES,
                    decoder_biases=_DECODER_BIASES,
                ).to(device)
            else:
                actor = Model(
                    input_node_dim=15 if _DYNAMIC else 11,
                    hidden_node_dim=h_node,
                    input_edge_dim=_EDGE_DIM,
                    hidden_edge_dim=h_edge,
                    conv_layers=conv_layers,
                    arch=_ARCH,
                    edge_features=_EDGE_FEATURES,
                    decoder_biases=_DECODER_BIASES,
                    margin_bias=_MARGIN_BIAS,
                    critic=_CRITIC,
                    dyn_features=_DYN_FEATURES,
                    battery_mask=_BATTERY_MASK,
                    battery_mask_reserve_frac=_BATTERY_MASK_RESERVE_FRAC,
                    battery_mask_reserve_frac_uav=_BATTERY_MASK_RESERVE_FRAC_UAV,
                    battery_mask_reserve_frac_adr=_BATTERY_MASK_RESERVE_FRAC_ADR,
                    battery_mask_lookahead=_BATTERY_MASK_LOOKAHEAD,
                    battery_mandatory_recharge=_BATTERY_MANDATORY_RECHARGE,
                    battery_mandatory_recharge_buffer=_BATTERY_MANDATORY_RECHARGE_BUFFER,
                    paper_strict_depot_mask=_PAPER_STRICT_DEPOT_MASK,
                    paper_strict_depot_margin=_PAPER_STRICT_DEPOT_MARGIN,
                    encoder_attn_mask=_ENCODER_ATTN_MASK,
                    use_chain_lookahead_in_mask=_CHAIN_LOOKAHEAD_IN_MASK,
                ).to(device)
        # Set continuation_check on both Model and decoder (decoder reads self.).
        actor.continuation_check = _CONTINUATION_CHECK
        if hasattr(actor, 'decoder'):
            actor.decoder.continuation_check = _CONTINUATION_CHECK
        if RANK == 0:
            print(f'[mask] continuation_check = {_CONTINUATION_CHECK}', flush=True)
            print(f'[selector] route_selector = {_ROUTE_SELECTOR} '
                  f'(training path; eval always sequential-greedy)', flush=True)
        critic = (
            RollingStateCritic(
                input_node_dim=15 if _DYNAMIC else 11,
                hidden_dim=h_node,
            ).to(device)
            if _USE_CRITIC else None
        )

        if IS_DIST:
            actor = DDP(actor, device_ids=[LOCAL_RANK], output_device=LOCAL_RANK,
                        find_unused_parameters=False)
            if RANK == 0:
                print(f'DDP active: {WORLD_SIZE} ranks.', flush=True)
        elif torch.cuda.device_count() > 1 and not _is_directml:
            actor = nn.DataParallel(actor)
            if RANK == 0:
                print(f'DataParallel on {torch.cuda.device_count()} GPUs.', flush=True)

        def _bl_transform_fn(batch):
            masked, iv, _ = build_dynamic_masked_batch(
                batch, n_req, n_depots, t_current=_VAL_T_BREAKDOWN, dynamic_features=True)
            return masked, iv

        baseline = None
        if _POLICY == 'route':
            baseline = RolloutBaseline(
                _inner(actor), _SlicedLoader(valid_loader),
                n_uav=n_uav, n_adr=n_adr, n_nodes=steps,
                transform_fn=_bl_transform_fn if _DYNAMIC else None,
                delivery_shaping_k=_DELIVERY_SHAPING_K,
                refresh_after_stale=_BL_REFRESH_STALE,
            )

        opt_params = list(actor.parameters())
        if critic is not None:
            opt_params += list(critic.parameters())
        optimizer = optim.Adam(opt_params, lr=effective_lr)
        scheduler = LambdaLR(
            optimizer,
            lr_lambda=lambda ep: (
                min(1.0, (ep + 1) / _WARMUP_EPOCHS) * (_LR_DECAY ** max(0, ep - _WARMUP_EPOCHS))
            ),
        )

        ckpt_path = os.path.join(folder, 'checkpoint.pth')
        if os.path.exists(ckpt_path):
            start_epoch, costs = load_checkpoint(actor, optimizer, scheduler, ckpt_path, critic=critic)
            if RANK == 0:
                print(f'Resuming from epoch {start_epoch}', flush=True)
        else:
            start_epoch, costs = 0, []
            rung_idx = RUNG_ORDER.index(RUNG)
            if _INIT_ACTOR:
                if not os.path.exists(_INIT_ACTOR):
                    raise FileNotFoundError(f'--init-actor path not found: {_INIT_ACTOR}')
                init_sd = load_tensor_file(_INIT_ACTOR, map_location=device, weights_only=True)
                validate_route_checkpoint(init_sd)
                _inner(actor).load_state_dict(init_sd['model_state_dict'], strict=False)
                if RANK == 0:
                    print(f'[init-actor] loaded weights from {_INIT_ACTOR}', flush=True)
            elif rung_idx > 0 and _WARMSTART:
                prev_rung = RUNG_ORDER[rung_idx - 1]
                prev_tag    = f'{prev_rung}-{_VARIANT}' if _VARIANT else prev_rung
                prev_folder = f'CE-PDPTW-{prev_tag}-HetGAT'
                prev_best   = os.path.join(prev_folder, 'best_actor.pt')
                prev_ckpt   = os.path.join(prev_folder, 'checkpoint.pth')
                if os.path.exists(prev_best):
                    prev_sd = load_tensor_file(prev_best, map_location=device, weights_only=True)
                    validate_route_checkpoint(prev_sd)
                    _inner(actor).load_state_dict(prev_sd['model_state_dict'], strict=False)
                    if RANK == 0:
                        print(f'[warmstart] Rung {prev_rung} best_actor.pt → Rung {RUNG}', flush=True)
                elif os.path.exists(prev_ckpt):
                    prev_sd = load_tensor_file(prev_ckpt, map_location=device, weights_only=True)
                    validate_route_checkpoint(prev_sd)
                    _inner(actor).load_state_dict(prev_sd['model_state_dict'], strict=False)
                    if RANK == 0:
                        print(f'[warmstart] Rung {prev_rung} checkpoint.pth → Rung {RUNG} (best_actor.pt not found)', flush=True)
                else:
                    if RANK == 0:
                        print(f'[warmstart] WARNING: {prev_folder} not found — starting from random init.', flush=True)
            elif not _WARMSTART and RANK == 0:
                print(f'[init] Rung {RUNG} starting from random init (use --warmstart for curriculum).', flush=True)
            if IS_DIST:
                for p in _inner(actor).parameters():
                    dist.broadcast(p.data, src=0)

        if _RLOO_K == 1:
            raise ValueError(
                '--rloo-k 1 is invalid: leave-one-out baseline requires K>=2 '
                '(K=1 divides by zero). Use --rloo-k 4.'
            )

        if _DYNAMIC and _RLOO_K == 0 and RANK == 0:
            print('[NOTE] --dynamic with rollout baseline is supported but RLOO is preferred.',
                  flush=True)

        _use_amp = torch.cuda.is_available() and not _is_directml

        max_epochs = _QUICK_MAX_EPOCHS if _QUICK_MAX_EPOCHS is not None else (_CLI_MAX_EPOCHS or 100)
        if _SMOKE and RANK == 0:
            print(
                f'\n[SMOKE] Pipeline check: Rung {RUNG}, {train_size} train, '
                f'{val_size} val, {max_epochs} epochs, device={device}',
                flush=True,
            )
        best_val_cost = min(costs) if costs else float('inf')
        for epoch in range(start_epoch, max_epochs):
            if RANK == 0:
                print(f'\nEpoch {epoch} -----------------------------------', flush=True)

            actor.train()
            # Linearly decay temperature from _T_START to _T_END over all epochs.
            # Default _T_END=1.0 reproduces the historical anneal; wider --temperature
            # with --temperature-end allows mitigation #3 (early exploration) without
            # changing the shape of the schedule.
            T_train = max(_T_END, _T_START - (_T_START - _T_END) * epoch / max(max_epochs - 1, 1))
            losses, rewards_log, grad_norms_log, critic_losses_log = [], [], [], []
            # Probe 1: per-term accumulators for reward-factor
            # inspection. Populated after reward1() returns _components.
            travel_log, pickup_early_log, pickup_late_log = [], [], []
            delivery_late_log, tw_total_log = [], []
            shaping_log, battery_slack_log = [], []
            oracle_excess_log = []   # Oracle-ratio telemetry
            entropies_log = []
            # Mitigation telemetry accumulators. Each list is per
            # batch (train loop); aggregated at epoch end into CSV columns.
            # See tools/diagnostics/mitigation_telemetry.py for the helpers.
            _telem_v_preds, _telem_returns = [], []
            _telem_dyn_grad_l2 = []
            _telem_depot_frac  = []
            # Paper-strict depot-mask + tour diagnostics accumulators
            # (per-batch fire fraction + depot-pick tour stats).
            _telem_dyn_depot_fire   = 0  # overlay-fire count
            _telem_dyn_depot_total  = 0  # total agent-decode slots
            _telem_depot_pick_tour_sum = 0
            _telem_depot_pick_count   = 0
            _telem_legal_cust_sum     = 0
            uav_actions_total, all_actions_total = 0, 0
            uav_demand_sum, uav_demand_count = 0.0, 0
            adr_demand_sum, adr_demand_count = 0.0, 0
            uav_dist_sum,   adr_dist_sum     = 0.0, 0.0
            _cached_bl_reward = None
            start = time.time()

            if (not _OVERFIT and train_loader is not None
                    and IS_DIST and hasattr(train_loader.sampler, 'set_epoch')):
                train_loader.sampler.set_epoch(epoch)

            _num_train_batches = max(1, int(np.ceil(train_size / batch_size)))
            _train_src = (
                [_overfit_batch] if _OVERFIT
                else range(_num_train_batches) if _GENERALIZED
                else train_loader
            )
            _train_len = len(_train_src)

            for batch_idx, batch in enumerate(_train_src):
                if _GENERALIZED:
                    batch, _meta, _state = _sample_generalized_batch(
                        batch_size, epoch, batch_idx,
                    )
                    _n_req_i = _meta['n_req']
                    _n_uav_i = _meta['n_uav']
                    _n_adr_i = _meta['n_adr']
                    _n_depots_i = _meta['n_depots']
                    _steps_i = _meta['steps']
                    (_initial_visited_train, _start_node, _start_time,
                     _start_battery, _start_capacity, _n_vis) = _state
                    _window_length = _meta['window_length']
                    _t_current = _meta['t_current']
                    _use_residual_batch = _initial_visited_train is not None
                else:
                    if not _OVERFIT:
                        batch = _prep_batch({k: v.to(device) if isinstance(v, torch.Tensor) else v
                                             for k, v in batch.items()})
                    _n_req_i = n_req
                    _n_uav_i = n_uav
                    _n_adr_i = n_adr
                    _n_depots_i = n_depots
                    _steps_i = steps
                    _initial_visited_train = None
                    _start_node = _start_time = _start_battery = _start_capacity = None
                    _window_length = float(torch.empty(1).uniform_(_WIN_MIN, _WIN_MAX).item())
                    _use_residual_batch = (
                        _RESIDUAL and _DYNAMIC and not _OVERFIT
                        and torch.rand((), device=device).item() < _RESIDUAL_FRAC
                    )
                    if _use_residual_batch:
                        _t_max_res = _GATE_T_MAX if _REPLAN_GATE else 100.0
                        _t_current = float(torch.empty(1).uniform_(15.0, _t_max_res).item())
                        _rng_res = np.random.default_rng()
                        (batch, _initial_visited_train,
                         _start_node, _start_time, _start_battery, _start_capacity,
                         _n_vis) = build_rolling_residual_batch(
                            batch, n_req, n_depots, n_uav, n_adr,
                            t_current=_t_current, dynamic_features=True,
                            n_depots_uav=n_depots_uav, rng=_rng_res)
                        _n_vis = _n_vis.float().clamp(min=1).to(device)
                    elif _DYNAMIC and not _OVERFIT:
                        _t_current = float(torch.empty(1).uniform_(10.0, 110.0).item())
                        batch, _initial_visited_train, _n_vis = build_dynamic_masked_batch(
                            batch, n_req, n_depots, t_current=_t_current, dynamic_features=True)
                        _n_vis = _n_vis.float().clamp(min=1).to(device)  # [B] float
                    elif _OVERFIT and _DYNAMIC:
                        # Overfit batch was pre-masked at init; reuse stored IV and n_vis.
                        _initial_visited_train = _overfit_iv
                        _n_vis = _overfit_n_vis
                        _window_length = 5.0
                        _t_current = 60.0
                    else:
                        _n_vis = float(n_req)
                        _t_current = 0.0

                    batch = attach_window_context(
                        batch, _initial_visited_train, _n_vis, n_req, n_uav, n_adr,
                        t_current=_t_current, window_length=_window_length,
                    )

                if batch_idx % grad_accum == 0:
                    optimizer.zero_grad(set_to_none=True)

                _amp_cm = (torch.amp.autocast(device_type='cuda', dtype=torch.bfloat16)
                           if _use_amp else contextlib.nullcontext())

                _K = _RLOO_K if (_RLOO_K > 0 and _POLICY != 'coop') else 1
                _B = batch['x'].shape[0]
                _batch_fw = (
                    {k: v.repeat_interleave(_K, dim=0) if isinstance(v, torch.Tensor) else v
                     for k, v in batch.items()}
                    if _K > 1 else batch
                )
                def _rloo_expand(t):
                    return (t.repeat_interleave(_K, dim=0)
                            if _K > 1 and isinstance(t, torch.Tensor) else t)

                _iv_fw          = _rloo_expand(_initial_visited_train)
                _start_node_fw  = _rloo_expand(_start_node)
                _start_time_fw  = _rloo_expand(_start_time)
                _start_bat_fw   = _rloo_expand(_start_battery)
                _start_cap_fw   = _rloo_expand(_start_capacity)

                with _amp_cm:
                    tour_idx_fw, tour_logp_fw, time_tensor_fw, value_fw_model, window_logits_fw, entropy_fw = _policy_forward(
                        actor, _batch_fw,
                        greedy=False, T=T_train, training=True,
                        initial_visited=_iv_fw,
                        start_node=_start_node_fw,
                        start_time=_start_time_fw,
                        start_battery=_start_bat_fw,
                        start_capacity=_start_cap_fw,
                        n_uav_i=_n_uav_i,
                        n_adr_i=_n_adr_i,
                    )
                # AMP may produce bfloat16 activations; cast to float32 so that
                # loss computation and backward use a consistent dtype.
                tour_logp_fw = tour_logp_fw.float()
                if value_fw_model is not None:
                    value_fw_model = value_fw_model.float()
                if batch_idx == 0 and epoch == 0 and RANK == 0:
                    _actor_inner = _inner(actor)
                    print(f'[E3-diag] inner.critic={getattr(_actor_inner, "critic", "ATTR-MISSING")} '
                          f'outer.critic={getattr(actor, "critic", "ATTR-MISSING")} '
                          f'value_fw_model={None if value_fw_model is None else list(value_fw_model.shape)}',
                          flush=True)

                _n_vis_fw = (
                    _n_vis.repeat_interleave(_K)
                    if _K > 1 and isinstance(_n_vis, torch.Tensor)
                    else _n_vis
                )
                # Partial-prefix reward + PBRS k*delivered; policy-invariant for complete
                # episodes (Ng/Harada/Russell 1999). Reported scoring never sees this shift.
                _prefix_cost, _components = reward1(
                    _batch_fw['time_window'], tour_idx_fw.detach(),
                    _batch_fw['edge_attr_d'], _batch_fw['edge_attr_r'],
                    time_tensor_fw, _n_uav_i,
                    num_depots=_n_depots_i,
                    start_time=_start_time_fw,
                    initial_visited=_iv_fw,
                    available_mask=_batch_fw.get('available_mask', None),
                    physics_batch=_batch_fw,
                    start_battery=_start_bat_fw,
                    start_capacity=_start_cap_fw,
                    allow_partial=True, return_breakdown=True,
                )
                reward_fw = _prefix_cost.sum(dim=1) / _n_vis_fw
                try:
                    _travel_batch = _components['travel'].sum(dim=1).mean().item()
                    _pe_batch     = _components['pickup_early_cost'].sum(dim=1).mean().item()
                    _pl_batch     = _components['pickup_late_cost'].sum(dim=1).mean().item()
                    _dl_batch     = _components['delivery_late_cost'].sum(dim=1).mean().item()
                    _tw_batch     = _components['tw_penalty'].sum(dim=1).mean().item()
                    travel_log.append(float(_travel_batch))
                    pickup_early_log.append(float(_pe_batch))
                    pickup_late_log.append(float(_pl_batch))
                    delivery_late_log.append(float(_dl_batch))
                    tw_total_log.append(float(_tw_batch))
                except (KeyError, AttributeError):
                    pass
                _shaping_delta = 0.0
                _battery_slack_delta = 0.0
                if _DELIVERY_SHAPING_K > 0.0:
                    # Credit delivered-from-ACTIVE (not from-ALL) for correct shaping delta.
                    _am_fw = _batch_fw.get('active_request_mask', None)
                    if _am_fw is not None:
                        _active_count_fw = torch.as_tensor(
                            _am_fw, device=reward_fw.device
                        ).bool().reshape(reward_fw.shape[0], -1).sum(dim=1).to(reward_fw.dtype)
                    else:
                        _active_count_fw = reward_fw.new_full(
                            (reward_fw.shape[0],), float(n_req)
                        )
                    _delivered = (_active_count_fw - _components['unassigned_count']).to(reward_fw.dtype)
                    if _DELIVERY_SHAPING_MODE == 'pbrs_step':
                        # Telescopes to +k*delivered under gamma=1 whole-episode REINFORCE.
                        _shaping_delta = (_DELIVERY_SHAPING_K * _delivered / _n_vis_fw).mean().item()
                        reward_fw = reward_fw - _DELIVERY_SHAPING_K * _delivered / _n_vis_fw
                    else:
                        _shaping_delta = (_DELIVERY_SHAPING_K * _delivered / _n_vis_fw).mean().item()
                        reward_fw = reward_fw - _DELIVERY_SHAPING_K * _delivered / _n_vis_fw
                shaping_log.append(float(_shaping_delta))

                if _BATTERY_SLACK_K > 0.0:
                    # Mid-shift recharge signal; step 0 is the forced initial assignment.
                    _tour_fw_det = tour_idx_fw.detach()
                    if _tour_fw_det.dim() == 3 and _tour_fw_det.size(-1) > 1:
                        _depot_visits = (_tour_fw_det[:, :, 1:] < _n_depots_i).float().sum(dim=(-1, -2))
                        _battery_slack_delta = (_BATTERY_SLACK_K * _depot_visits / _n_vis_fw).mean().item()
                        reward_fw = reward_fw - _BATTERY_SLACK_K * _depot_visits / _n_vis_fw
                battery_slack_log.append(float(_battery_slack_delta))

                # Oracle-ratio reward: add k * clamp((rl_cost - oracle_cost) / oracle_cost).
                _oracle_excess_mean = 0.0
                if _ORACLE_DB is not None and _ORACLE_REWARD_K > 0.0:
                    _oc = _batch_fw.get('oracle_cost', None)
                    if _oc is not None:
                        _oc_t = torch.as_tensor(_oc, device=reward_fw.device,
                                                 dtype=reward_fw.dtype)
                        _rl_cost_per_inst = _prefix_cost.sum(dim=1) / _n_vis_fw
                        _safe_oc = _oc_t.clamp(min=1e-3)
                        _excess = ((_rl_cost_per_inst - _oc_t) / _safe_oc).clamp(-1.0, 3.0)
                        reward_fw = reward_fw + _ORACLE_REWARD_K * _excess
                        _oracle_excess_mean = float(_excess.mean().item())
                oracle_excess_log.append(float(_oracle_excess_mean))

                if _REPLAN_GATE and _use_residual_batch:
                    reward_fw = reward_fw - _GATE_COST

                if _K > 1:
                    reward_2d = reward_fw.view(_B, _K)
                    bl_rloo   = (reward_2d.sum(1, keepdim=True) - reward_2d) / (_K - 1)
                    advantage = adv_normalize((reward_2d - bl_rloo).view(-1))
                    reward    = reward_2d.mean(1)
                    tour_idx  = tour_idx_fw.view(_B, _K, _n_uav_i + _n_adr_i, -1)[:, 0]
                elif _OVERFIT:
                    reward        = reward_fw
                    baseline_reward = reward_fw.detach().mean().expand_as(reward_fw)
                    advantage     = adv_normalize(reward_fw - baseline_reward)
                    tour_idx      = tour_idx_fw
                else:
                    reward = reward_fw
                    if baseline is None:
                        baseline_reward = reward_fw.detach().mean().expand_as(reward_fw)
                    elif (_cached_bl_reward is None
                            or batch_idx % BASELINE_EVAL_FREQ == 0
                            or _cached_bl_reward.size(0) != batch['time_window'].size(0)):
                        _cached_bl_reward = baseline.eval(
                            batch, _n_uav_i, _n_adr_i, _steps_i,
                            initial_visited=_initial_visited_train).detach()
                        baseline_reward = _cached_bl_reward / _n_vis
                    else:
                        baseline_reward = _cached_bl_reward / _n_vis
                    advantage = adv_normalize(reward_fw - baseline_reward)
                    tour_idx  = tour_idx_fw

                critic_loss = torch.zeros((), device=device)
                # Actor-critic baseline when actor exposes value head (coop always, route via --critic).
                if value_fw_model is not None:
                    _value = value_fw_model.float()
                    if _value.dim() < reward_fw.dim():
                        _value = _value.view(reward_fw.shape[0], *([1] * (reward_fw.dim() - 1))).expand_as(reward_fw)
                    advantage = adv_normalize(reward_fw - _value.detach())
                    critic_loss = nn.functional.mse_loss(_value, reward_fw.detach())
                elif critic is not None:
                    value_fw = critic(_batch_fw, initial_visited=_iv_fw,
                                      num_uav=_n_uav_i, num_adr=_n_adr_i)
                    advantage = adv_normalize(reward_fw - value_fw.detach())
                    critic_loss = nn.functional.mse_loss(value_fw, reward_fw.detach())

                # Normalise by team size (not sequence length) to avoid biasing short episodes.
                _n_agents = _n_uav_i + _n_adr_i
                logp_per_step = tour_logp_fw.sum(dim=1) / _n_agents
                actor_loss = torch.mean(advantage.detach() * logp_per_step)
                # entropy_fw is per-step softmax entropy summed over decisions, per agent.
                entropy_mean = (entropy_fw.sum(dim=1) / _n_agents).mean()
                window_loss = torch.zeros((), device=device)
                total_loss = (actor_loss
                              - ent_coef * entropy_mean
                              + _CRITIC_COEF * critic_loss
                              + 0.01 * window_loss)

                _eff_accum = 1 if _OVERFIT else grad_accum
                (total_loss / _eff_accum).backward()

                _is_step = (
                    _OVERFIT
                    or (batch_idx + 1) % grad_accum == 0
                    or (batch_idx + 1) == _train_len
                )
                if _is_step:
                    grad_norm = torch.nn.utils.clip_grad_norm_(opt_params, max_grad_norm)
                    optimizer.step()
                else:
                    grad_norm = torch.tensor(0.0)

                rewards_log.append(reward.mean().item())
                entropies_log.append(float(entropy_mean.detach().item()))
                losses.append(actor_loss.item())
                critic_losses_log.append(float(critic_loss.detach().item()))
                grad_norms_log.append(grad_norm.item())

                # Mitigation telemetry: dyn_grad_l2 must be read after backward, before step.
                try:
                    from tools.diagnostics.mitigation_telemetry import (
                        depot_visit_fraction, dyn_proj_grad_l2)
                    _dyn_g = dyn_proj_grad_l2(actor, policy=_POLICY)
                    if _dyn_g > 0.0:
                        _telem_dyn_grad_l2.append(_dyn_g)
                    _d_frac = depot_visit_fraction(
                        tour_idx[:, :, 1:].detach(), int(_n_depots_i))
                    _telem_depot_frac.append(float(_d_frac))
                    _dec_owner = actor
                    for _n in ('module', 'actor', 'model', 'solver'):
                        _dec_owner = getattr(_dec_owner, _n, _dec_owner)
                        if hasattr(_dec_owner, 'decoder'):
                            break
                    _dec_obj = getattr(_dec_owner, 'decoder', None)
                    _dec_diag = getattr(_dec_obj, '_diag', None)
                    if isinstance(_dec_diag, dict):
                        _telem_dyn_depot_fire  += int(_dec_diag.get('dyn_depot_fires', 0))
                        _telem_dyn_depot_total += int(_dec_diag.get('dyn_depot_total_slots', 0))
                        _telem_depot_pick_tour_sum += int(
                            _dec_diag.get('depot_pick_sum_tour_len', 0))
                        _telem_depot_pick_count  += int(_dec_diag.get('depot_pick_count', 0))
                        _telem_legal_cust_sum    += int(
                            _dec_diag.get('legal_cust_at_depot_pick_sum', 0))
                    if value_fw_model is not None:
                        _v = value_fw_model.detach().float().flatten()
                        _r = reward_fw.detach().float().flatten()
                        _n = min(_v.numel(), _r.numel())
                        if _n > 0:
                            _telem_v_preds.extend(_v[:_n].tolist())
                            _telem_returns.extend(_r[:_n].tolist())
                except Exception as _telem_err:  # noqa: BLE001
                    if batch_idx == 0 and RANK == 0:
                        print(f'[telemetry] disabled this epoch: {_telem_err}',
                              flush=True)

                with torch.no_grad():
                    batch_size_b = tour_idx.size(0)
                    actions_b = tour_idx[:, :, 1:]
                    is_customer = actions_b >= _n_depots_i
                    uav_actions_total += int(is_customer[:, :_n_uav_i, :].sum().item())
                    all_actions_total += int(is_customer.sum().item())

                    # ── Vehicle-specialisation metrics ────────────────────────
                    # uav_demand_ratio <1 → UAVs carry lighter loads  (good)
                    # uav_dist_ratio   >1 → UAVs travel farther       (good)
                    is_pickup_b = (
                        (actions_b >= _n_depots_i)
                        & (actions_b < _n_depots_i + _n_req_i)
                    )
                    _dem = batch['demand'].squeeze(-1)              # [B, N]
                    B_b, A_b, T_b = actions_b.shape
                    _act_c  = actions_b.clamp(0, _dem.size(1) - 1)
                    _dem_exp = _dem.unsqueeze(1).expand(-1, A_b, -1)
                    _pkup_dem = torch.gather(_dem_exp, 2, _act_c) * is_pickup_b.float()
                    uav_demand_sum   += float(_pkup_dem[:, :_n_uav_i, :].sum().item())
                    uav_demand_count += int(is_pickup_b[:, :_n_uav_i, :].sum().item())
                    adr_demand_sum   += float(_pkup_dem[:, _n_uav_i:, :].sum().item())
                    adr_demand_count += int(is_pickup_b[:, _n_uav_i:, :].sum().item())

                    _ea_d = batch.get('edge_attr_d', None)
                    _ea_r = batch.get('edge_attr_r', None)
                    if _ea_d is not None and _ea_r is not None:
                        _n_sq = _ea_d.squeeze(-1).size(1)
                        _n_ea = int(_n_sq ** 0.5 + 0.5)
                        _mat_d = _ea_d.squeeze(-1).view(B_b, _n_ea, _n_ea)
                        _mat_r = _ea_r.squeeze(-1).view(B_b, _n_ea, _n_ea)
                        _prev = tour_idx[:, :, :-1].clamp(0, _n_ea - 1)
                        _nxt  = tour_idx[:, :,  1:].clamp(0, _n_ea - 1)
                        _bi   = torch.arange(B_b, device=_ea_d.device).view(-1, 1, 1).expand_as(_prev)
                        _leg_d = _mat_d[_bi, _prev, _nxt].clamp(max=50.0)
                        _leg_r = _mat_r[_bi, _prev, _nxt].clamp(max=50.0)
                        uav_dist_sum += float(_leg_d[:, :_n_uav_i, :].sum().item())
                        adr_dist_sum += float(_leg_r[:, _n_uav_i:, :].sum().item())

                if RANK == 0 and (batch_idx + 1) % 50 == 0:
                    print(
                        f'  [{batch_idx+1}/{_train_len}] '
                        f'rew={np.mean(rewards_log[-50:]):.4f}  '
                        f'loss={np.mean(losses[-50:]):.4f}  '
                        f'gnorm={np.mean(grad_norms_log[-50:]):.3f}  '
                        f'elapsed={time.time()-start:.0f}s',
                        flush=True,
                    )

            scheduler.step()

            if baseline is None or _OVERFIT or _RLOO_K > 0 or epoch < _WARMUP_EPOCHS:
                baseline_updated = 0
            elif not IS_DIST:
                baseline.epoch_callback(_inner(actor), epoch)
                baseline_updated = int(getattr(baseline, 'last_update_accepted', False))
            else:
                _upd_flag = torch.zeros(1, dtype=torch.long, device=device)
                _bl_mean  = torch.zeros(1, device=device)
                if RANK == 0:
                    baseline.epoch_callback(_inner(actor), epoch)
                    _upd_flag[0] = int(getattr(baseline, 'last_update_accepted', False))
                    _bl_mean[0]  = baseline.mean
                dist.broadcast(_upd_flag, src=0)
                dist.broadcast(_bl_mean,  src=0)
                baseline_updated = int(_upd_flag.item())
                if baseline_updated:
                    for _p in baseline.model.parameters():
                        dist.broadcast(_p.data, src=0)
                    for _b in baseline.model.buffers():
                        dist.broadcast(_b, src=0)
                baseline.mean = float(_bl_mean.item())

            if RANK == 0 and (epoch % 25 == 0 or epoch == max_epochs - 1):
                epoch_dir = os.path.join(folder, str(epoch))
                os.makedirs(epoch_dir, exist_ok=True)
                torch.save(dict(model_state_dict=_inner(actor).state_dict(),
                                route_contract=route_checkpoint_contract()),
                           os.path.join(epoch_dir, 'actor.pt'))

            _seed_costs = torch.zeros(len(_val_loaders_ms), device=device)
            _breakdown_vals = torch.zeros(9, device=device)
            # val_complete_rate complements val_cost (soft prefix scorer can improve
            # while mandatory-completion still fails).
            _complete_rate_val = float('nan')
            if not IS_DIST or RANK == 0:
                _breakdown_costs, _tr, _tw, _ud, _dc, _bat = rollout(
                    _inner(actor), _val_loaders_ms[0], breakdown=True)
                _breakdown_vals[0] = _tr.mean()
                _breakdown_vals[1] = _tw.mean()
                _breakdown_vals[2] = _ud.mean()
                _breakdown_vals[5] = _dc.mean()
                _breakdown_vals[8] = _bat.mean()
                _complete_rate_val = float(
                    getattr(_breakdown_costs, '_complete_rate', float('nan')))

                # Dynamic: average over t snapshots so selection reflects full training range.
                for _si, _vl in enumerate(_val_loaders_ms):
                    if _DYNAMIC:
                        _per_t = torch.stack([
                            rollout(_inner(actor), _vl, t_current=_t).mean()
                            for _t in _VAL_T_SNAPSHOTS
                        ])
                        _seed_costs[_si] = _per_t.mean()
                    else:
                        _seed_costs[_si] = rollout(_inner(actor), _vl).mean()

                _hard_cost_vals, _hard_tw_vals = [], []
                for _vl_h in _val_loaders_hard:
                    try:
                        _hc, _, _htw, _, _, _ = rollout(_inner(actor), _vl_h, breakdown=True)
                        _hard_cost_vals.append(_hc.mean().item())
                        _hard_tw_vals.append(_htw.mean().item())
                    except RuntimeError:
                        pass
                if _hard_cost_vals:
                    _breakdown_vals[3] = sum(_hard_cost_vals) / len(_hard_cost_vals)
                    _breakdown_vals[4] = sum(_hard_tw_vals)  / len(_hard_tw_vals)
                else:
                    _breakdown_vals[3] = float('nan')
                    _breakdown_vals[4] = float('nan')

                # Unvalidated prefix diagnostics must not select a checkpoint.
                _breakdown_vals[6:8] = float('nan')
            if IS_DIST:
                dist.broadcast(_seed_costs,     src=0)
                dist.broadcast(_breakdown_vals, src=0)
            val_cost          = float(_seed_costs.mean().item())
            val_cost_std      = float(_seed_costs.std().item()) if len(_val_loaders_ms) > 1 else 0.0
            val_complete_rate = _complete_rate_val
            val_travel_cost   = float(_breakdown_vals[0])
            val_tw_penalty    = float(_breakdown_vals[1])
            val_undeliv_cost  = float(_breakdown_vals[2])
            val_hard_cost     = float(_breakdown_vals[3])
            val_hard_tw_penalty = float(_breakdown_vals[4])
            val_deliv_credit  = float(_breakdown_vals[5])
            val_battery_cost  = float(_breakdown_vals[8])
            val_residual_cost = float(_breakdown_vals[6])
            val_residual_tw_penalty = float(_breakdown_vals[7])
            val_select_cost = (
                0.7 * val_cost + 0.3 * val_residual_cost
                if _RESIDUAL else val_cost
            )

            if RANK == 0:
                costs.append(val_select_cost)
                np.savetxt(os.path.join(folder, 'costs.txt'), costs)
                save_checkpoint(epoch, actor, optimizer, scheduler, folder, costs, critic=critic)

                # Lexicographic selection: (-complete_rate, val_select_cost).
                # Falls back to cost-only when complete_rate is NaN.
                import math as _m_sel
                _cur_cr = val_complete_rate if not _m_sel.isnan(val_complete_rate) else -1.0
                _best_cr = getattr(save_checkpoint, '_best_complete_rate', -1.0)
                _accept = False
                if _cur_cr > _best_cr + 1e-9:
                    _accept = True
                elif _cur_cr >= _best_cr - 1e-9 and val_select_cost < best_val_cost:
                    _accept = True
                if _accept:
                    best_val_cost = val_select_cost
                    save_checkpoint._best_complete_rate = _cur_cr  # type: ignore[attr-defined]
                    torch.save(dict(model_state_dict=_inner(actor).state_dict(),
                                    route_contract=route_checkpoint_contract()),
                               os.path.join(folder, 'best_actor.pt'))
                    print(f'  New best: select_cost={val_select_cost:.4f} '
                          f'complete_rate={val_complete_rate:.3f} -> saved best_actor.pt',
                          flush=True)

                mode_uav_frac = (uav_actions_total / max(all_actions_total, 1))
                _u_dem = uav_demand_sum / max(uav_demand_count, 1)
                _a_dem = adr_demand_sum / max(adr_demand_count, 1)
                uav_demand_ratio = _u_dem / max(_a_dem, 1e-8)
                _u_dis = uav_dist_sum / max(uav_actions_total, 1)
                _a_dis = adr_dist_sum / max(all_actions_total - uav_actions_total, 1)
                uav_dist_ratio = _u_dis / max(_a_dis, 1e-8)
                current_lr = optimizer.param_groups[0]['lr']
                elapsed = time.time() - start
                grad_norm_max = float(max(grad_norms_log)) if grad_norms_log else 0.0

                _std_str = f' ±{val_cost_std:.4f}' if val_cost_std > 0 else ''
                print(
                    f'  Rung {RUNG} | {_ARCH} | Epoch {epoch} | '
                    f'Val cost: {val_cost:.4f}{_std_str} | '
                    f'[travel={val_travel_cost:.4f} tw={val_tw_penalty:.4f} '
                    f'ud={val_undeliv_cost:.4f} bat={val_battery_cost:.4f} dc={val_deliv_credit:.4f}] | '
                    f'Hard: cost={val_hard_cost:.4f} tw={val_hard_tw_penalty:.4f} | '
                    f'Residual: cost={val_residual_cost:.4f} tw={val_residual_tw_penalty:.4f} | '
                    f'Select: {val_select_cost:.4f} | '
                    f'Complete: {val_complete_rate:.3f} | '
                    f'UAV frac: {mode_uav_frac:.3f} | '
                    f'demand_ratio: {uav_demand_ratio:.3f} dist_ratio: {uav_dist_ratio:.3f} | '
                    f'critic_loss: {np.mean(critic_losses_log):.3f} | '
                    f'grad_norm: {np.mean(grad_norms_log):.3f} (max {grad_norm_max:.3f}) | '
                    f'bl_updated: {bool(baseline_updated)}',
                    flush=True,
                )
                print(f'  History: {[f"{c:.3f}" for c in costs[-5:]]}', flush=True)

                if _csv_writer is not None:
                    _csv_writer.writerow({
                        'epoch':              epoch,
                        'rung':               RUNG,
                        'arch':               _ARCH,
                        'rloo_k':             _RLOO_K,
                        'entropy_coef':       _ENTROPY_COEF,
                        'lr_decay':           _LR_DECAY,
                        'train_reward_mean':  np.mean(rewards_log),
                        'train_loss_mean':    np.mean(losses),
                        'critic_loss_mean':   np.mean(critic_losses_log),
                        'val_cost':           val_cost,
                        'val_cost_std':       val_cost_std,
                        'val_travel_cost':    val_travel_cost,
                        'val_tw_penalty':     val_tw_penalty,
                        'val_undeliv_cost':   val_undeliv_cost,
                        'val_deliv_credit':   val_deliv_credit,
                        'val_battery_cost':   val_battery_cost,
                        'val_hard_cost':      val_hard_cost,
                        'val_hard_tw_penalty': val_hard_tw_penalty,
                        'val_residual_cost':  val_residual_cost,
                        'val_residual_tw_penalty': val_residual_tw_penalty,
                        'val_select_cost':    val_select_cost,
                        'val_complete_rate':  val_complete_rate,
                        'grad_norm_mean':     np.mean(grad_norms_log),
                        'grad_norm_max':      grad_norm_max,
                        'entropy_mean':       float(np.mean(entropies_log)) if entropies_log else float('nan'),
                        'baseline_updated':   baseline_updated,
                        'mode_uav_frac':      mode_uav_frac,
                        'uav_demand_ratio':   uav_demand_ratio,
                        'uav_dist_ratio':     uav_dist_ratio,
                        'lr':                 current_lr,
                        'elapsed_s':          elapsed,
                        'critic_corr':        (
                            __import__('tools.diagnostics.mitigation_telemetry',
                                       fromlist=['critic_pearson_corr'])
                            .critic_pearson_corr(_telem_v_preds, _telem_returns)
                            if _telem_v_preds else 0.0
                        ),
                        'dyn_grad_l2':        (
                            float(np.mean(_telem_dyn_grad_l2))
                            if _telem_dyn_grad_l2 else 0.0
                        ),
                        'depot_visit_frac':   (
                            float(np.mean(_telem_depot_frac))
                            if _telem_depot_frac else 0.0
                        ),
                        # Probe 1 — per-term batch means across the epoch
                        'train_travel_mean':        (float(np.mean(travel_log))        if travel_log        else 0.0),
                        'train_pickup_early_mean':  (float(np.mean(pickup_early_log))  if pickup_early_log  else 0.0),
                        'train_pickup_late_mean':   (float(np.mean(pickup_late_log))   if pickup_late_log   else 0.0),
                        'train_delivery_late_mean': (float(np.mean(delivery_late_log)) if delivery_late_log else 0.0),
                        'train_tw_total_mean':      (float(np.mean(tw_total_log))      if tw_total_log      else 0.0),
                        'train_shaping_mean':       (float(np.mean(shaping_log))       if shaping_log       else 0.0),
                        'train_battery_slack_mean': (float(np.mean(battery_slack_log)) if battery_slack_log else 0.0),
                        'train_oracle_excess_mean': (float(np.mean(oracle_excess_log)) if oracle_excess_log else 0.0),
                        # Paper-strict mask diagnostics
                        'dyn_depot_fire_frac': (
                            float(_telem_dyn_depot_fire) / max(1, _telem_dyn_depot_total)
                            if _telem_dyn_depot_total > 0 else 0.0
                        ),
                        'depot_pick_mean_tour': (
                            float(_telem_depot_pick_tour_sum) / max(1, _telem_depot_pick_count)
                            if _telem_depot_pick_count > 0 else 0.0
                        ),
                        'legal_cust_at_depot': (
                            float(_telem_legal_cust_sum) / max(1, _telem_depot_pick_count)
                            if _telem_depot_pick_count > 0 else 0.0
                        ),
                    })
                    _csv_file.flush()

                if _tb_writer is not None:
                    step = epoch
                    _tb_writer.add_scalar('train/reward_mean', np.mean(rewards_log), step)
                    _tb_writer.add_scalar('train/loss_mean', np.mean(losses), step)
                    _tb_writer.add_scalar('train/grad_norm_mean', np.mean(grad_norms_log), step)
                    _tb_writer.add_scalar('train/grad_norm_max', grad_norm_max, step)
                    _tb_writer.add_scalar('train/baseline_updated', baseline_updated, step)
                    _tb_writer.add_scalar('val/cost', val_cost, step)
                    _tb_writer.add_scalar('train/mode_uav_frac', mode_uav_frac, step)
                    _tb_writer.add_scalar('train/lr', current_lr, step)

        if _SMOKE and RANK == 0:
            first_cost, last_cost = costs[0], costs[-1]
            cost_improved = last_cost <= first_cost
            checkpoint_ok = os.path.exists(os.path.join(folder, 'checkpoint.pth'))
            best_ok       = os.path.exists(os.path.join(folder, 'best_actor.pt'))
            print('\n' + '=' * 55)
            print('[SMOKE] Pipeline verdict')
            print('=' * 55)
            print(f'  Val cost trajectory: {[f"{c:.4f}" for c in costs]}')
            print(f'  Cost improved epoch 0->last: {"YES" if cost_improved else "NO - may need more epochs"}')
            print(f'  Checkpoint saved:    {"YES" if checkpoint_ok else "FAIL"}')
            print(f'  best_actor.pt saved: {"YES" if best_ok else "FAIL"}')
            print(f'  CSV log exists:      {"YES" if os.path.exists(csv_log_path) else "FAIL"}')
            print('  Diagnostic pipeline check only; scientific release gates remain open.'
                  if (checkpoint_ok and best_ok) else '  Fix diagnostic failures before further validation.')
            print('=' * 55)

        if IS_DIST:
            dist.barrier()

    if RANK == 0:
        if _csv_file is not None:
            _csv_file.close()
        if _tb_writer is not None:
            _tb_writer.close()

if __name__ == '__main__':
    try:
        train()
    finally:
        if IS_DIST:
            dist.destroy_process_group()
