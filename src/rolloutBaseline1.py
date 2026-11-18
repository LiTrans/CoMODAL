import copy
import numpy as np
import torch
from scipy.stats import ttest_rel
from torch.nn import DataParallel
from creat_vrp import reward1
from reward import IncompleteTensorRouteError

# Baseline rollout runs an untrained/weak policy with greedy=True under the
# mandatory-service contract; the decoder correctly raises when it cannot make
# progress. Treat such batches as the baseline failing to complete and return a
# large but finite penalty cost so (a) the arithmetic (.mean(), ttest_rel) stays
# well-defined and (b) a candidate model with any completion improvement swamps
# the baseline on update. Successful finite costs at n_req=5 fall in roughly
# the 10^1-10^4 range; 1e6 keeps failures dominant without going nonfinite.
BASELINE_FAILURE_PENALTY = 1e6

def get_inner_model(model):
    return model.module if isinstance(model, DataParallel) else model

def rollout1(model, dataset, num_uav, num_adr, n_nodes, transform_fn=None,
             delivery_shaping_k=0.0):
    model.eval()
    device = next(model.parameters()).device

    def eval_batch(batch):
        batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}
        batch_size = int(batch['time_window'].shape[0])
        initial_visited = None
        if transform_fn is not None:
            batch, initial_visited = transform_fn(batch)
        # Baseline is a weak policy by construction; match the training rollout's
        # rolling_prefix scope so partial decodes return a finite cost rather than
        # raising. Reported scoring (eval, release) still uses complete_episode.
        batch = dict(batch)
        batch['route_scope'] = 'rolling_prefix'
        with torch.no_grad():
            try:
                # Model.forward returns 3-tuple normally, 4-tuple when
                # critic=True (adds V(s) at the end). Baseline only needs
                # (tour, log_p, time_tensor), so take the first three
                # elements regardless of critic flag.
                _out = model(batch, num_uav, num_adr, greedy=True,
                             initial_visited=initial_visited)
                tour, _, time_tensor = _out[0], _out[1], _out[2]
                nd = batch.get('n_depots', None)
                if nd is not None:
                    num_depots = int(nd[0]) if hasattr(nd, '__len__') else int(nd)
                else:
                    num_depots = None
                # Partial trajectories need allow_partial=True so replay_mandatory_trajectory
                # returns the prefix_cost rather than raising. PBRS shaping (if k>0) subtracts
                # a per-episode bonus proportional to delivered_count to counter the "learn
                # to not deliver" incentive that partial trajectories otherwise create (no
                # missed-order penalty under mandatory). Policy-invariant per Ng et al. 1999.
                prefix_cost, components = reward1(
                    batch['time_window'],
                    tour.detach(),
                    batch['edge_attr_d'],
                    batch['edge_attr_r'],
                    time_tensor,
                    num_uav,
                    num_depots=num_depots,
                    initial_visited=initial_visited,
                    start_time=batch.get('start_time', None),
                    available_mask=batch.get('available_mask', None),
                    physics_batch=batch,
                    service_policy=batch.get('service_policy', 'legacy_optional'),
                    allow_partial=True, return_breakdown=True,
                )
                cost = prefix_cost.sum(dim=1)
                if delivery_shaping_k > 0.0:
                    # Credit delivered-from-ACTIVE requests,
                    # not delivered-from-ALL. Under rolling residuals some
                    # requests are hidden (active_request_mask==False); the
                    # old (n_req - unassigned_count) overcounted them as
                    # "delivered" and leaked unrevealed-request credit into
                    # the baseline value.
                    n_total = batch['demand'].shape[1]
                    n_req = max(0, (n_total - (num_depots or 0)) // 2)
                    _am = batch.get('active_request_mask', None)
                    if _am is not None:
                        active_count = torch.as_tensor(_am, device=cost.device).bool().reshape(cost.shape[0], -1).sum(dim=1).to(cost.dtype)
                    else:
                        active_count = cost.new_full((cost.shape[0],), float(n_req))
                    delivered = (active_count - components['unassigned_count']).to(cost.dtype)
                    cost = cost - delivery_shaping_k * delivered
            except IncompleteTensorRouteError:
                return torch.full((batch_size,), BASELINE_FAILURE_PENALTY, dtype=torch.float32)
        return cost.cpu()

    return torch.cat([eval_batch(batch) for batch in dataset], dim=0)

class RolloutBaseline:
    def __init__(self, model, dataset, n_uav, n_adr, n_nodes=50, epoch=0,
                 transform_fn=None, delivery_shaping_k=0.0,
                 refresh_after_stale=0):
        self.n_nodes = n_nodes
        self.n_uav = n_uav
        self.n_adr = n_adr
        self.dataset = dataset
        self.last_update_accepted = False
        self.transform_fn = transform_fn
        self.delivery_shaping_k = max(0.0, float(delivery_shaping_k))
        # Break lock-in: once this many consecutive candidate rejections pile up,
        # force a baseline refresh at the next epoch_callback. The one-sided t-test
        # gate (p<0.05) otherwise lets a lucky snapshot freeze indefinitely on the
        # fixed baseline bank, even when the policy has drifted into a different
        # local mode that scores well on the training distribution. 0 = disabled.
        self.refresh_after_stale = int(max(0, refresh_after_stale))
        self.consecutive_stale = 0
        self._update_model(model, epoch)

    def _update_model(self, model, epoch, dataset=None):
        if dataset is not None:
            self.dataset = dataset
        self.model = copy.deepcopy(model)
        self.model.eval()
        self.bl_vals = rollout1(
            self.model, self.dataset, self.n_uav, self.n_adr, self.n_nodes,
            transform_fn=self.transform_fn,
            delivery_shaping_k=self.delivery_shaping_k,
        ).cpu().numpy()
        self.mean = float(self.bl_vals.mean())
        self.epoch = epoch

    def eval(self, x, num_uav, num_adr, n_nodes, initial_visited=None):
        device = next(self.model.parameters()).device
        x = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in x.items()}
        self.model.eval()
        # Match rollout1: rolling_prefix scope + allow_partial reward + PBRS shift.
        x = dict(x)
        x['route_scope'] = 'rolling_prefix'
        with torch.no_grad():
            try:
                tour, _, time_tensor = self.model(x, num_uav, num_adr, greedy=True,
                                                  initial_visited=initial_visited)
                nd = x.get('n_depots', None)
                if nd is not None:
                    num_depots = int(nd[0]) if hasattr(nd, '__len__') else int(nd)
                else:
                    num_depots = None
                prefix_cost, components = reward1(
                    x['time_window'],
                    tour.detach(),
                    x['edge_attr_d'],
                    x['edge_attr_r'],
                    time_tensor,
                    num_uav,
                    num_depots=num_depots,
                    initial_visited=initial_visited,
                    start_time=x.get('start_time', None),
                    available_mask=x.get('available_mask', None),
                    physics_batch=x,
                    service_policy=x.get('service_policy', 'legacy_optional'),
                    allow_partial=True, return_breakdown=True,
                )
                value = prefix_cost.sum(dim=1)
                if self.delivery_shaping_k > 0.0:
                    # Credit delivered-from-ACTIVE requests.
                    n_total = x['demand'].shape[1]
                    n_req = max(0, (n_total - (num_depots or 0)) // 2)
                    _am = x.get('active_request_mask', None)
                    if _am is not None:
                        active_count = torch.as_tensor(_am, device=value.device).bool().reshape(value.shape[0], -1).sum(dim=1).to(value.dtype)
                    else:
                        active_count = value.new_full((value.shape[0],), float(n_req))
                    delivered = (active_count - components['unassigned_count']).to(value.dtype)
                    value = value - self.delivery_shaping_k * delivered
            except IncompleteTensorRouteError:
                batch_size = int(x['time_window'].shape[0])
                value = torch.full((batch_size,), BASELINE_FAILURE_PENALTY,
                                   dtype=torch.float32, device=device)
        return value

    def epoch_callback(self, model, epoch):
        print('Evaluating candidate model on validation dataset')
        cand_vals = rollout1(
            model, self.dataset, self.n_uav, self.n_adr, self.n_nodes,
            transform_fn=self.transform_fn,
            delivery_shaping_k=self.delivery_shaping_k,
        ).cpu().numpy()

        cand_mean = float(cand_vals.mean())
        diff = cand_mean - self.mean
        print(
            f'Epoch {epoch} candidate mean {cand_mean:.4f}, '
            f'baseline epoch {self.epoch} mean {self.mean:.4f}, '
            f'diff {diff:.4f}'
        )

        def _keep(reason):
            print(f'Keep baseline ({reason})')
            self.last_update_accepted = False
            self.consecutive_stale += 1
            if (self.refresh_after_stale > 0
                    and self.consecutive_stale >= self.refresh_after_stale):
                print(f'Baseline stale for {self.consecutive_stale} consecutive '
                      f'epochs; force refresh at epoch {epoch}')
                self._update_model(model, epoch)
                self.last_update_accepted = True
                self.consecutive_stale = 0

        if diff >= 0:
            _keep('candidate not better')
            return

        cand_episode = cand_vals
        base_episode = self.bl_vals

        if len(cand_episode) < 2:
            _keep('too few samples for t-test')
            return

        t_stat, p_two_sided = ttest_rel(cand_episode, base_episode, nan_policy='omit')
        if np.isnan(t_stat) or np.isnan(p_two_sided):
            _keep('t-test undefined')
            return

        p_val = p_two_sided / 2.0
        print(f'p-value: {p_val:.4f}')

        if t_stat < 0 and p_val < 0.05:
            print('Updating baseline')
            self._update_model(model, epoch)
            self.last_update_accepted = True
            self.consecutive_stale = 0
        else:
            _keep(f't-test not significant (t={t_stat:.3f}, p={p_val:.3f})')

    def state_dict(self):
        return {
            'model': self.model,
            'dataset': self.dataset,
            'epoch': self.epoch,
        }

    def load_state_dict(self, state_dict):
        model_copy = copy.deepcopy(self.model)
        get_inner_model(model_copy).load_state_dict(
            get_inner_model(state_dict['model']).state_dict()
        )
        self._update_model(model_copy, state_dict['epoch'], state_dict['dataset'])
