
from artifacts import load_tensor_file
import sys
import argparse
from typing import Dict, List, Optional, Tuple

def select_route_candidate(candidates, vehicles, requests, full_instance):
    import math
    from reward import score_route_plan, mandatory_service
    required_service = mandatory_service(full_instance)
    best_plan, best_key, scores = None, None, []
    for plan in candidates:
        score = score_route_plan(plan, vehicles, requests, full_instance)
        scores.append(score)
        cost = score['prefix_cost'] if required_service else score['total']
        key = (score['unassigned_count'] if required_service else 0, cost)
        if math.isfinite(cost) and (best_key is None or key < best_key):
            best_plan, best_key = plan, key
    if best_plan is None:
        raise ValueError('No finite route candidate was available')
    return best_plan, scores

def build_solver(model_path: str, arch: str = 'hetgat', n_samples: int = 1):
    """Build the rolling-horizon RL solver. n_samples>1 enables sampled decoding."""
    import torch
    from VRP_Actor import Model
    from VRP_CoopActorCritic import CoopActorCritic
    from dispatch_sim import Leg, Request
    from rl_value_baseline import attach_window_context
    from replan_gate import build_replan_features

    if not model_path:
        raise ValueError(
            'A trained model checkpoint is required. Pass --model path/to/actor.pt'
        )

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    _raw = load_tensor_file(model_path, map_location=device, weights_only=True)
    _state = _raw['model_state_dict'] if isinstance(_raw, dict) and 'model_state_dict' in _raw else _raw
    _input_node_dim = 11
    _is_coop = any(str(k).startswith('request_encoder.') for k in _state)
    _policy = 'coop' if _is_coop else 'route'
    if _is_coop:
        _w = _state.get('request_encoder.net.0.weight')
        _hidden_dim = int(_w.shape[0]) if _w is not None else 128
        _input_node_dim = int((_w.shape[1] - 10) // 2) if _w is not None else 15
    else:
        for _k, _v in _state.items():
            if _k.endswith('encoder.W_depot.weight'):
                _input_node_dim = int(_v.shape[1])
                break
    _dynamic_features = (_input_node_dim > 11)
    if _is_coop:
        actor = CoopActorCritic(
            input_node_dim=_input_node_dim,
            hidden_dim=_hidden_dim,
            n_heads=4,
            n_layers=2,
        ).to(device)
    else:
        edge_dim = 1 if arch == 'simplegat' else 4
        if arch == 'koolam':
            # Kool et al. 2019 Attention Model; see baselines/kool_am.py.
            from baselines import KoolAMModel
            actor = KoolAMModel(
                input_node_dim=_input_node_dim,
                hidden_node_dim=128,
                input_edge_dim=edge_dim,
                hidden_edge_dim=16,
                conv_layers=3,
            ).to(device)
        elif arch == 'hetero_attn_pdp':
            # Li et al. 2019 heterogeneous attention; see baselines/hetero_attn_pdp.py.
            from baselines import HAPDPModel
            actor = HAPDPModel(
                input_node_dim=_input_node_dim,
                hidden_node_dim=128,
                input_edge_dim=edge_dim,
                hidden_edge_dim=16,
                conv_layers=3,
            ).to(device)
        elif arch == 'gann':
            # Guo GCN-style message passing; see baselines/gann.py.
            from baselines import GANNModel
            actor = GANNModel(
                input_node_dim=_input_node_dim,
                hidden_node_dim=128,
                input_edge_dim=edge_dim,
                hidden_edge_dim=16,
                conv_layers=3,
            ).to(device)
        else:
            actor = Model(
                input_node_dim=_input_node_dim,
                hidden_node_dim=128,
                input_edge_dim=edge_dim,
                hidden_edge_dim=16,
                conv_layers=3,
                arch=arch,
            ).to(device)
    if isinstance(_raw, dict) and 'model_state_dict' in _raw:
        actor.load_state_dict(_raw['model_state_dict'])
        print(f'Loaded checkpoint from {model_path} (epoch {_raw.get("epoch", "?")}, '
              f'policy={_policy}, input_node_dim={_input_node_dim})')
    else:
        actor.load_state_dict(_raw)
        print(f'Loaded model weights from {model_path} '
              f'(policy={_policy}, input_node_dim={_input_node_dim})')
    actor.eval()

    class RLSolver:
        def __init__(self, actor_model, torch_device, dynamic_features=False,
                     input_node_dim=11, policy='route'):
            self.actor = actor_model
            self.device = torch_device
            self.dynamic_features = dynamic_features
            self.policy = policy
            self._n_dyn_dims = max(0, input_node_dim - 11)

        def _batchify_instance(self, instance: dict) -> dict:
            from reward import native_physical_configuration
            config = native_physical_configuration(instance)
            batched = {}
            for key, value in instance.items():
                if torch.is_tensor(value):
                    batched[key] = value.unsqueeze(0).to(self.device)
                else:
                    batched[key] = value
            if 'remaining_delivery_ids' in instance:
                from ce_cpdptw_alns import remaining_delivery_ids
                ids = torch.tensor(sorted(remaining_delivery_ids(instance)),
                                   dtype=torch.long, device=self.device)
                pending = torch.zeros((1, int(instance['n_req'])), dtype=torch.bool,
                                      device=self.device)
                pending[0, ids] = True
                supplied = batched.get('remaining_deliveries')
                if supplied is not None and (not torch.is_tensor(supplied)
                        or supplied.dtype != torch.bool or not torch.equal(supplied, pending)):
                    raise ValueError('Scalar and tensor pending-delivery ledgers disagree')
                batched['remaining_deliveries'] = pending
            if config is not None:
                batched['physical_config'] = config.to_dict()
            return batched

        def _num_agents(self, vehicles: Dict[int, object]) -> Tuple[int, int]:
            n_uav = sum(1 for v in vehicles.values() if v.mode == 'uav')
            n_adr = sum(1 for v in vehicles.values() if v.mode == 'adr')
            return n_uav, n_adr

        def _vehicle_ids(self, vehicles):
            if any(v.mode not in ('uav', 'adr') for v in vehicles.values()):
                raise ValueError('Unsupported RL vehicle mode')
            return sorted(vehicles, key=lambda vid: (vehicles[vid].mode != 'uav', vid))

        def _validate_request_context(self, full_instance, active_requests):
            if full_instance.get('service_policy') != 'mandatory':
                return
            n_depots, n_req = int(full_instance['n_depots']), int(full_instance['n_req'])
            demand = full_instance['demand'].reshape(-1)
            tw = full_instance['time_window'].reshape(-1)
            import math
            for rid, req in active_requests.items():
                if (rid != req.req_id or not 0 <= rid < n_req
                        or req.pickup_node != n_depots + rid
                        or req.delivery_node != n_depots + n_req + rid):
                    raise ValueError('Native RL requires explicit dense request/node mapping')
                checks = ((float(demand[req.pickup_node]), req.demand),
                          (float(demand[req.delivery_node]), -req.demand),
                          (float(tw[req.pickup_node]), req.t_pickup),
                          (float(tw[req.delivery_node]), req.t_delivery))
                if req.demand <= 0 or any(not math.isclose(a, b, rel_tol=1e-5, abs_tol=1e-4)
                                         for a, b in checks):
                    raise ValueError('Native tensor request data disagree with the actual scalar task book')

        def _distance_matrix(self, full_instance: dict, mode: str):
            key = 'edge_attr_d' if mode == 'uav' else 'edge_attr_r'
            raw = full_instance[key]
            n_total = full_instance['x'].shape[0]
            return raw.view(n_total, n_total).detach().cpu()

        def _speed(self, mode: str) -> float:
            from vrpUpdate import V_UAV_MAX, V_ADR_MAX
            return V_UAV_MAX if mode == 'uav' else V_ADR_MAX

        def _travel_time(self, full_instance: dict, mode: str, from_node: int, to_node: int) -> float:
            dm = self._distance_matrix(full_instance, mode)
            d = float(dm[from_node, to_node].item())
            if d >= 1e9:
                return float('inf')
            return d / max(self._speed(mode), 1e-9)

        def _mask_unrevealed_nodes(self, batch: dict, active_requests, full_instance) -> dict:
            import torch
            n_depots = int(full_instance['n_depots'])
            n_total = batch['x'].shape[1]

            revealed_nodes = set(range(n_depots))
            for req in active_requests.values():
                revealed_nodes.add(int(req.pickup_node))
                revealed_nodes.add(int(req.delivery_node))

            unrevealed = [i for i in range(n_depots, n_total) if i not in revealed_nodes]
            if not unrevealed:
                return batch

            batch = {k: (v.clone() if isinstance(v, torch.Tensor) else v)
                     for k, v in batch.items()}
            device = batch['x'].device
            ut = torch.tensor(unrevealed, dtype=torch.long, device=device)

            batch['x'][:, ut, :] = 0.0
            if 'demand' in batch and isinstance(batch['demand'], torch.Tensor):
                batch['demand'][:, ut] = 0.0
            if 'time_window' in batch and isinstance(batch['time_window'], torch.Tensor):
                batch['time_window'][:, ut] = 0.0

            n_sq = n_total * n_total
            for mask_key in ('mask_adjacency_uav', 'mask_adjacency_adr'):
                m = batch.get(mask_key)
                if not isinstance(m, torch.Tensor) or m.dim() != 3 or m.shape[1] != n_sq:
                    continue
                ch = m.shape[-1]
                m2d = m.view(m.shape[0], n_total, n_total, ch)
                m2d[:, ut, :, :] = 0.0
                m2d[:, :, ut, :] = 0.0

            for edge_key in ('edge_attr_uav', 'edge_attr_adr',
                             'edge_attr_d', 'edge_attr_r'):
                if (full_instance.get('service_policy') == 'mandatory'
                        and edge_key in ('edge_attr_d', 'edge_attr_r')):
                    # Visibility hides tasks, not street geometry.
                    continue
                e = batch.get(edge_key)
                if not isinstance(e, torch.Tensor) or e.dim() != 3 or e.shape[1] != n_sq:
                    continue
                ch = e.shape[-1]
                e2d = e.view(e.shape[0], n_total, n_total, ch)
                e2d[:, ut, :, :] = 0.0
                e2d[:, :, ut, :] = 0.0

            return batch

        def _attach_vehicle_state(self, batch, vehicles: Dict[int, object],
                                  current_time: float) -> dict:
            import torch
            from ce_cpdptw_alns import remaining_recharges
            ordered = [vehicles[vid] for vid in self._vehicle_ids(vehicles)]
            device = batch['x'].device
            batch = dict(batch)
            onboard_owner = torch.full((1, int(batch['n_req'])), -1, dtype=torch.long, device=device)
            for slot, vehicle in enumerate(ordered):
                for req_id in vehicle.onboard_set():
                    if not 0 <= req_id < onboard_owner.size(1) or onboard_owner[0, req_id] >= 0:
                        raise ValueError('Invalid physically onboard request ledger')
                    onboard_owner[0, req_id] = slot
            batch['onboard_owner'] = onboard_owner
            batch['start_node'] = torch.tensor(
                [[int(v.current_node) for v in ordered]],
                dtype=torch.long, device=device,
            )
            batch['start_time'] = torch.tensor(
                [[float(max(current_time, v.current_time)) for v in ordered]],
                dtype=torch.float32, device=device,
            )
            batch['start_battery'] = torch.tensor(
                [[float(v.battery) for v in ordered]], dtype=torch.float32, device=device,
            )
            batch['start_capacity'] = torch.tensor(
                [[float(v.capacity) - float(v.load) for v in ordered]],
                dtype=torch.float32, device=device,
            )
            batch['available_mask'] = torch.tensor(
                [[1.0 if v.committed_leg is None else 0.0 for v in ordered]],
                dtype=torch.float32, device=device,
            )
            batch['capacity'] = torch.tensor(
                [[float(v.capacity) for v in ordered]], dtype=torch.float32, device=device,
            )
            batch['battery'] = torch.tensor(
                [[float(getattr(v, 'battery_init', None) or v.battery) for v in ordered]],
                dtype=torch.float32, device=device,
            )
            n_depots = int(batch['n_depots'])
            batch['agent_depot_allowed'] = torch.tensor([[
                [v.allowed_depots is None or depot in v.allowed_depots for depot in range(n_depots)]
                for v in ordered]], dtype=torch.bool, device=device)
            recharge_budget = [remaining_recharges(v, batch) for v in ordered]
            if recharge_budget and recharge_budget[0] is not None:
                batch['remaining_recharges'] = torch.tensor(
                    [recharge_budget], dtype=torch.long, device=device,
                )
            return batch

        def _restore_occupied_origins(self, batch, full_instance, vehicles, initial_visited):
            if full_instance.get('service_policy') != 'mandatory':
                return batch
            n = batch['x'].shape[1]
            known = initial_visited[0] <= .5
            known[:int(full_instance['n_depots'])] = True
            batch = dict(batch)
            for key in ('mask_adjacency_uav', 'mask_adjacency_adr'):
                if key not in batch:
                    continue
                restored = batch[key].clone().reshape(1, n, n, -1)
                original = full_instance[key].reshape(n, n, -1).to(restored)
                for vehicle in vehicles.values():
                    origin = int(vehicle.current_node)
                    restored[0, origin, known, :] = original[origin, known, :]
                batch[key] = restored.reshape_as(batch[key])
            return batch

        def _attach_delivery_owner(self, batch, active_requests, vehicles, full_instance) -> dict:
            import torch
            n_depots = int(full_instance['n_depots'])
            n_req = int(full_instance['n_req'])
            local_by_vehicle = {vid: i for i, vid in enumerate(self._vehicle_ids(vehicles))}
            owner = torch.full(
                (1, n_req), -1, dtype=torch.long, device=batch['x'].device,
            )
            for req in active_requests.values():
                if req.status not in ('pickup_committed', 'onboard', 'delivery_committed'):
                    continue
                local_idx = local_by_vehicle.get(req.assigned_vehicle)
                if local_idx is None:
                    continue
                req_idx = int(req.delivery_node) - n_depots - n_req
                if 0 <= req_idx < n_req:
                    owner[0, req_idx] = int(local_idx)
            batch = dict(batch)
            batch['delivery_owner'] = owner
            return batch

        def _attach_arrival_history(self, batch, residual: dict) -> dict:
            import torch
            hist = residual.get('arrival_history')
            if hist is None:
                features = build_replan_features(
                    residual,
                    delta_minutes=float(residual.get('delta_minutes', 10.0)),
                    shift_minutes=float(residual.get('shift_minutes', 120.0)),
                    last_replan_time=None,
                    new_requests_since_replan=sum(
                        1 for req in residual.get('active_requests', {}).values()
                        if getattr(req, 'status', '') == 'waiting_for_pickup'
                    ),
                )
                hist = features['arrival_history']
            hist_t = torch.as_tensor(hist, dtype=torch.float32, device=batch['x'].device)
            if hist_t.dim() == 1:
                hist_t = hist_t.view(1, 1, -1)
            elif hist_t.dim() == 2:
                hist_t = hist_t.unsqueeze(0)
            batch = dict(batch)
            batch['arrival_history'] = hist_t
            return batch

        def _node_role(self, node: int, n_depots: int, n_req: int) -> Optional[str]:
            if n_depots <= node < n_depots + n_req:
                return 'pickup'
            if n_depots + n_req <= node < n_depots + 2 * n_req:
                return 'delivery'
            return None

        def _match_request(
            self,
            active_requests: Dict[int, Request],
            node: int,
            leg_type: str,
            vehicle_id: int,
            seen_pickups: set,
            onboard_reqs: set,
        ) -> Optional[Request]:
            matches: List[Request] = []
            for req in active_requests.values():
                target_node = req.pickup_node if leg_type == 'pickup' else req.delivery_node
                if int(target_node) != int(node):
                    continue
                if leg_type == 'pickup':
                    if req.status == 'waiting_for_pickup' and req.req_id not in seen_pickups and (
                        req.assigned_vehicle is None or req.assigned_vehicle == vehicle_id
                    ):
                        matches.append(req)
                else:
                    if req.status in ('onboard', 'delivery_committed') and req.assigned_vehicle == vehicle_id:
                        matches.append(req)
                    elif req.status == 'pickup_committed' and req.assigned_vehicle == vehicle_id and req.req_id in seen_pickups:
                        matches.append(req)
                    elif req.req_id in onboard_reqs:
                        matches.append(req)
                    elif req.req_id in seen_pickups:
                        matches.append(req)
            if not matches:
                return None
            matches.sort(key=lambda r: (r.t_delivery, r.req_id))
            return matches[0]

        def _decode_route_plan(self, residual: dict, decoded_tour) -> Dict[int, List['Leg']]:
            import copy
            from ce_cpdptw_alns import _schedule_route_departure, _simulate_leg

            vehicles = residual['vehicles']
            active = residual['active_requests']
            instance = residual['full_instance']
            n_depots, n_req = int(instance['n_depots']), int(instance['n_req'])
            routes = {vid: [] for vid in vehicles}
            self.last_route_rejections = []
            context = dict(instance, sharing_assignments=dict(instance.get('sharing_assignments', {})))
            for agent_idx, vid in enumerate(self._vehicle_ids(vehicles)):
                shadow = copy.deepcopy(vehicles[vid])
                if shadow.committed_leg is not None:
                    continue
                shadow.current_time = max(shadow.current_time, residual.get('current_time', 0.))
                seen_pickups = set()
                for node in decoded_tour[agent_idx].tolist()[1:]:
                    node = int(node)
                    if node == shadow.current_node:
                        continue
                    if node < n_depots:
                        role, req = 'depot', None
                    else:
                        role = self._node_role(node, n_depots, n_req)
                        if role is None:
                            self.last_route_rejections.append((vid, node, 'invalid_node'))
                            break
                        req = self._match_request(
                            active, node, role, vid, seen_pickups, shadow.onboard_set(),
                        )
                        if req is None:
                            self.last_route_rejections.append((vid, node, 'request_or_carrier'))
                            break
                    leg, following, ok = _simulate_leg(shadow, req, role, node, context)
                    if not ok or leg is None:
                        self.last_route_rejections.append((vid, node, 'physical_or_contract'))
                        break
                    routes[vid].append(leg)
                    shadow = following
                    if role == 'pickup':
                        seen_pickups.add(req.req_id)
                        contract = context.get('sharing_contract')
                        if contract is not None and not contract.reserve(req.req_id, shadow.mode,
                                                                        context['sharing_assignments']):
                            raise ValueError('Native RL plan disagrees with the sharing ledger')
                routes[vid], _ = _schedule_route_departure(routes[vid], active, context)
            return routes

        def _build_initial_visited(self, batch, active_requests, full_instance) -> 'torch.Tensor':
            import torch
            n_total = batch['x'].shape[1]
            n_depots = int(full_instance['n_depots'])
            visited = torch.zeros(1, n_total, device=batch['x'].device)

            # Pre-mark all customers visited; only un-mark nodes the decoder must route.
            # Unrevealed nodes stay masked (adjacency zeroed) so the while-loop terminates.
            visited[0, n_depots:] = 1.0

            for req in active_requests.values():
                p = int(req.pickup_node)
                d = int(req.delivery_node)
                status = req.status
                if status == 'waiting_for_pickup':
                    visited[0, p] = 0.0
                    visited[0, d] = 0.0
                elif status == 'onboard':
                    visited[0, p] = 1.0
                    visited[0, d] = 0.0
                elif status in ('pickup_committed', 'delivery_committed'):
                    visited[0, p] = 1.0
                    visited[0, d] = 1.0

            return visited

        def _augment_dynamic_features(self, batch, active_requests, full_instance, t_current: float):
            import torch
            n_total = batch['x'].shape[1]
            n_depots = int(full_instance['n_depots'])
            n_req = int(full_instance['n_req'])
            _TIME_SCALE = 180.0
            _T_MAX = 120.0
            device = batch['x'].device

            order_age_col = torch.zeros(1, n_total, device=device)
            del_slack_col = torch.zeros(1, n_total, device=device)

            tw = batch['time_window']
            n_vis = 0
            for req in active_requests.values():
                p_idx = int(req.pickup_node)
                d_idx = int(req.delivery_node)
                t_arr = float(req.t_arrival) if hasattr(req, 't_arrival') else 0.0
                age = max(0.0, min(1.0, (t_current - t_arr) / _TIME_SCALE))
                order_age_col[0, p_idx] = age

                t_del = float(tw[0, d_idx].item())
                slack = max(-1.0, min(1.0, (t_del - t_current) / _TIME_SCALE))
                del_slack_col[0, d_idx] = slack
                n_vis += 1

            if self._n_dyn_dims >= 4:
                t_frac_col = torch.full((1, n_total), t_current / _T_MAX, device=device)
                vis_frac_col = torch.full((1, n_total), n_vis / max(n_req, 1), device=device)
                extra = torch.stack([order_age_col, del_slack_col,
                                     t_frac_col, vis_frac_col], dim=-1)
            else:
                extra = torch.stack([order_age_col, del_slack_col], dim=-1)
            batch = dict(batch)
            batch['x'] = torch.cat([batch['x'], extra], dim=-1)
            return batch

        def solve(self, residual: dict) -> Dict[int, List['Leg']]:
            import torch
            full_instance = residual['full_instance']
            active_requests = residual['active_requests']
            vehicles = residual['vehicles']
            t_current = residual.get('current_time', 0.0)
            n_uav, n_adr = self._num_agents(vehicles)
            from reward import native_physical_configuration
            native_physical_configuration(full_instance)

            if full_instance.get('service_policy') == 'mandatory':
                from reward import validate_route_checkpoint
                validate_route_checkpoint(_raw)
            self._validate_request_context(full_instance, active_requests)

            batch = self._batchify_instance(full_instance)
            batch = self._mask_unrevealed_nodes(batch, active_requests, full_instance)
            initial_visited = self._build_initial_visited(batch, active_requests, full_instance)
            batch = self._restore_occupied_origins(batch, full_instance, vehicles, initial_visited)

            # If all customers are pre-visited the decoder would skip entirely and crash.
            n_depots_fi = int(full_instance['n_depots'])
            if bool(initial_visited[0, n_depots_fi:].all()):
                return {vid: [] for vid in vehicles.keys()}

            if self.dynamic_features:
                batch = self._augment_dynamic_features(
                    batch, active_requests, full_instance, float(t_current)
                )
            batch = self._attach_vehicle_state(batch, vehicles, float(t_current))
            batch = self._attach_delivery_owner(batch, active_requests, vehicles, full_instance)
            from reward import tensor_sharing_context
            batch.update(tensor_sharing_context(full_instance, list(range(int(full_instance['n_req']))),
                                                self.device))
            batch['route_scope'] = 'rolling_prefix'
            batch['initial_visited'] = initial_visited
            batch = self._attach_arrival_history(batch, residual)
            if self.policy == 'coop':
                active_count = sum(
                    1 for req in active_requests.values()
                    if req.status in ('waiting_for_pickup', 'pickup_committed', 'onboard')
                )
                n_req_total = int(full_instance['n_req'])
                n_vis = torch.tensor([max(active_count, 1)], dtype=torch.float32, device=self.device)
                batch = attach_window_context(
                    batch,
                    initial_visited,
                    n_vis,
                    n_req_total,
                    n_uav,
                    n_adr,
                    t_current=float(t_current),
                    window_length=10.0,
                )
            def _decode_once(greedy: bool) -> "torch.Tensor":
                if self.policy == 'coop':
                    out = self.actor(batch, n_uav, n_adr, greedy=greedy,
                                     initial_visited=initial_visited)
                    return out['tour_idx']
                tour, _, _ = self.actor(
                    batch, n_uav, n_adr,
                    greedy=greedy, checkpoint_encoder=False,
                    training=False, initial_visited=initial_visited,
                    delivery_owner=batch.get('delivery_owner'),
                )
                self.last_decode_status = getattr(self.actor.decoder, 'last_decode_status',
                                                  {'reason': 'external_policy_status_unavailable'})
                return tour

            def _candidate_plan(tour):
                if tour.dim() != 3 or tour.size(0) != 1:
                    raise RuntimeError(f'Unexpected actor output shape: {tuple(tour.shape)}')
                return self._decode_route_plan(residual, tour[0].detach().cpu())

            with torch.no_grad():
                if self.n_samples <= 1:
                    return _candidate_plan(_decode_once(greedy=True))
                plan, self.last_candidate_scores = select_route_candidate(
                    (_candidate_plan(_decode_once(greedy=False)) for _ in range(self.n_samples)),
                    vehicles, active_requests, full_instance,
                )
                return plan

    solver = RLSolver(actor, device, dynamic_features=_dynamic_features,
                      input_node_dim=_input_node_dim, policy=_policy)
    solver.n_samples = max(1, int(n_samples))
    return solver
