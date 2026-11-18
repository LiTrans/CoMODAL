import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset
from torch_geometric.loader import DataLoader
import os
import random
import math
from reward import (ALPHA_U as _ALPHA_U, ALPHA_DELIV_BONUS as _ALPHA_DELIV_BONUS,
                    ALPHA_1 as _ALPHA_1, ALPHA_2 as _ALPHA_2,
                    ALPHA_E as _ALPHA_E, ALPHA_P as _ALPHA_P, ALPHA_D as _ALPHA_D)

INF = float('inf')
SEED_BASE = 42
REQUEST_PAIRING_VERSION = 'same-carrier-endpoints-1'

from model_config import (
    SCALE_M_PER_COORD, UAV_CRUISE_MPS, ADR_CRUISE_MPS,
    UAV_BATTERY_KJ, ADR_BATTERY_KJ, ENERGY_MODEL_VERSION, GENERATOR_VERSION,
    UAV_CAPACITY_KG, ADR_CAPACITY_KG,
    MODEL_VERSION, UAV_PROFILE, ADR_PROFILE,
)

V_UAV_COORD_PER_MIN = UAV_CRUISE_MPS * 60.0 / SCALE_M_PER_COORD
V_ADR_COORD_PER_MIN = ADR_CRUISE_MPS * 60.0 / SCALE_M_PER_COORD
UAV_LAND_TAKEOFF_MIN = 2.0
V_UAV_MIN_PICKUP = V_UAV_COORD_PER_MIN
V_ADR_MIN_PICKUP = min(2.0, ADR_CRUISE_MPS) * 60.0 / SCALE_M_PER_COORD

def _euclidean_dist_matrix(coords: np.ndarray) -> np.ndarray:
    diff = coords[:, None, :] - coords[None, :, :]
    return np.sqrt((diff ** 2).sum(-1)).astype(np.float32)

def _segment_intersects_circle(p1, p2, center, radius):
    d = p2 - p1
    f = p1 - center
    a = float(np.dot(d, d))
    b = float(2 * np.dot(f, d))
    c = float(np.dot(f, f) - radius ** 2)
    disc = b ** 2 - 4 * a * c
    if disc < 0 or a == 0:
        return False
    disc = math.sqrt(disc)
    t1 = (-b - disc) / (2 * a)
    t2 = (-b + disc) / (2 * a)
    return (0 <= t1 <= 1) or (0 <= t2 <= 1)

def _segments_intersect_circle_batch(p1: np.ndarray, p2: np.ndarray,
                                     center: np.ndarray, radius: float) -> np.ndarray:
    d = p2 - p1
    f = p1 - center
    a = (d * d).sum(-1)
    b = 2.0 * (f * d).sum(-1)
    c = (f * f).sum(-1) - radius ** 2
    disc = b ** 2 - 4.0 * a * c
    valid = (disc >= 0) & (a > 1e-12)
    disc_safe = np.where(valid, disc, 0.0)
    disc_sqrt = np.sqrt(disc_safe)
    denom = 2.0 * a + 1e-12
    t1 = np.where(valid, (-b - disc_sqrt) / denom, 2.0)
    t2 = np.where(valid, (-b + disc_sqrt) / denom, 2.0)
    return valid & (((t1 >= 0) & (t1 <= 1)) | ((t2 >= 0) & (t2 <= 1)))

def _segments_intersect(p1, p2, p3, p4):
    def _cross(o, a, b):
        return (a[0] - o[0]) * (b[1] - o[1]) - (a[1] - o[1]) * (b[0] - o[0])
    d1, d2 = _cross(p3, p4, p1), _cross(p3, p4, p2)
    d3, d4 = _cross(p1, p2, p3), _cross(p1, p2, p4)
    return (d1 * d2 < 0) and (d3 * d4 < 0)

def vectorized_floyd_warshall(dist: np.ndarray) -> np.ndarray:
    n = dist.shape[0]
    for k in range(n):
        dist = np.minimum(dist, dist[:, k, np.newaxis] + dist[k, :])
    return dist

def _synthetic_no_fly_circle(coords: np.ndarray, rng: np.random.Generator,
                              nfz_radius: float = None):
    cx, cy = float(rng.uniform(0, 5)), float(rng.uniform(0, 5))
    radius = nfz_radius if nfz_radius is not None else float(rng.uniform(0.4, 0.9))
    dists = np.sqrt((coords[:, 0] - cx) ** 2 + (coords[:, 1] - cy) ** 2)
    return dists > radius, cx, cy, radius

def _adr_barrier_threshold(coords: np.ndarray, rng: np.random.Generator):
    pct_blocked = rng.uniform(0.15, 0.25)
    return float(np.percentile(coords[:, 1], 100.0 * (1.0 - pct_blocked)))

def _synthetic_building_clusters(coords: np.ndarray, rng: np.random.Generator):
    n_clusters = int(rng.integers(3, 6))
    clusters = []
    for _ in range(n_clusters):
        cx = float(rng.uniform(0.3, 4.7))
        cy = float(rng.uniform(0.3, 4.7))
        radius = float(rng.uniform(0.2, 0.5))
        clusters.append((np.array([cx, cy], dtype=np.float32), radius))
    return clusters

def static_signed_margin(t_pickup: np.ndarray, dist_matrix: np.ndarray, v_max: float) -> np.ndarray:
    n = len(t_pickup)
    tp_i = t_pickup[:, None].repeat(n, axis=1)
    tp_j = t_pickup[None, :].repeat(n, axis=0)
    return (tp_j - tp_i - dist_matrix / v_max).astype(np.float32)


def _create_spatial_template(n_req, n_depots_uav, n_depots_adr,
                             rng: np.random.Generator,
                             lower=0.0, high=5.0,
                             spatial_cluster_k=0,
                             nfz_radius: float = None):
    n_depots = n_depots_uav + n_depots_adr
    n_nodes = n_req * 2
    n_total = n_depots + n_nodes

    coords = rng.uniform(lower, high, size=(n_total, 2)).astype(np.float32)
    centers = None
    if spatial_cluster_k > 0:
        k = spatial_cluster_k
        centers = rng.uniform(lower + 0.5, high - 0.5, size=(k, 2)).astype(np.float32)
        sigma = (high - lower) * 0.15
        assigns = rng.integers(0, k, size=n_nodes)
        clustered = (centers[assigns] + rng.normal(0.0, sigma, size=(n_nodes, 2))).astype(np.float32)
        coords[n_depots:] = np.clip(clustered, lower, high)

    _nfz_r = nfz_radius
    if _nfz_r is None:
        _base_r = float(rng.uniform(0.4, 0.9))
        _nfz_r = float(np.clip(_base_r * math.sqrt(n_req / 5.0), 0.4, 3.0))
    _, nfz_cx, nfz_cy, nfz_radius = _synthetic_no_fly_circle(coords, rng, _nfz_r)
    adr_threshold = _adr_barrier_threshold(coords, rng)
    road_edges = np.isfinite(_build_road_network(coords, rng))
    layout = dict(lower=lower, high=high, centers=centers, adr_threshold=adr_threshold,
                  nfz=(nfz_cx, nfz_cy, nfz_radius), road_edges=road_edges,
                  buildings=_synthetic_building_clusters(coords, rng))
    return _render_spatial_template(coords, n_depots_uav, n_depots_adr, layout)


def _site_access(coords, layout):
    cx, cy, radius = layout['nfz']
    outside_nfz = np.sqrt((coords[:, 0] - cx) ** 2 + (coords[:, 1] - cy) ** 2) > radius
    return outside_nfz, coords[:, 1] <= layout['adr_threshold']


def _render_spatial_template(coords, n_depots_uav, n_depots_adr, layout):
    n_depots = n_depots_uav + n_depots_adr
    n_total = len(coords)
    acc_uav, acc_adr = _site_access(coords, layout)
    d_adr = vectorized_floyd_warshall(np.where(
        layout['road_edges'], _euclidean_dist_matrix(coords), INF,
    ))
    d_uav = _euclidean_dist_matrix(coords)

    nfz_cx, nfz_cy, nfz_radius = layout['nfz']
    nfz_center = np.array([nfz_cx, nfz_cy], dtype=np.float32)
    ii, jj = np.meshgrid(np.arange(n_depots, n_total),
                         np.arange(n_depots, n_total), indexing='ij')
    ii_f, jj_f = ii.flatten(), jj.flatten()
    off_diag = ii_f != jj_f
    ii_f, jj_f = ii_f[off_diag], jj_f[off_diag]

    blocked = _segments_intersect_circle_batch(
        coords[ii_f], coords[jj_f], nfz_center, nfz_radius
    )
    d_uav[ii_f[blocked], jj_f[blocked]] = INF

    for _bc_center, _bc_radius in layout['buildings']:
        _bld_blocked = _segments_intersect_circle_batch(
            coords[ii_f], coords[jj_f], _bc_center, _bc_radius
        )
        d_uav[ii_f[_bld_blocked], jj_f[_bld_blocked]] = INF

    d_uav = vectorized_floyd_warshall(d_uav)
    # Force depot accessibility for each fleet's own mode (depots are infrastructure
    # that must always be enter/leave-able by the fleet that owns them).
    acc_uav[:n_depots_uav] = True
    acc_adr[n_depots_uav:n_depots] = True

    return {
        'coords': coords,
        'acc_uav': acc_uav,
        'acc_adr': acc_adr,
        'd_uav': d_uav.astype(np.float32),
        'd_adr': d_adr.astype(np.float32),
        'nfz': (nfz_cx, nfz_cy, nfz_radius),
        'layout': layout,
    }


def request_pair_modes(template, demand, n_depots_uav, n_depots_adr, n_uav, n_adr):
    """Return a [mode, request] table for same-carrier service eligibility."""
    n_depots = n_depots_uav + n_depots_adr
    pickup = np.arange(n_depots, n_depots + len(demand))
    delivery = pickup + len(demand)
    eligible = np.zeros((2, len(demand)), dtype=bool)
    for col, (mode, start, stop, fleet_count, capacity) in enumerate((
        ('uav', 0, n_depots_uav, n_uav, UAV_CAPACITY_KG),
        ('adr', n_depots_uav, n_depots, n_adr, ADR_CAPACITY_KG),
    )):
        if fleet_count <= 0 or start >= stop:
            continue
        access, distance = template['acc_' + mode], template['d_' + mode]
        reach_pickup = (distance[start:stop, pickup] < 1e9).any(axis=0)
        eligible[col] = (access[pickup] & access[delivery] & (demand <= capacity)
                         & reach_pickup & (distance[pickup, delivery] < 1e9))
    return eligible


def resample_request_pairs(template, demand, n_depots_uav, n_depots_adr, n_uav, n_adr, rng):
    """Redraw only impossible pairs on the same map and rebuild their distances."""
    demand = np.asarray(demand)
    capacities = []
    if n_uav and n_depots_uav:
        capacities.append(UAV_CAPACITY_KG)
    if n_adr and n_depots_adr:
        capacities.append(ADR_CAPACITY_KG)
    if (demand.size and (not capacities or not np.isfinite(demand).all()
                        or (demand <= 0).any() or (demand > max(capacities)).any())):
        raise ValueError('A request payload is outside the available fleet capacity')
    args = (demand, n_depots_uav, n_depots_adr, n_uav, n_adr)
    modes = request_pair_modes(template, *args)
    if modes.any(axis=0).all():
        return template, modes
    layout = template.get('layout')
    if layout is None:
        raise ValueError('Cannot resample incompatible pairs without the spatial map layout')
    n_depots = n_depots_uav + n_depots_adr
    coords = template['coords'].copy()
    for _ in range(256):
        bad = np.flatnonzero(~modes.any(axis=0))
        pair_nodes = n_depots + np.column_stack((bad, bad + len(demand)))
        lower, high, centers = layout['lower'], layout['high'], layout['centers']
        if centers is None:
            sites = rng.uniform(lower, high, size=(len(bad), 2, 2))
        else:
            chosen = rng.integers(len(centers), size=(len(bad), 2))
            sites = centers[chosen] + rng.normal(0., (high - lower) * .15, size=(len(bad), 2, 2))
            sites = np.clip(sites, lower, high)
        sites = sites.astype(np.float32)
        uav, adr = (m.reshape(-1, 2).all(axis=1) for m in _site_access(sites.reshape(-1, 2), layout))
        viable = ((uav & bool(n_uav and n_depots_uav) & (demand[bad] <= UAV_CAPACITY_KG))
                  | (adr & bool(n_adr and n_depots_adr) & (demand[bad] <= ADR_CAPACITY_KG)))
        if not viable.any():
            continue
        coords[pair_nodes[viable]] = sites[viable]
        # Rendering keeps the sampled restrictions and road links. It only updates
        # the geometry and shortest paths after moving the selected endpoints.
        template = _render_spatial_template(coords.copy(), n_depots_uav, n_depots_adr, layout)
        modes = request_pair_modes(template, *args)
        if modes.any(axis=0).all():
            return template, modes
    raise ValueError('Could not generate same-carrier endpoint pairs for this map')


def create_instance(n_req, n_uav, n_adr, n_depots_uav, n_depots_adr,
                    rng: np.random.Generator, lower=0.0, high=5.0,
                    demand_range=(1.0, 10.0),
                    wind_speed_range=(0.0, 12.0),
                    tw_slack_mean=30.0, tw_slack_std=5.0,
                    tw_slack_clip=(15.0, 60.0),
                    temporal_peaks=False,
                    temporal_peaks_intensity: float = 1.0,
                    spatial_cluster_k=0,
                    demand_heavy_tail=False,
                    nfz_radius: float = None,
                    battery_fraction: float = 1.0,
                    mode_bias_frac: float = 0.0,
                    map_template: dict = None, wind_override=None):
    n_depots = n_depots_uav + n_depots_adr
    n_nodes = n_req * 2
    n_total = n_depots + n_nodes

    if map_template is None:
        map_template = _create_spatial_template(
            n_req, n_depots_uav, n_depots_adr, rng,
            lower=lower, high=high,
            spatial_cluster_k=spatial_cluster_k,
            nfz_radius=nfz_radius,
        )
    # Heavy-tail demand: lognormal so a few orders are much larger than the rest.
    if demand_heavy_tail:
        raw = rng.lognormal(mean=0.8, sigma=0.9, size=n_req).astype(np.float32)
        demand_vals = np.clip(raw, demand_range[0], demand_range[1])
    else:
        demand_vals = rng.uniform(demand_range[0], demand_range[1], size=n_req).astype(np.float32)
    demand = np.zeros(n_total, dtype=np.float32)
    demand[n_depots:n_depots + n_req] = demand_vals
    demand[n_depots + n_req:] = -demand_vals

    shift_minutes = 120.0
    # Peak times chosen by the generator are exposed via the returned
    # instance so the dispatcher (and downstream eval) can compute the
    # trigger-lead-time-to-burst metric without re-deriving them.
    arrival_peaks: np.ndarray = np.zeros(0, dtype=np.float32)
    if temporal_peaks:
        # Intensity selects a peak profile.  intensity=1.0 reproduces the
        # original (mild) bursts; >1 yields sharper, more frequent, more
        # concentrated peaks that the rolling-horizon gate must time
        # precisely.  Profile parameters chosen so that intensity ∈
        # {1, 2, 3} approximately doubles the peak-to-trough arrival-rate
        # ratio at each step.
        intensity = float(max(0.0, temporal_peaks_intensity))
        if intensity <= 1.0:
            n_peaks_min, n_peaks_max = 2, 4
            peak_window = (20.0, 100.0)
            burst_frac = 0.40
            peak_sd = 10.0 / 3.0
            quiet_windows = 0
        elif intensity <= 2.0:
            n_peaks_min, n_peaks_max = 3, 6
            peak_window = (15.0, 110.0)
            burst_frac = 0.55
            peak_sd = 2.0
            quiet_windows = 0
        else:
            n_peaks_min, n_peaks_max = 4, 7
            peak_window = (10.0, 115.0)
            burst_frac = 0.70
            peak_sd = 1.25
            quiet_windows = 1
        n_peaks = int(rng.integers(n_peaks_min, n_peaks_max + 1))
        peak_times = rng.uniform(peak_window[0], peak_window[1], size=n_peaks)
        n_peak = max(1, int(round(n_req * burst_frac)))
        n_bg = max(0, n_req - n_peak)
        bg_lo, bg_hi = 0.0, shift_minutes * 0.95
        # Optional quiet (zero-arrival) window keeps a stretch of the shift
        # request-free, which the gate should learn to hold through.
        if quiet_windows > 0 and n_bg > 0:
            q_lo = rng.uniform(20.0, shift_minutes - 35.0)
            q_hi = q_lo + rng.uniform(15.0, 25.0)
            bg = []
            while len(bg) < n_bg:
                candidates = rng.uniform(bg_lo, bg_hi, size=n_bg - len(bg))
                bg.extend(c for c in candidates if not (q_lo <= c <= q_hi))
            bg = np.sort(np.asarray(bg[:n_bg], dtype=np.float32))
        elif n_bg > 0:
            bg = np.sort(rng.uniform(bg_lo, bg_hi, size=n_bg))
        else:
            bg = np.zeros(0, dtype=np.float32)
        peak_src = rng.integers(0, n_peaks, size=n_peak)
        peak = np.clip(rng.normal(peak_times[peak_src], peak_sd),
                       0.0, shift_minutes - 10.0)
        t_pickup = np.sort(np.concatenate([bg, peak])).astype(np.float32)
        arrival_peaks = np.sort(peak_times.astype(np.float32))
    else:
        inter_arrival = rng.exponential(scale=shift_minutes / n_req, size=n_req)
        t_pickup = np.cumsum(inter_arrival).astype(np.float32)
        t_pickup = np.clip(t_pickup, 0, shift_minutes - 10)
    delivery_slack = rng.normal(loc=tw_slack_mean, scale=tw_slack_std, size=n_req).astype(np.float32)
    delivery_slack = np.clip(delivery_slack, tw_slack_clip[0], tw_slack_clip[1])

    # Mode-bias: correlate demand weight with deadline tightness so the model
    # sees a meaningful signal for UAV vs ADR specialisation.
    # Light orders → tight TW  (drone-ideal: fast, can make it; robot may not)
    # Heavy orders → relaxed TW (robot-ideal: carries more, doesn't need to rush)
    if mode_bias_frac > 0.0:
        n_half = max(1, int(n_req * mode_bias_frac * 0.5))
        sorted_by_demand = np.argsort(demand_vals)
        light_idx = sorted_by_demand[:n_half]
        heavy_idx = sorted_by_demand[-n_half:]
        delivery_slack[light_idx] = np.clip(
            delivery_slack[light_idx] * 0.55,
            tw_slack_clip[0] * 0.4,
            tw_slack_clip[1] * 0.5,
        )
        delivery_slack[heavy_idx] = np.clip(
            delivery_slack[heavy_idx] * 1.6,
            tw_slack_clip[0] * 1.3,
            tw_slack_clip[1] * 2.0,
        )

    t_delivery = (t_pickup + delivery_slack).astype(np.float32)

    # Request ids are used as prefix-ordered reveal sets by build_dynamic_sub_batch
    # and by several rolling-horizon evaluators.  Random preparation times can make
    # t_arrival non-monotone even when pickup targets are sorted, so reorder the
    # paired pickup/delivery arrays by actual reveal time once, before building
    # distances, accessibility, edge features, and node features.
    prep_time = rng.uniform(5.0, 20.0, size=n_req).astype(np.float32)
    t_arrival = np.maximum(0.0, t_pickup - prep_time)
    order = np.argsort(t_arrival)
    if not np.all(order == np.arange(n_req)):
        demand_vals = demand_vals[order]
        t_pickup = t_pickup[order]
        t_delivery = t_delivery[order]
        delivery_slack = delivery_slack[order]
        t_arrival = t_arrival[order]
        demand = np.zeros(n_total, dtype=np.float32)
        demand[n_depots:n_depots + n_req] = demand_vals
        demand[n_depots + n_req:] = -demand_vals

    wind_dir = float(rng.uniform(0, 2 * math.pi))
    wind_mag = float(rng.uniform(wind_speed_range[0], wind_speed_range[1]))
    if wind_override is not None:
        wind_mag, wind_dir = map(float, wind_override)
        if not np.isfinite([wind_mag, wind_dir]).all() or wind_mag < 0:
            raise ValueError('Invalid prescribed wind')

    template, modes = resample_request_pairs(
        map_template, demand_vals, n_depots_uav, n_depots_adr, n_uav, n_adr, rng,
    )
    coords, d_uav, d_adr = template['coords'], template['d_uav'], template['d_adr']
    acc_uav, acc_adr = template['acc_uav'].copy(), template['acc_adr'].copy()
    for access, eligible in zip((acc_uav, acc_adr), modes):
        access[n_depots:] = np.tile(eligible, 2)

    from energy import uav_energy_matrix, adr_energy_matrix
    wind_vec = np.array([wind_mag * math.cos(wind_dir), wind_mag * math.sin(wind_dir)], dtype=np.float32)
    Q_adr = ADR_CAPACITY_KG
    e_uav = uav_energy_matrix(
        d_uav * SCALE_M_PER_COORD, demand_vals, demand, n_depots, n_req, wind_vec, UAV_CRUISE_MPS,
        coordinates=coords,
    )
    e_adr = adr_energy_matrix(
        d_adr * SCALE_M_PER_COORD, demand_vals, demand, n_depots, n_req, ADR_CRUISE_MPS
    )

    tp_full = np.zeros(n_total, dtype=np.float32)
    tp_full[n_depots:n_depots + n_req] = t_pickup
    margin_uav = static_signed_margin(tp_full, d_uav, V_UAV_COORD_PER_MIN)
    margin_adr = static_signed_margin(tp_full, d_adr, V_ADR_COORD_PER_MIN)

    DIST_SCALE = high * math.sqrt(2)
    ENERGY_SCALE = 300_000.0

    tp_i = tp_full[:, None]
    tp_j = tp_full[None, :]
    t_gap = np.abs(tp_i - tp_j) / shift_minutes

    edge_attr_uav = np.stack([
        np.clip(margin_uav.reshape(-1) / shift_minutes, -2.0, 2.0),
        np.clip(d_uav.reshape(-1) / DIST_SCALE, 0.0, 2.0),
        np.clip(e_uav.reshape(-1) / ENERGY_SCALE, 0.0, 1.0),
        t_gap.reshape(-1),
    ], axis=-1).astype(np.float32)

    edge_attr_adr = np.stack([
        np.clip(margin_adr.reshape(-1) / shift_minutes, -2.0, 2.0),
        np.clip(d_adr.reshape(-1) / DIST_SCALE, 0.0, 2.0),
        np.clip(e_adr.reshape(-1) / ENERGY_SCALE, 0.0, 1.0),
        t_gap.reshape(-1),
    ], axis=-1).astype(np.float32)

    node_type = np.zeros((n_total, 3), dtype=np.float32)
    node_type[:n_depots, 0] = 1.0
    node_type[n_depots:n_depots + n_req, 1] = 1.0
    node_type[n_depots + n_req:, 2] = 1.0

    t_pickup_feat = np.zeros(n_total, dtype=np.float32)
    t_pickup_feat[n_depots:n_depots + n_req] = t_pickup
    t_delivery_feat = np.zeros(n_total, dtype=np.float32)
    t_delivery_feat[n_depots + n_req:] = t_delivery

    DEMAND_MAX = max(float(demand_range[1]), Q_adr)
    delivery_slack_node = np.zeros(n_total, dtype=np.float32)
    delivery_slack_node[n_depots:n_depots + n_req] = delivery_slack / shift_minutes
    delivery_slack_node[n_depots + n_req:]         = delivery_slack / shift_minutes

    COORD_SCALE = high
    TIME_SCALE = shift_minutes + 60.0

    node_feat = np.concatenate([
        coords / COORD_SCALE,
        demand[:, None] / DEMAND_MAX,
        t_pickup_feat[:, None] / TIME_SCALE,
        t_delivery_feat[:, None] / TIME_SCALE,
        node_type,
        acc_uav[:, None].astype(np.float32),
        acc_adr[:, None].astype(np.float32),
        delivery_slack_node[:, None],
    ], axis=-1).astype(np.float32)

    # Gate row AND column mode-accessibility: acc_mode must be True both at
    # the source and destination, else a wrong-mode entry can strand the vehicle.
    _acc_uav_f = acc_uav.astype(np.float32)
    _acc_adr_f = acc_adr.astype(np.float32)
    mask_adj_uav = (d_uav < 1e9).astype(np.float32) * _acc_uav_f[:, None] * _acc_uav_f[None, :]
    mask_adj_adr = (d_adr < 1e9).astype(np.float32) * _acc_adr_f[:, None] * _acc_adr_f[None, :]

    edges_index = np.stack(np.indices((n_total, n_total)).reshape(2, -1), axis=0)

    capacity = np.hstack([
        np.ones(n_uav, dtype=np.float32) * UAV_CAPACITY_KG,
        np.ones(n_adr, dtype=np.float32) * Q_adr,
    ])
    BATTERY_UAV = UAV_BATTERY_KJ
    BATTERY_ADR = ADR_BATTERY_KJ
    battery = np.hstack([
        np.ones(n_uav, dtype=np.float32) * BATTERY_UAV * battery_fraction,
        np.ones(n_adr, dtype=np.float32) * BATTERY_ADR * battery_fraction,
    ])

    return {
        'x': torch.tensor(node_feat, dtype=torch.float32),
        'coordinates': torch.tensor(coords, dtype=torch.float32),
        'energy_model_version': ENERGY_MODEL_VERSION,
        'generator_version': GENERATOR_VERSION,
        'request_pairing_version': REQUEST_PAIRING_VERSION,
        'model_version': MODEL_VERSION,
        'uav_hardware_profile': UAV_PROFILE,
        'adr_hardware_profile': ADR_PROFILE,
        'battery_unit': 'kJ',
        'edge_index': torch.tensor(edges_index, dtype=torch.long),
        'edge_attr_uav': torch.tensor(edge_attr_uav, dtype=torch.float32),
        'edge_attr_adr': torch.tensor(edge_attr_adr, dtype=torch.float32),
        'mask_adjacency_uav': torch.tensor(mask_adj_uav.reshape(-1, 1), dtype=torch.float32),
        'mask_adjacency_adr': torch.tensor(mask_adj_adr.reshape(-1, 1), dtype=torch.float32),
        'demand': torch.tensor(demand, dtype=torch.float32).unsqueeze(-1),
        'capacity': torch.tensor(capacity, dtype=torch.float32),
        'battery': torch.tensor(battery, dtype=torch.float32),
        'edge_attr_d': torch.tensor(d_uav.reshape(-1, 1), dtype=torch.float32),
        'edge_attr_r': torch.tensor(d_adr.reshape(-1, 1), dtype=torch.float32),
        'time_window': torch.tensor(
            np.concatenate([np.zeros(n_depots), t_pickup, t_delivery])[:, None],
            dtype=torch.float32
        ),
        'wind': torch.tensor([wind_mag, wind_dir], dtype=torch.float32),
        'n_depots': n_depots,
        'n_depots_uav': n_depots_uav,
        'n_req': n_req,
        't_arrival': torch.tensor(t_arrival, dtype=torch.float32),
        # Peak centres (minutes) when temporal_peaks is on, empty otherwise.
        # The dispatcher uses these for the trigger-lead-time-to-burst metric.
        'arrival_peaks': torch.tensor(arrival_peaks, dtype=torch.float32),
    }

def _build_road_network(coords: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    n = len(coords)
    dist = np.full((n, n), INF, dtype=np.float32)
    np.fill_diagonal(dist, 0)
    for i in range(n - 1):
        n_neighbors = max(1, (n - i - 1) // 2)
        neighbors = rng.choice(range(i + 1, n), size=n_neighbors, replace=False)
        dists_ij = np.linalg.norm(coords[i] - coords[neighbors], axis=1)
        dist[i, neighbors] = dists_ij
        dist[neighbors, i] = dists_ij
    return dist

class CEPDPTWDataset(Dataset):
    def __init__(self, n_req, n_uav, n_adr, n_depots_uav, n_depots_adr,
                 num_samples, seed=None,
                 tw_slack_mean=30.0, tw_slack_std=5.0, tw_slack_clip=(15.0, 60.0),
                 temporal_peaks=False, spatial_cluster_k=0, demand_heavy_tail=False,
                 mode_bias_frac=0.0):
        self.n_req = n_req
        self.n_uav = n_uav
        self.n_adr = n_adr
        self.n_depots_uav = n_depots_uav
        self.n_depots_adr = n_depots_adr
        self.num_samples = num_samples
        self.config = dict(
            energy_model_version=ENERGY_MODEL_VERSION, generator_version=GENERATOR_VERSION,
            request_pairing_version=REQUEST_PAIRING_VERSION,
            model_version=MODEL_VERSION, uav_hardware_profile=UAV_PROFILE, adr_hardware_profile=ADR_PROFILE,
            n_req=n_req, n_uav=n_uav, n_adr=n_adr,
            n_depots_uav=n_depots_uav, n_depots_adr=n_depots_adr,
            num_samples=num_samples, seed=seed,
            tw_slack_mean=tw_slack_mean, tw_slack_std=tw_slack_std,
            tw_slack_clip=tuple(tw_slack_clip),
            temporal_peaks=temporal_peaks,
            spatial_cluster_k=spatial_cluster_k,
            demand_heavy_tail=demand_heavy_tail,
            mode_bias_frac=mode_bias_frac,
        )
        base_seed = seed if seed is not None else SEED_BASE
        self.data = [
            create_instance(n_req, n_uav, n_adr, n_depots_uav, n_depots_adr,
                            np.random.default_rng(base_seed + i),
                            tw_slack_mean=tw_slack_mean,
                            tw_slack_std=tw_slack_std,
                            tw_slack_clip=tw_slack_clip,
                            temporal_peaks=temporal_peaks,
                            spatial_cluster_k=spatial_cluster_k,
                            demand_heavy_tail=demand_heavy_tail,
                            mode_bias_frac=mode_bias_frac)
            for i in range(num_samples)
        ]

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        # Match CEPDPTWStreamDataset: strip the variable-length arrival_peaks field,
        # which default_collate cannot stack across instances with different
        # temporal_peaks settings. dispatch_sim.py:421 handles the missing key via
        # .get(..., None) and only needs it when invoked outside the training path.
        instance = dict(self.data[idx])
        instance.pop('arrival_peaks', None)
        return instance

class CEPDPTWStreamDataset(Dataset):
    def __init__(self, n_req, n_uav, n_adr, n_depots_uav, n_depots_adr,
                 num_samples, seed=None,
                 tw_slack_mean=30.0, tw_slack_std=5.0, tw_slack_clip=(15.0, 60.0),
                 temporal_peaks=False, spatial_cluster_k=0, demand_heavy_tail=False,
                 mix_hard_frac=0.0, mode_bias_frac=0.0, map_pool_size=0):
        self.n_req = n_req
        self.n_uav = n_uav
        self.n_adr = n_adr
        self.n_depots_uav = n_depots_uav
        self.n_depots_adr = n_depots_adr
        self.num_samples = num_samples
        self.base_seed = seed if seed is not None else SEED_BASE
        self.tw_slack_mean     = tw_slack_mean
        self.tw_slack_std      = tw_slack_std
        self.tw_slack_clip     = tw_slack_clip
        self.temporal_peaks    = temporal_peaks
        self.spatial_cluster_k = spatial_cluster_k
        self.demand_heavy_tail = demand_heavy_tail
        self.mix_hard_frac     = mix_hard_frac
        self.mode_bias_frac    = mode_bias_frac
        self.map_pool_size     = int(map_pool_size or 0)
        self.map_pool = None
        if self.map_pool_size > 0:
            # Spatial templates contain the expensive pieces: coordinates,
            # accessibility masks, road shortest paths, and UAV shortest paths.
            # Order demand/time/wind are still regenerated in __getitem__.
            self.map_pool = []
            for i in range(self.map_pool_size):
                rng_map = np.random.default_rng(self.base_seed + 2_000_000_000 + i)
                if mix_hard_frac > 0.0:
                    sck_map = int(rng_map.integers(2, 5)) if rng_map.random() < mix_hard_frac else 0
                else:
                    sck_map = spatial_cluster_k
                self.map_pool.append(
                    _create_spatial_template(
                        n_req, n_depots_uav, n_depots_adr,
                        rng_map,
                        spatial_cluster_k=sck_map,
                    )
                )

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        # Use a separate RNG for sampling instance flags so the instance-generation
        # RNG (base_seed + idx) is unaffected whether mix_hard_frac is on or off.
        rng_flags = np.random.default_rng(self.base_seed + idx + 1_000_000_000)
        rng_inst  = np.random.default_rng(self.base_seed + idx)
        map_template = None

        if self.mix_hard_frac > 0.0:
            f = self.mix_hard_frac
            tp  = bool(rng_flags.random() < f)
            sck = int(rng_flags.integers(2, 5)) if rng_flags.random() < f else 0
            dht = bool(rng_flags.random() < f)
            # Tighten TW proportionally to how many hard flags fired
            n_hard = int(tp) + int(sck > 0) + int(dht)
            if n_hard > 0:
                blend = n_hard / 3.0
                tw_mean = self.tw_slack_mean * (1.0 - blend) + 10.0 * blend
                tw_clip_lo = self.tw_slack_clip[0] * (1.0 - blend) + 5.0 * blend
                tw_clip_hi = self.tw_slack_clip[1] * (1.0 - blend) + 20.0 * blend
            else:
                tw_mean   = self.tw_slack_mean
                tw_clip_lo, tw_clip_hi = self.tw_slack_clip[0], self.tw_slack_clip[1]
        else:
            tp, sck, dht = self.temporal_peaks, self.spatial_cluster_k, self.demand_heavy_tail
            tw_mean   = self.tw_slack_mean
            tw_clip_lo, tw_clip_hi = self.tw_slack_clip[0], self.tw_slack_clip[1]
        if self.map_pool:
            map_template = self.map_pool[idx % self.map_pool_size]

        instance = create_instance(self.n_req, self.n_uav, self.n_adr,
                                   self.n_depots_uav, self.n_depots_adr, rng_inst,
                                   tw_slack_mean=tw_mean,
                                   tw_slack_std=self.tw_slack_std,
                                   tw_slack_clip=(tw_clip_lo, tw_clip_hi),
                                   temporal_peaks=tp,
                                   spatial_cluster_k=sck,
                                   demand_heavy_tail=dht,
                                   mode_bias_frac=self.mode_bias_frac,
                                   map_template=map_template)
        # arrival_peaks has instance-dependent length (0 when temporal_peaks is off,
        # 2-7 when on). default_collate cannot stack mixed-length tensors across a
        # batch, which breaks training when mix_hard_frac > 0 randomizes temporal_peaks
        # per instance. The field is only consumed by dispatch_sim's burst-lead-time
        # metric via instances constructed directly via create_instance(), not via the
        # training dataset. dispatch_sim handles the missing-key case with .get(..., None).
        instance.pop('arrival_peaks', None)
        return instance

def creat_data(n_req, n_uav, n_adr, n_depots_uav, n_depots_adr,
               num_samples, batch_size=512, num_workers=0,
               use_cache=False, cache_file='dataset_cache.pkl', seed=None,
               shuffle=None, streaming=False,
               distributed=False, rank=0, world_size=1,
               tw_slack_mean=30.0, tw_slack_std=5.0, tw_slack_clip=(15.0, 60.0),
               temporal_peaks=False, spatial_cluster_k=0, demand_heavy_tail=False,
               mix_hard_frac=0.0, mode_bias_frac=0.0, map_pool_size=0):
    _tw_kwargs = dict(tw_slack_mean=tw_slack_mean, tw_slack_std=tw_slack_std,
                      tw_slack_clip=tw_slack_clip,
                      temporal_peaks=temporal_peaks,
                      spatial_cluster_k=spatial_cluster_k,
                      demand_heavy_tail=demand_heavy_tail)
    if streaming:
        dataset = CEPDPTWStreamDataset(n_req, n_uav, n_adr, n_depots_uav, n_depots_adr,
                                       num_samples, seed=seed,
                                       mix_hard_frac=mix_hard_frac,
                                       mode_bias_frac=mode_bias_frac,
                                       map_pool_size=map_pool_size,
                                       **_tw_kwargs)
    else:
        dataset = CEPDPTWDataset(n_req, n_uav, n_adr, n_depots_uav, n_depots_adr,
                                 num_samples, seed=seed,
                                 mode_bias_frac=mode_bias_frac,
                                 **_tw_kwargs)

    if distributed and world_size > 1:
        from torch.utils.data.distributed import DistributedSampler
        sampler = DistributedSampler(dataset, num_replicas=world_size, rank=rank,
                                     shuffle=(shuffle is not False))
        return DataLoader(dataset, batch_size=batch_size, sampler=sampler,
                          num_workers=num_workers, pin_memory=torch.cuda.is_available(),
                          persistent_workers=(num_workers > 0))

    if shuffle is None:
        if streaming:
            shuffle = True
        else:
            cache_name = str(cache_file).lower()
            shuffle = ('valid' not in cache_name) and ('test' not in cache_name)

    pin = torch.cuda.is_available()
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle,
                      num_workers=num_workers, pin_memory=pin,
                      persistent_workers=(num_workers > 0))

def build_dynamic_sub_batch(batch: dict, n_vis: int, n_req: int, n_depots: int,
                            t_current=None, dynamic_features=False) -> dict:
    """
    Slice a full batch to expose only revealed requests.

    If t_current is provided and 't_arrival' is in batch: n_vis is determined by
    how many requests have t_arrival <= t_current (min across the batch), giving
    training-evaluation alignment with the rolling-horizon dispatcher.
    Otherwise falls back to the supplied n_vis cap (random index-based slicing).

    If dynamic_features=True and t_current is provided: augments node features
    with order_age, delivery_slack_dynamic, t_current fraction, and visibility fraction,
    expanding x from 11D to 15D.  Requires model trained with input_node_dim=15.
    """
    device = batch['x'].device
    B      = batch['x'].shape[0]

    # ── Determine n_vis from t_arrival when t_current is given ──────────────
    if t_current is not None and 't_arrival' in batch:
        t_arr = batch['t_arrival']          # [B, n_req]
        n_vis = int((t_arr <= t_current).sum(dim=1).min().item())
        n_vis = max(1, min(n_vis, n_req))

    n_total = n_depots + n_req * 2
    n_sub   = n_depots + n_vis * 2

    keep = (
        list(range(n_depots)) +
        list(range(n_depots, n_depots + n_vis)) +
        list(range(n_depots + n_req, n_depots + n_req + n_vis))
    )
    keep_t = torch.tensor(keep, dtype=torch.long, device=device)

    def _slice_edge(ea):
        D   = ea.shape[-1]
        mat = ea.view(B, n_total, n_total, D)
        return mat[:, keep_t][:, :, keep_t].reshape(B, n_sub * n_sub, D)

    idx = torch.arange(n_sub, device=device)
    ii, jj = torch.meshgrid(idx, idx, indexing='ij')
    ei = torch.stack([ii.reshape(-1), jj.reshape(-1)], dim=0)

    sub = {
        'x':                  batch['x'][:, keep_t],
        'demand':             batch['demand'][:, keep_t],
        'time_window':        batch['time_window'][:, keep_t],
        'capacity':           batch['capacity'],
        'battery':            batch['battery'],
        'wind':               batch['wind'],
        'n_depots':           n_depots,
        'n_depots_uav':       batch.get('n_depots_uav', n_depots),
        'n_req':              n_vis,
        'edge_attr_uav':      _slice_edge(batch['edge_attr_uav']),
        'edge_attr_adr':      _slice_edge(batch['edge_attr_adr']),
        'edge_attr_d':        _slice_edge(batch['edge_attr_d']),
        'edge_attr_r':        _slice_edge(batch['edge_attr_r']),
        'mask_adjacency_uav': _slice_edge(batch['mask_adjacency_uav']),
        'mask_adjacency_adr': _slice_edge(batch['mask_adjacency_adr']),
        'edge_index':         ei.unsqueeze(0).expand(B, -1, -1).contiguous(),
    }
    if 't_arrival' in batch:
        sub['t_arrival'] = batch['t_arrival'][:, :n_vis]
    if 'coordinates' in batch:
        sub['coordinates'] = batch['coordinates'][:, keep_t]
    for key in ('model_version', 'energy_model_version', 'generator_version',
                'uav_hardware_profile', 'adr_hardware_profile', 'objective_weights',
                'service_policy', 'objective_version', 'route_scope', 'sharing_budget', 'sharing_used',
                'physical_config'):
        if key in batch:
            sub[key] = batch[key]
    for key in ('request_firm_owner', 'request_market_group', 'request_allowed_modes',
                'request_assigned_mode', 'delivery_owner', 'active_request_mask'):
        if key in batch:
            sub[key] = batch[key][:, :n_vis]
    if any(key in batch for key in ('start_node', 'start_capacity', 'start_battery', 'sharing_contract')):
        raise NotImplementedError('Sub-batch slicing of rolling states requires node/carrier remapping; use full masked batches')

    # Dynamic node features: 4 extra dims → 15D (matches build_dynamic_masked_batch).
    if dynamic_features and t_current is not None:
        _TIME_SCALE = 180.0
        _T_MAX      = 120.0
        if 't_arrival' in batch:
            t_arr_vis = batch['t_arrival'][:, :n_vis]
            order_age = ((t_current - t_arr_vis) / _TIME_SCALE).clamp(0.0, 1.0)
        else:
            order_age = sub['x'].new_zeros(B, n_vis)
        t_del = sub['time_window'][:, n_depots + n_vis:n_sub, 0]
        del_slack = ((t_del - t_current) / _TIME_SCALE).clamp(-1.0, 1.0)
        extra = sub['x'].new_zeros(B, n_sub, 4)
        extra[:, n_depots:n_depots + n_vis, 0] = order_age
        extra[:, n_depots + n_vis:n_sub, 1]    = del_slack
        extra[:, :, 2] = t_current / _T_MAX
        extra[:, :, 3] = n_vis / max(n_req, 1)
        sub['x'] = torch.cat([sub['x'], extra], dim=-1)

    return sub


def build_dynamic_masked_batch(batch: dict, n_req: int, n_depots: int,
                               t_current: float, dynamic_features: bool = True) -> tuple:
    """
    Full-graph dynamic masking without slicing.
    Returns (masked_batch, initial_visited, vis_counts).

    Zeroes x / time_window / demand for unrevealed nodes per-instance and
    optionally appends order_age + delivery_slack_dynamic (→ 13-D x).
    initial_visited is [B, n_total] with 1 for every unrevealed node.
    vis_counts is a [B] long tensor with the number of revealed orders per instance.
    """
    device = batch['x'].device
    B      = batch['x'].shape[0]
    n_total = n_depots + n_req * 2

    # Per-instance revealed mask
    if 't_arrival' in batch:
        t_arr    = batch['t_arrival']                               # [B, n_req]
        revealed = (t_arr <= t_current)                             # [B, n_req] bool
        vis_counts = revealed.sum(dim=1).clamp(min=1, max=n_req)    # [B] long
    else:
        revealed   = torch.ones(B, n_req, dtype=torch.bool, device=device)
        vis_counts = torch.full((B,), n_req, dtype=torch.long, device=device)

    unrevealed = ~revealed  # [B, n_req] bool

    # Per-instance initial_visited: [B, n_total]
    initial_visited = batch['x'].new_zeros(B, n_total)
    initial_visited[:, n_depots:n_depots + n_req]           = unrevealed.float()
    initial_visited[:, n_depots + n_req:n_depots + 2*n_req] = unrevealed.float()

    out = dict(batch)

    # Zero unrevealed nodes per-instance
    revealed_f = revealed.float().unsqueeze(-1)  # [B, n_req, 1]

    x = batch['x'].clone()
    x[:, n_depots:n_depots + n_req, :]           *= revealed_f
    x[:, n_depots + n_req:n_depots + 2*n_req, :] *= revealed_f
    out['x'] = x

    if 'time_window' in batch:
        tw = batch['time_window'].clone()
        tw[:, n_depots:n_depots + n_req, :]           *= revealed_f
        tw[:, n_depots + n_req:n_depots + 2*n_req, :] *= revealed_f
        out['time_window'] = tw

    if 'demand' in batch:
        dm = batch['demand'].clone()
        dm[:, n_depots:n_depots + n_req, :]           *= revealed_f
        dm[:, n_depots + n_req:n_depots + 2*n_req, :] *= revealed_f
        out['demand'] = dm

    # Block unrevealed nodes in encoder adjacency masks or softmax attention
    # pools include contributions from zero-value but non-zero-weight neighbours.
    unrevealed_full = torch.zeros(B, n_total, dtype=torch.bool, device=device)
    unrevealed_full[:, n_depots:n_depots + n_req]           = unrevealed
    unrevealed_full[:, n_depots + n_req:n_depots + 2*n_req] = unrevealed
    revealed_node = (~unrevealed_full).float()  # [B, n_total]: 1 = revealed/depot, 0 = hidden
    for _adj_key in ('mask_adjacency_uav', 'mask_adjacency_adr'):
        if _adj_key in batch:
            adj = batch[_adj_key].squeeze(-1).view(B, n_total, n_total).clone()
            adj = adj * revealed_node.unsqueeze(2)
            adj = adj * revealed_node.unsqueeze(1)
            out[_adj_key] = adj.view(B, n_total * n_total, 1)

    # Dynamic node features → 15D total: [order_age, del_slack, t_current, n_vis/n_req].
    if dynamic_features:
        _TIME_SCALE = 180.0
        _T_MAX      = 120.0
        x_base = out['x']
        extra  = x_base.new_zeros(B, n_total, 4)

        # dim 0: order_age at pickup nodes
        if 't_arrival' in batch:
            order_age = ((t_current - batch['t_arrival']) / _TIME_SCALE).clamp(0.0, 1.0)
            order_age = order_age * revealed.float()
            extra[:, n_depots:n_depots + n_req, 0] = order_age

        t_del     = out['time_window'][:, n_depots + n_req:n_depots + 2*n_req, 0]
        del_slack = ((t_del - t_current) / _TIME_SCALE).clamp(-1.0, 1.0)
        del_slack = del_slack * revealed.float()
        extra[:, n_depots + n_req:n_depots + 2*n_req, 1] = del_slack

        extra[:, :, 2] = t_current / _T_MAX
        vis_frac = vis_counts.float() / max(n_req, 1)
        extra[:, :, 3] = vis_frac.unsqueeze(1)

        out['x'] = torch.cat([x_base, extra], dim=-1)

    return out, initial_visited, vis_counts


def build_rolling_residual_batch(
    batch: dict,
    n_req: int,
    n_depots: int,
    n_uav: int,
    n_adr: int,
    t_current: float = None,
    served_frac_range: tuple = (0.0, 0.5),
    committed_frac: float = 0.15,
    dynamic_features: bool = True,
    rng: np.random.Generator = None,
    n_depots_uav: int = None,
) -> tuple:
    """
    Build a mid-horizon residual training state from a full batched instance.

    Simulates a rolling-horizon dispatch epoch:
      - Reveals only orders with t_arrival <= t_current
      - Marks a random fraction of revealed orders as already served
      - Places each vehicle at a served delivery node (60%) or its depot (40%)
      - Sets non-zero start_time and partially-depleted start_battery

    Returns
    -------
    residual_batch : dict  (masked, 15-D dynamic features if dynamic_features=True)
    initial_visited : [B, n_total]  (1 = unrevealed or already served)
    start_node      : [B, n_agents] (current node index per vehicle)
    start_time      : [B, n_agents] (current clock per vehicle, minutes)
    start_battery   : [B, n_agents] (remaining battery — < full)
    start_capacity  : [B, n_agents] (remaining capacity — full; no onboard tracking)
    vis_counts      : [B]           (number of revealed orders per instance)
    """
    device = batch['x'].device
    B = batch['x'].shape[0]
    n_agents = n_uav + n_adr
    n_total = n_depots + n_req * 2
    _n_dep_uav = n_depots_uav if n_depots_uav is not None else n_depots
    _n_dep_adr = n_depots - _n_dep_uav

    if rng is None:
        rng = np.random.default_rng()
    if t_current is None:
        t_current = float(rng.uniform(15.0, 100.0))

    # ── Reveal requests ──────────────────────────────────────────────────────
    if 't_arrival' in batch:
        t_arr = batch['t_arrival']                                  # [B, n_req]
        revealed = t_arr <= t_current                               # [B, n_req] bool
        vis_counts = revealed.sum(dim=1).clamp(min=1, max=n_req)
    else:
        revealed = torch.ones(B, n_req, dtype=torch.bool, device=device)
        vis_counts = torch.full((B,), n_req, dtype=torch.long, device=device)

    # ── Mark served requests (random fraction of revealed) ───────────────────
    served_frac = float(rng.uniform(served_frac_range[0], served_frac_range[1]))
    served = revealed & (torch.rand(B, n_req, device=device) < served_frac)  # [B, n_req]

    unrevealed = ~revealed

    # initial_visited: unrevealed + served (both pickup and delivery nodes)
    initial_visited = batch['x'].new_zeros(B, n_total)
    initial_visited[:, n_depots:n_depots + n_req]           = (unrevealed | served).float()
    initial_visited[:, n_depots + n_req:n_depots + 2*n_req] = (unrevealed | served).float()

    # ── Build masked batch (same pattern as build_dynamic_masked_batch) ──────
    revealed_f = revealed.float().unsqueeze(-1)               # [B, n_req, 1]
    out = dict(batch)

    x = batch['x'].clone()
    x[:, n_depots:n_depots + n_req, :]           *= revealed_f
    x[:, n_depots + n_req:n_depots + 2*n_req, :] *= revealed_f
    out['x'] = x

    if 'time_window' in batch:
        tw = batch['time_window'].clone()
        tw[:, n_depots:n_depots + n_req, :]           *= revealed_f
        tw[:, n_depots + n_req:n_depots + 2*n_req, :] *= revealed_f
        out['time_window'] = tw

    if 'demand' in batch:
        dm = batch['demand'].clone()
        dm[:, n_depots:n_depots + n_req, :]           *= revealed_f
        dm[:, n_depots + n_req:n_depots + 2*n_req, :] *= revealed_f
        out['demand'] = dm

    unrevealed_full = torch.zeros(B, n_total, dtype=torch.bool, device=device)
    unrevealed_full[:, n_depots:n_depots + n_req]           = unrevealed
    unrevealed_full[:, n_depots + n_req:n_depots + 2*n_req] = unrevealed
    revealed_node = (~unrevealed_full).float()
    for _adj_key in ('mask_adjacency_uav', 'mask_adjacency_adr'):
        if _adj_key in batch:
            adj = batch[_adj_key].squeeze(-1).view(B, n_total, n_total).clone()
            adj = adj * revealed_node.unsqueeze(2)
            adj = adj * revealed_node.unsqueeze(1)
            out[_adj_key] = adj.view(B, n_total * n_total, 1)

    # ── Vehicle start positions ───────────────────────────────────────────────
    # Default: each agent at its assigned depot
    start_node_np = np.zeros((B, n_agents), dtype=np.int64)
    for ag in range(n_uav):
        start_node_np[:, ag] = ag % max(_n_dep_uav, 1)
    for ag in range(n_adr):
        start_node_np[:, n_uav + ag] = _n_dep_uav + ag % max(_n_dep_adr, 1)

    # 60% probability: place each vehicle at the delivery node of a served request
    # (vehicle just completed that delivery and is free at that location)
    served_cpu = served.cpu().numpy()                          # [B, n_req]
    for b in range(B):
        served_idxs = np.where(served_cpu[b])[0]
        if len(served_idxs) == 0:
            continue
        for ag in range(n_agents):
            if rng.random() < 0.6:
                req_i = int(rng.choice(served_idxs))
                start_node_np[b, ag] = n_depots + n_req + req_i  # delivery node
    start_node = torch.tensor(start_node_np, dtype=torch.long, device=device)  # [B, n_agents]

    available_mask = torch.ones(B, n_agents, device=device)
    if committed_frac > 0.0 and n_agents > 1:
        committed = torch.rand(B, n_agents, device=device) < committed_frac
        all_committed = committed.all(dim=1)
        if all_committed.any():
            committed[all_committed, 0] = False
        available_mask = (~committed).float()
        start_node = torch.where(available_mask.bool(), start_node, torch.zeros_like(start_node))

    # ── Vehicle start times (≈ t_current with ±3 min per-vehicle jitter) ─────
    spread = torch.randn(B, n_agents, device=device) * 3.0
    start_time = (t_current + spread).clamp(0.0, 115.0)       # [B, n_agents]

    # ── Vehicle start battery (partially depleted) ────────────────────────────
    # Depletion is roughly proportional to elapsed fraction of 120-min shift,
    # plus per-vehicle noise so the model sees a distribution of battery states.
    full_bat = batch['battery'].view(B, n_agents)              # [B, n_agents]
    decay_frac = t_current / 120.0
    noise = torch.rand(B, n_agents, device=device) * 0.25
    bat_frac = (1.0 - decay_frac * 0.65 - noise * 0.35).clamp(0.15, 1.0)
    start_battery = full_bat * bat_frac                        # [B, n_agents]

    # ── Vehicle start capacity (full; no onboard tracking in this pass) ───────
    start_capacity = batch['capacity'].view(B, n_agents)       # [B, n_agents]

    # ── Dynamic node features (same 4 extra dims as build_dynamic_masked_batch) ──
    if dynamic_features:
        _TIME_SCALE = 180.0
        _T_MAX = 120.0
        x_base = out['x']
        extra = x_base.new_zeros(B, n_total, 4)

        if 't_arrival' in batch:
            order_age = ((t_current - batch['t_arrival']) / _TIME_SCALE).clamp(0.0, 1.0)
            order_age = order_age * revealed.float()
            extra[:, n_depots:n_depots + n_req, 0] = order_age

        t_del = out['time_window'][:, n_depots + n_req:n_depots + 2*n_req, 0]
        del_slack = ((t_del - t_current) / _TIME_SCALE).clamp(-1.0, 1.0)
        del_slack = del_slack * revealed.float()
        extra[:, n_depots + n_req:n_depots + 2*n_req, 1] = del_slack
        extra[:, :, 2] = t_current / _T_MAX
        extra[:, :, 3] = vis_counts.float().unsqueeze(1) / max(n_req, 1)

        out['x'] = torch.cat([x_base, extra], dim=-1)         # [B, n_total, 15]

    out['start_node'] = start_node
    out['start_time'] = start_time
    out['start_battery'] = start_battery
    out['start_capacity'] = start_capacity
    out['available_mask'] = available_mask

    return out, initial_visited, start_node, start_time, start_battery, start_capacity, vis_counts


def _pickup_speed(distance, t_now, t_target, v_min, v_max):
    slack = torch.clamp(t_target - t_now, min=0.0)
    exact_speed = distance / torch.clamp(slack, min=1e-8)
    speed = torch.clamp(exact_speed, min=v_min, max=v_max)
    must_rush = (slack <= 0.0) | (distance / v_max >= slack)
    speed = torch.where(must_rush, torch.full_like(speed, v_max), speed)
    speed = torch.where(distance <= 1e-8, torch.full_like(speed, v_min), speed)
    return speed

def reward1(time_window, tour_indices, edge_attr_d, edge_attr_r, time, num_drones,
            num_depots=None, return_breakdown=False, start_time=None,
            active_request_mask=None, initial_visited=None, available_mask=None,
            cost_weights=None, physics_batch=None, start_battery=None, start_capacity=None,
            service_policy=None, allow_partial=False):
    """Common route cost; full battery accounting requires physics_batch.

    The no-context signature is retained for legacy diagnostics only, not
    corrected experimental evidence.
    """
    from reward import objective_weights, trajectory_battery_cost, replay_mandatory_trajectory
    import warnings
    policy = service_policy or (physics_batch or {}).get('service_policy')
    if policy is None:
        warnings.warn('Unlabelled reward1 calls are legacy_optional diagnostics only; '
                      'set service_policy explicitly.', RuntimeWarning, stacklevel=2)
        policy = 'legacy_optional'
    if policy not in ('mandatory', 'legacy_optional'):
        raise ValueError('Unknown tensor service policy')
    if policy != 'mandatory' and (physics_batch or {}).get('physical_config') is not None:
        raise NotImplementedError('Configured physical modes require mandatory replay, not legacy reward reconstruction')
    if policy == 'mandatory':
        if physics_batch is None or num_depots is None:
            raise ValueError('Mandatory tensor objective requires physical context and explicit depot count')
        context = dict(physics_batch, time_window=time_window,
                       edge_attr_d=edge_attr_d, edge_attr_r=edge_attr_r)
        if available_mask is not None:
            context['available_mask'] = available_mask
        replay = replay_mandatory_trajectory(
            context, tour_indices, time, num_drones, num_depots,
            weights=cost_weights, initial_visited=initial_visited,
            active_request_mask=active_request_mask, start_time=start_time,
            start_battery=start_battery, start_capacity=start_capacity,
            allow_partial=allow_partial,
        )
        if return_breakdown:
            return replay['prefix_cost'], dict(replay['components'], complete=replay['complete'],
                                             unassigned_count=replay['unassigned_count'],
                                             valuation_scope=replay['valuation_scope'])
        if allow_partial:
            raise ValueError('Residual prefix scoring requires return_breakdown=True and completion metadata')
        return replay['prefix_cost']
    weights = objective_weights(cost_weights if cost_weights is not None else (physics_batch or {}).get('objective_weights'))
    if start_time is None and physics_batch is not None:
        start_time = physics_batch.get('start_time')
    if available_mask is None and physics_batch is not None:
        available_mask = physics_batch.get('available_mask')
    dev = tour_indices.device
    batch_size, n_agent, steps = tour_indices.size()
    if num_depots is None:
        num_depots = int((time_window == 0).sum().item() // batch_size)
    num_nodes = time_window.size(1) - num_depots
    num_pickup = num_nodes // 2

    tw = time_window.view(batch_size, num_depots + num_nodes)
    edge_attr_d = edge_attr_d.view(batch_size, num_depots + num_nodes, num_depots + num_nodes)
    edge_attr_r = edge_attr_r.view(batch_size, num_depots + num_nodes, num_depots + num_nodes)

    prev_idx = tour_indices[:, :, :-1]
    cur_idx = tour_indices[:, :, 1:]

    moved = cur_idx != prev_idx
    is_depot = (cur_idx < num_depots) & moved
    is_pickup = (cur_idx >= num_depots) & (cur_idx < num_depots + num_pickup) & moved
    is_delivery = (cur_idx >= num_depots + num_pickup) & moved
    is_customer = is_pickup | is_delivery
    is_consec_depot = is_depot & (prev_idx < num_depots)

    ALPHA_1, ALPHA_2 = weights.operating_uav, weights.operating_adr
    ALPHA_E, ALPHA_P, ALPHA_D = weights.pickup_early, weights.pickup_late, weights.delivery_late
    ALPHA_U = weights.unserved

    n_uav = num_drones
    if available_mask is not None:
        available_mask = available_mask.to(dev).float().view(batch_size, n_agent)
    batch_d = torch.arange(batch_size, device=dev).unsqueeze(1).expand(-1, n_uav).unsqueeze(2)
    batch_r = torch.arange(batch_size, device=dev).unsqueeze(1).expand(-1, n_agent - n_uav).unsqueeze(2)

    dis_d = edge_attr_d[batch_d, prev_idx[:, :n_uav], cur_idx[:, :n_uav]]
    dis_r = edge_attr_r[batch_r, prev_idx[:, n_uav:], cur_idx[:, n_uav:]]

    _t0 = (start_time.view(batch_size, n_agent, 1).to(dev).to(time.dtype)
           if start_time is not None
           else torch.zeros(batch_size, n_agent, 1, device=dev, dtype=time.dtype))
    prev_time = torch.cat([_t0, time[:, :, :-1]], dim=2)
    tw_cur = torch.gather(tw.unsqueeze(1).expand(-1, n_agent, -1), 2, cur_idx)

    travel_time_uav = dis_d / V_UAV_COORD_PER_MIN
    travel_time_adr = dis_r / V_ADR_COORD_PER_MIN

    if n_agent - n_uav > 0:
        pickup_adr = is_pickup[:, n_uav:]
        if pickup_adr.any():
            travel_time_adr = travel_time_adr.clone()
            speed_a = _pickup_speed(
                dis_r,
                prev_time[:, n_uav:, :],
                tw_cur[:, n_uav:, :],
                v_min=V_ADR_MIN_PICKUP,
                v_max=V_ADR_COORD_PER_MIN,
            )
            travel_time_adr[pickup_adr] = dis_r[pickup_adr] / speed_a[pickup_adr]

    travel_time = torch.cat([travel_time_uav, travel_time_adr], dim=1)
    adr_agents = torch.arange(n_agent, device=dev).view(1, n_agent, 1) >= n_uav
    travel_time = torch.where(is_depot & adr_agents, travel_time * 2.0, travel_time)
    travel_time = torch.where(moved, travel_time, torch.zeros_like(travel_time))

    alpha_mode = torch.cat([
        torch.full((batch_size, n_uav, steps - 1), ALPHA_1, device=dev),
        torch.full((batch_size, n_agent - n_uav, steps - 1), ALPHA_2, device=dev),
    ], dim=1)
    operating_cost = (alpha_mode * travel_time).sum(dim=2)
    if available_mask is not None:
        operating_cost = operating_cost * available_mask

    arrival_pre_service = prev_time + travel_time

    completion_time = arrival_pre_service.clone()
    if n_uav > 0:
        uav_customer = is_customer[:, :n_uav].float()
        completion_time[:, :n_uav, :] = completion_time[:, :n_uav, :] + UAV_LAND_TAKEOFF_MIN * uav_customer

    early_pickup = F.relu(tw_cur - arrival_pre_service) * is_pickup.float()
    late_pickup = F.relu(arrival_pre_service - tw_cur) * is_pickup.float()
    late_delivery = F.relu(completion_time - tw_cur) * is_delivery.float()

    early_pickup = torch.where(is_consec_depot, torch.zeros_like(early_pickup), early_pickup)
    late_pickup = torch.where(is_consec_depot, torch.zeros_like(late_pickup), late_pickup)
    late_delivery = torch.where(is_consec_depot, torch.zeros_like(late_delivery), late_delivery)

    delivery_start = num_depots + num_pickup
    if active_request_mask is None and initial_visited is not None:
        iv = initial_visited.view(batch_size, num_depots + num_nodes)
        active_request_mask = iv[:, delivery_start:delivery_start + num_pickup] <= 0.5
    if active_request_mask is None:
        active_request_mask = torch.ones(batch_size, num_pickup, dtype=torch.bool, device=dev)
    else:
        active_request_mask = active_request_mask.to(dev).bool().view(batch_size, num_pickup)

    pickup_req_id = cur_idx - num_depots
    delivery_req_id = cur_idx - delivery_start
    req_id = torch.where(is_delivery, delivery_req_id, pickup_req_id)
    req_id = req_id.clamp(min=0, max=max(num_pickup - 1, 0))
    active_cur = torch.gather(
        active_request_mask.unsqueeze(1).expand(-1, n_agent, -1),
        2,
        req_id,
    )
    active_customer = torch.where(is_customer, active_cur, torch.ones_like(active_cur, dtype=torch.bool))

    early_pickup = early_pickup * active_customer.float()
    late_pickup = late_pickup * active_customer.float()
    late_delivery = late_delivery * active_customer.float()

    delivery_nodes = torch.arange(delivery_start, delivery_start + num_pickup, device=dev)
    delivery_events = is_delivery
    if available_mask is not None:
        delivery_events = delivery_events & available_mask.unsqueeze(-1).bool()
    visited_any = ((cur_idx.unsqueeze(-1) == delivery_nodes) & delivery_events.unsqueeze(-1)).any(dim=1).any(dim=1)
    undelivered  = (~visited_any) & active_request_mask                       # [B, num_pickup]

    # Attribution changes per-agent credit, never the collective objective.
    # Unreachable orders and an entirely locked fleet still owe missed-order cost.
    allocation = (available_mask if available_mask is not None else time.new_ones(batch_size, n_agent))
    allocation = torch.where(allocation.sum(dim=1, keepdim=True) > 0, allocation, torch.ones_like(allocation))
    allocation = allocation / allocation.sum(dim=1, keepdim=True)
    undeliv_cost = ALPHA_U * undelivered.sum(dim=1, keepdim=True) * allocation

    tw_pen = (
        ALPHA_P * late_pickup.sum(dim=2)
        + ALPHA_D * late_delivery.sum(dim=2)
        + ALPHA_E * early_pickup.sum(dim=2)
    )
    if available_mask is not None:
        tw_pen = tw_pen * available_mask
    delivery_credit = _ALPHA_DELIV_BONUS * (is_delivery & active_customer).float().sum(dim=2)  # [B, n_agents]
    if available_mask is not None:
        delivery_credit = delivery_credit * available_mask
    battery_cost = torch.zeros_like(operating_cost)
    if physics_batch is not None:
        battery_cost = trajectory_battery_cost(
            physics_batch, tour_indices, time, num_drones, num_depots, weights,
            start_time=start_time, start_battery=start_battery, start_capacity=start_capacity,
        )['cost']
    total_cost = operating_cost + tw_pen + battery_cost + undeliv_cost - delivery_credit
    if return_breakdown:
        return total_cost, {
            'travel':          operating_cost,
            'tw_penalty':      tw_pen,
            'undeliv':         undeliv_cost,
            'delivery_credit': delivery_credit,
            'pickup_early_cost': ALPHA_E * early_pickup.sum(dim=2) * (available_mask if available_mask is not None else 1.),
            'pickup_late_cost': ALPHA_P * late_pickup.sum(dim=2) * (available_mask if available_mask is not None else 1.),
            'delivery_late_cost': ALPHA_D * late_delivery.sum(dim=2) * (available_mask if available_mask is not None else 1.),
            'battery': battery_cost,
        }
    return total_cost

