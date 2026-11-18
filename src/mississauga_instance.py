from __future__ import annotations

import math
import os
import hashlib
import json
from dataclasses import asdict, dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
from model_config import UAV_CAPACITY_KG, ADR_CAPACITY_KG

from creat_vrp import (
    SCALE_M_PER_COORD,
    UAV_CRUISE_MPS,
    ADR_CRUISE_MPS,
    V_UAV_COORD_PER_MIN,
    V_ADR_COORD_PER_MIN,
    static_signed_margin,
)

MISSISSAUGA_POLYGON_LONLAT = [
    (-79.619, 43.476),
    (-79.534, 43.576),
    (-79.620, 43.639),
    (-79.720, 43.548),
]

UAV_NETWORK_FILTER = '["highway"~"primary|secondary"]'
ADR_NETWORK_FILTER = ('["highway"~"^(residential|living_street|footway|pedestrian|path)$"]'
                      '["access"!~"^(private|no)$"]["foot"!~"^(private|no)$"]')
GEOGRAPHY_VERSION = "residential-proxy-synthetic-air-exclusions-1"

PEARSON_LONLAT = (-79.6306, 43.6777)
PEARSON_NO_FLY_RADIUS_KM = 5.6


@dataclass(frozen=True)
class FixedHeightObstacle:
    center_xy_km: Tuple[float, float]
    radius_km: float
    height_m: float


@dataclass(frozen=True)
class GeographyPolicy:
    # These are declared simulation exclusions, not aviation authorization.
    airport_radius_km: float = PEARSON_NO_FLY_RADIUS_KM
    cruise_height_m: float = 100.0
    obstacles: Tuple[FixedHeightObstacle, ...] = ()

    def __post_init__(self):
        if not math.isfinite(self.airport_radius_km) or self.airport_radius_km < 0:
            raise ValueError("Airport radius must be finite and nonnegative")
        if not math.isfinite(self.cruise_height_m) or self.cruise_height_m <= 0:
            raise ValueError("Cruise height must be finite and positive")
        for obstacle in self.obstacles:
            if (len(obstacle.center_xy_km) != 2
                    or not np.isfinite(obstacle.center_xy_km).all()
                    or not np.isfinite([obstacle.radius_km, obstacle.height_m]).all()
                    or obstacle.radius_km <= 0 or obstacle.height_m < 0):
                raise ValueError("Invalid declared fixed-height obstacle")


def geography_contract(policy: GeographyPolicy) -> dict:
    return dict(version=GEOGRAPHY_VERSION, policy=asdict(policy),
                adr_filter=ADR_NETWORK_FILTER, candidate_filter=UAV_NETWORK_FILTER,
                polygon=MISSISSAUGA_POLYGON_LONLAT, airport_lonlat=PEARSON_LONLAT,
                adr_interpretation="undirected residential/pedestrian road proxy; only mapped crossings; no invented connectors",
                uav_interpretation="straight segments with synthetic exclusions; no authorization or landing-feasibility claim")


def _cache_key(policy: GeographyPolicy) -> str:
    return hashlib.sha256(json.dumps(geography_contract(policy), sort_keys=True).encode()).hexdigest()

@dataclass
class MississaugaNetworks:
    coords_km: np.ndarray
    is_restaurant_candidate: np.ndarray
    acc_uav_node: np.ndarray
    acc_adr_node: np.ndarray
    d_uav_km: np.ndarray
    d_adr_km: np.ndarray
    bbox_xy_km: Tuple[float, float, float, float]
    bbox_lonlat: Tuple[float, float, float, float]
    pearson_xy_km: np.ndarray
    cache_key: str
    geography_metadata: dict = field(default_factory=dict)

_NETWORKS_CACHE: Optional[MississaugaNetworks] = None
_CACHE_PATH_DEFAULT = "cache/mississauga_networks.npz"

def _project_lonlat_to_local_km(lon: np.ndarray, lat: np.ndarray,
                                 lon0: float, lat0: float) -> np.ndarray:
    R = 6371.0088
    x = np.radians(lon - lon0) * np.cos(np.radians(lat0)) * R
    y = np.radians(lat - lat0) * R
    return np.stack([x, y], axis=-1).astype(np.float32)

def _build_networks_from_osm(cache_path: str,
                             policy: GeographyPolicy = GeographyPolicy()) -> MississaugaNetworks:
    import networkx as nx
    import osmnx as ox
    from shapely.geometry import Polygon

    poly = Polygon(MISSISSAUGA_POLYGON_LONLAT)

    G_uav = ox.graph_from_polygon(poly, simplify=True, retain_all=False,
                                   network_type="all",
                                   custom_filter=UAV_NETWORK_FILTER)
    G_adr = ox.graph_from_polygon(poly, simplify=True, retain_all=True,
                                   network_type="all",
                                   custom_filter=ADR_NETWORK_FILTER)

    candidate_lonlat: Dict[int, Tuple[float, float]] = {}
    for nid, data in G_uav.nodes(data=True):
        candidate_lonlat[nid] = (data['x'], data['y'])
    for nid, data in G_adr.nodes(data=True):
        candidate_lonlat.setdefault(nid, (data['x'], data['y']))

    node_ids = list(candidate_lonlat.keys())
    lon = np.array([candidate_lonlat[n][0] for n in node_ids], dtype=np.float32)
    lat = np.array([candidate_lonlat[n][1] for n in node_ids], dtype=np.float32)
    lon0 = lon.mean()
    lat0 = lat.mean()
    coords_km = _project_lonlat_to_local_km(lon, lat, lon0, lat0)

    coords_km -= coords_km.min(axis=0)
    bbox_xy_km = (0.0, 0.0,
                  float(coords_km[:, 0].max()),
                  float(coords_km[:, 1].max()))
    bbox_lonlat = (float(lon.min()), float(lat.min()),
                   float(lon.max()), float(lat.max()))

    pearson_xy = _project_lonlat_to_local_km(
        np.array([PEARSON_LONLAT[0]]), np.array([PEARSON_LONLAT[1]]),
        lon0, lat0,
    )[0]
    pearson_xy = pearson_xy - np.array(
        _project_lonlat_to_local_km(lon, lat, lon0, lat0).min(axis=0)
    )

    adr_node_set = set(G_adr.nodes())
    acc_adr_node = np.array([nid in adr_node_set for nid in node_ids], dtype=bool)

    uav_node_set = set(G_uav.nodes())
    is_restaurant = np.array([nid in uav_node_set for nid in node_ids], dtype=bool)

    print(f"[mississauga] Computing ADR shortest paths on {len(node_ids)} nodes...")
    d_adr_km = _all_pairs_shortest_path_km(G_adr, node_ids, default_blocked=True)

    diff = coords_km[:, None, :] - coords_km[None, :, :]
    d_uav_km = np.sqrt((diff ** 2).sum(-1)).astype(np.float32)
    acc_uav_node = apply_uav_exclusions(d_uav_km, coords_km, pearson_xy, policy)

    nets = MississaugaNetworks(
        coords_km=coords_km,
        is_restaurant_candidate=is_restaurant,
        acc_uav_node=acc_uav_node,
        acc_adr_node=acc_adr_node,
        d_uav_km=d_uav_km,
        d_adr_km=d_adr_km,
        bbox_xy_km=bbox_xy_km,
        bbox_lonlat=bbox_lonlat,
        pearson_xy_km=pearson_xy.astype(np.float32),
        cache_key=_cache_key(policy),
        geography_metadata=geography_contract(policy),
    )

    os.makedirs(os.path.dirname(os.path.abspath(cache_path)), exist_ok=True)
    from experiment_manifest import atomic_output
    with atomic_output(cache_path, "wb") as f:
        payload = vars(nets).copy()
        payload["geography_metadata"] = json.dumps(nets.geography_metadata)
        np.savez_compressed(f, **payload)
    print(f"[mississauga] Saved network cache to {cache_path}")
    return nets

def _all_pairs_shortest_path_km(G, node_ids: List[int],
                                  default_blocked: bool = True) -> np.ndarray:
    import networkx as nx

    n = len(node_ids)
    INF = float("inf")
    out = np.full((n, n), INF if default_blocked else 0.0, dtype=np.float32)
    idx_of = {nid: i for i, nid in enumerate(node_ids)}

    H = G.to_undirected()
    for _, _, data in H.edges(data=True):
        if "length" not in data or not math.isfinite(data["length"]) or data["length"] < 0:
            raise ValueError("Mapped ADR edges require measured finite nonnegative lengths")

    for src in node_ids:
        if src not in G:
            continue
        lengths = nx.single_source_dijkstra_path_length(
            H, src, weight="length"
        )
        i = idx_of[src]
        for tgt, dist_m in lengths.items():
            if tgt in idx_of:
                out[i, idx_of[tgt]] = dist_m / 1000.0

    np.fill_diagonal(out, 0.0)
    return out

def _block_segments_through_circle(d: np.ndarray, coords: np.ndarray,
                                    center: np.ndarray, radius_km: float) -> None:
    if not math.isfinite(radius_km) or radius_km < 0:
        raise ValueError("Exclusion radius must be finite and nonnegative")
    if radius_km == 0:
        return
    coords = np.asarray(coords, dtype=np.float64)
    p1 = coords[:, None, :] - np.asarray(center, dtype=np.float64)
    dvec = coords[None, :, :] - coords[:, None, :]
    norm2 = (dvec * dvec).sum(-1)
    along = np.divide(-(p1 * dvec).sum(-1), norm2,
                      out=np.zeros_like(norm2), where=norm2 > 0)
    closest = p1 + np.clip(along, 0, 1)[..., None] * dvec
    # Also blocks contained and zero-length arcs, and tangent contact.
    d[(closest * closest).sum(-1) <= radius_km ** 2] = float("inf")


def apply_uav_exclusions(d: np.ndarray, coords: np.ndarray, airport_xy: np.ndarray,
                        policy: GeographyPolicy) -> np.ndarray:
    circles = [(airport_xy, policy.airport_radius_km)]
    circles.extend((obstacle.center_xy_km, obstacle.radius_km)
                   for obstacle in policy.obstacles if obstacle.height_m >= policy.cruise_height_m)
    accessible = np.ones(len(coords), dtype=bool)
    for center, radius in circles:
        _block_segments_through_circle(d, coords, np.asarray(center), radius)
        if radius > 0:
            accessible &= np.linalg.norm(coords - np.asarray(center), axis=1) > radius
    return accessible

def load_mississauga_networks(cache_path: str = _CACHE_PATH_DEFAULT,
                              rebuild: bool = False,
                              policy: GeographyPolicy = GeographyPolicy()) -> MississaugaNetworks:
    global _NETWORKS_CACHE
    if (_NETWORKS_CACHE is not None and not rebuild
            and _NETWORKS_CACHE.cache_key == _cache_key(policy)
            and getattr(_NETWORKS_CACHE, '_cache_path', None) == os.path.abspath(cache_path)):
        return _NETWORKS_CACHE

    if os.path.exists(cache_path) and not rebuild:
        try:
            with np.load(cache_path, allow_pickle=False) as values:
                data = {name: values[name] for name in MississaugaNetworks.__dataclass_fields__}
                data["geography_metadata"] = json.loads(data["geography_metadata"].item())
                data["cache_key"] = data["cache_key"].item()
                for name in ("bbox_xy_km", "bbox_lonlat"):
                    data[name] = tuple(data[name].tolist())
                nets = MississaugaNetworks(**data)
            if nets.cache_key != _cache_key(policy):
                raise ValueError("stale network policy/cache version")
            nets._cache_path = os.path.abspath(cache_path)
            _NETWORKS_CACHE = nets
            return nets
        except Exception as exc:
            print(f"[mississauga] Cache invalid ({exc}), rebuilding from OSM...")

    nets = _build_networks_from_osm(cache_path, policy)
    nets._cache_path = os.path.abspath(cache_path)
    _NETWORKS_CACHE = nets
    return nets

def create_mississauga_instance(
    n_req: int,
    n_uav: int,
    n_adr: int,
    n_depots_uav: int,
    n_depots_adr: int,
    rng: np.random.Generator,
    networks: Optional[MississaugaNetworks] = None,
    cache_path: str = _CACHE_PATH_DEFAULT,
    wind_override=None,
    policy: GeographyPolicy = GeographyPolicy(),
) -> Dict[str, torch.Tensor]:
    if n_req <= 0 or min(n_uav, n_adr, n_depots_uav, n_depots_adr) < 0 or n_uav + n_adr == 0:
        raise ValueError("Positive workload and a nonempty nonnegative fleet required")
    if (n_uav and not n_depots_uav) or (n_adr and not n_depots_adr):
        raise ValueError("Each active mode needs a depot")
    if networks is None:
        networks = load_mississauga_networks(cache_path, policy=policy)

    nets = networks
    if nets.cache_key != _cache_key(policy):
        raise ValueError('Supplied network does not match the declared geography policy/version')
    n_depots = n_depots_uav + n_depots_adr
    n_total = n_depots + 2 * n_req

    coords_all = nets.coords_km
    rest_pool = np.where(nets.is_restaurant_candidate
                          & (nets.acc_uav_node | nets.acc_adr_node))[0]
    cust_pool = np.where(nets.acc_uav_node | nets.acc_adr_node)[0]
    uav_depot_pool = np.where(nets.acc_uav_node)[0]
    adr_depot_pool = np.where(nets.acc_adr_node)[0]

    if len(rest_pool) < n_req:
        raise RuntimeError(
            f"Mississauga restaurant pool has only {len(rest_pool)} candidates "
            f"but {n_req} pickups requested."
        )
    if len(cust_pool) < n_req:
        raise RuntimeError("Mississauga customer pool too small.")

    if len(uav_depot_pool) < n_depots_uav or len(adr_depot_pool) < n_depots_adr:
        raise RuntimeError("Not enough physically accessible depot candidates")
    uav_depot_global = rng.choice(uav_depot_pool, size=n_depots_uav, replace=False)
    adr_depot_global = rng.choice(adr_depot_pool, size=n_depots_adr, replace=False)
    pickup_global, delivery_global = [], []
    remaining_customers = set(map(int, cust_pool))
    # Geometry screening is not a battery/quota/full-routing feasibility certificate.
    for pickup in rng.permutation(rest_pool):
        viable = []
        for delivery in sorted(remaining_customers):
            reachable = False
            for active, depots, access, distances in (
                (n_uav, uav_depot_global, nets.acc_uav_node, nets.d_uav_km),
                (n_adr, adr_depot_global, nets.acc_adr_node, nets.d_adr_km),
            ):
                if active and access[pickup] and access[delivery]:
                    reachable |= (np.isfinite(distances[depots, pickup]).any()
                                  and np.isfinite(distances[pickup, delivery])
                                  and np.isfinite(distances[delivery, depots]).any())
            if reachable:
                viable.append(delivery)
        if not viable:
            continue
        delivery = int(rng.choice(viable))
        pickup_global.append(int(pickup))
        delivery_global.append(delivery)
        remaining_customers.remove(delivery)
        if len(pickup_global) == n_req:
            break
    if len(pickup_global) != n_req:
        raise RuntimeError("Cannot sample geometry-reachable pairs under the declared access policy; do not invent access")
    pickup_global = np.asarray(pickup_global, dtype=int)
    delivery_global = np.asarray(delivery_global, dtype=int)

    selected_global = np.concatenate([uav_depot_global, adr_depot_global,
                                       pickup_global, delivery_global])

    coords = coords_all[selected_global].astype(np.float32)
    d_uav_km = nets.d_uav_km[np.ix_(selected_global, selected_global)].astype(np.float32)
    d_adr_km = nets.d_adr_km[np.ix_(selected_global, selected_global)].astype(np.float32)

    KM_PER_COORD = SCALE_M_PER_COORD / 1000.0
    d_uav = d_uav_km / KM_PER_COORD
    d_adr = d_adr_km / KM_PER_COORD
    coords_coord = coords / KM_PER_COORD

    acc_uav = nets.acc_uav_node[selected_global].copy()
    acc_adr = nets.acc_adr_node[selected_global].copy()

    demand_vals = rng.uniform(1.0, 6.0, size=n_req).astype(np.float32)
    demand = np.zeros(n_total, dtype=np.float32)
    demand[n_depots:n_depots + n_req] = demand_vals
    demand[n_depots + n_req:] = -demand_vals

    shift_minutes = 120.0
    # Conditional Poisson arrivals given n_req, without clipping an artificial end burst.
    t_arrival = np.sort(rng.uniform(0, shift_minutes, size=n_req)).astype(np.float32)
    prep_time = rng.uniform(5, 20, size=n_req).astype(np.float32)
    t_pickup = t_arrival + prep_time
    delivery_slack = rng.normal(loc=30.0, scale=5.0, size=n_req).astype(np.float32)
    delivery_slack = np.clip(delivery_slack, 15.0, 60.0)
    t_delivery = (t_pickup + delivery_slack).astype(np.float32)

    Q_uav = UAV_CAPACITY_KG

    wind_dir = float(rng.uniform(0, 2 * math.pi))
    wind_mag = float(rng.uniform(0, 12.0))
    if wind_override is not None:
        wind_mag, wind_dir = map(float, wind_override)
        if not np.isfinite([wind_mag, wind_dir]).all() or wind_mag < 0:
            raise ValueError('Invalid prescribed wind')
    wind_vec = np.array([wind_mag * math.cos(wind_dir),
                         wind_mag * math.sin(wind_dir)], dtype=np.float32)

    from energy import uav_energy_matrix, adr_energy_matrix
    e_uav = uav_energy_matrix(
        d_uav * SCALE_M_PER_COORD, demand_vals, demand, n_depots, n_req,
        wind_vec, UAV_CRUISE_MPS, coordinates=coords_coord,
    )
    e_adr = adr_energy_matrix(
        d_adr * SCALE_M_PER_COORD, demand_vals, demand, n_depots, n_req,
        ADR_CRUISE_MPS,
    )

    tp_full = np.zeros(n_total, dtype=np.float32)
    tp_full[n_depots:n_depots + n_req] = t_pickup
    margin_uav = static_signed_margin(tp_full, d_uav, V_UAV_COORD_PER_MIN)
    margin_adr = static_signed_margin(tp_full, d_adr, V_ADR_COORD_PER_MIN)

    high = max(coords_coord.max(), 5.0)
    DIST_SCALE = high * math.sqrt(2)
    ENERGY_SCALE = 200_000.0
    DEMAND_MAX = 6.0
    COORD_SCALE = high
    TIME_SCALE = shift_minutes + 60.0

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

    delivery_slack_node = np.zeros(n_total, dtype=np.float32)
    delivery_slack_node[n_depots:n_depots + n_req] = delivery_slack / shift_minutes
    delivery_slack_node[n_depots + n_req:]         = delivery_slack / shift_minutes

    node_feat = np.concatenate([
        coords_coord / COORD_SCALE,
        demand[:, None] / DEMAND_MAX,
        t_pickup_feat[:, None] / TIME_SCALE,
        t_delivery_feat[:, None] / TIME_SCALE,
        node_type,
        acc_uav[:, None].astype(np.float32),
        acc_adr[:, None].astype(np.float32),
        delivery_slack_node[:, None],
    ], axis=-1).astype(np.float32)

    edge_attr_uav = np.nan_to_num(edge_attr_uav, nan=0.0, posinf=2.0, neginf=-2.0)
    edge_attr_adr = np.nan_to_num(edge_attr_adr, nan=0.0, posinf=2.0, neginf=-2.0)
    mask_adj_uav = (np.isfinite(d_uav) & acc_uav[:, None] & acc_uav[None, :]).astype(np.float32)
    mask_adj_adr = (np.isfinite(d_adr) & acc_adr[:, None] & acc_adr[None, :]).astype(np.float32)
    edges_index = np.stack(np.indices((n_total, n_total)).reshape(2, -1), axis=0)

    Q_adr = ADR_CAPACITY_KG
    capacity = np.hstack([np.ones(n_uav, dtype=np.float32) * Q_uav,
                          np.ones(n_adr, dtype=np.float32) * Q_adr])
    from model_config import (UAV_BATTERY_KJ, ADR_BATTERY_KJ, ENERGY_MODEL_VERSION, GENERATOR_VERSION,
                             MODEL_VERSION, UAV_PROFILE, ADR_PROFILE)
    BATTERY_UAV = UAV_BATTERY_KJ
    BATTERY_ADR = ADR_BATTERY_KJ
    battery = np.hstack([np.ones(n_uav, dtype=np.float32) * BATTERY_UAV,
                         np.ones(n_adr, dtype=np.float32) * BATTERY_ADR])

    inst = {
        'x': torch.tensor(node_feat, dtype=torch.float32),
        'coordinates': torch.tensor(coords_coord, dtype=torch.float32),
        'energy_model_version': ENERGY_MODEL_VERSION,
        'generator_version': GENERATOR_VERSION,
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
            np.concatenate([np.zeros(n_depots, dtype=np.float32),
                            t_pickup, t_delivery]),
            dtype=torch.float32,
        ),
        't_arrival': torch.tensor(t_arrival, dtype=torch.float32),
        'coords_km': torch.tensor(coords, dtype=torch.float32),
        'osm_node_indices': torch.tensor(selected_global, dtype=torch.long),
        'n_depots': n_depots,
        'n_depots_uav': n_depots_uav,
        'n_depots_adr': n_depots_adr,
        'n_req': n_req,
        'is_real_world': False,
        'geography_version': GEOGRAPHY_VERSION,
        'geography_metadata': nets.geography_metadata,
        'geography_cache_key': nets.cache_key,
        'geometry_screen_only': True,
        'scenario_interpretation': 'synthetic orders on an OSM-derived proxy; not empirical or authorized operations',
    }
    return inst

if __name__ == "__main__":
    nets = load_mississauga_networks()
    print(f"Loaded {len(nets.coords_km)} candidate nodes.")
    print(f"Restaurant candidates: {nets.is_restaurant_candidate.sum()}")
    print(f"UAV-accessible: {nets.acc_uav_node.sum()}")
    print(f"ADR-accessible: {nets.acc_adr_node.sum()}")
    inst = create_mississauga_instance(
        n_req=10, n_uav=2, n_adr=2, n_depots_uav=1, n_depots_adr=1,
        rng=np.random.default_rng(0), networks=nets,
    )
    print(f"Instance x shape: {inst['x'].shape}")
    print(f"edge_attr_uav shape: {inst['edge_attr_uav'].shape}")
    print(f"Map extent (km): {nets.bbox_xy_km}")
