"""Travel propulsion and customer-visit energy in SI units.

Scalar helpers take metres, kg and m/s and return joules. The coordinate
adapter returns kJ. Static graph features describe an empty vehicle, not the
payload of the destination request. Runtime callers must supply pre-arc load.

The UAV law solves the cited steady-level-flight momentum equation with the
selected aircraft geometry. Drag, efficiency and coaxial interaction remain
uncalibrated. Customer service uses a hover-equivalent powered-phase proxy;
landed waiting has zero draw. This is not a calibrated FlyCart replica and
must not support final physical-performance claims.
"""

import math
import numpy as np

from model_config import (JOULES_PER_KJ, SCALE_M_PER_COORD, speed_to_mps,
                          UAV_LANDING_MIN, UAV_TAKEOFF_MIN, UAV_SERVICE_MIN,
                          SECONDS_PER_MINUTE)
from hardware_profiles import FLYCART30_SINGLE, STARSHIP_2017, UAVPhasePolicy
from rotor_energy import RotorParameters, momentum_uav_power, momentum_uav_power_torch

RHO = 1.225
M_UAV = FLYCART30_SINGLE.installed_mass_kg
C_D1 = 1.49
C_D2 = 2.2
A1 = 0.224
A2 = 0.1
ZETA = 1.4
N_ROTORS = 8
NU_UAV = 0.9
ROTOR_DIAMETER_M = FLYCART30_SINGLE.rotor_diameter_m
UAV_PARAMETERS = RotorParameters(M_UAV, N_ROTORS, ROTOR_DIAMETER_M, RHO,
                                 NU_UAV, C_D1 * A1 + C_D2 * A2)

C_R_ADR = 0.25
M_ADR = STARSHIP_2017.installed_mass_kg
G = 9.81
NU_ADR = 0.8

def _uav_power(payload_kg: float, v_ground: float, wind_vec: np.ndarray,
               heading_rad: float = 0.0) -> float:
    wind = np.asarray(wind_vec, dtype=float).reshape(2)
    if not np.isfinite([payload_kg, v_ground, heading_rad, *wind]).all():
        raise ValueError("Non-finite propulsion input")
    if payload_kg < 0 or v_ground < 0:
        raise ValueError("Payload and speed must be nonnegative")
    return float(momentum_uav_power(payload_kg, v_ground, wind, heading_rad, parameters=UAV_PARAMETERS))


def uav_power_torch(payload, v_ground, wind_xy=None, heading_rad=None):
    # Keep torch out of solver subprocesses until a tensor caller needs it.
    import torch
    dtype = torch.float64 if payload.dtype == torch.float64 or v_ground.dtype == torch.float64 else torch.float32
    payload = payload.to(dtype=dtype)
    v_ground = v_ground.to(device=payload.device, dtype=dtype)
    heading = torch.zeros_like(v_ground) if heading_rad is None else heading_rad.to(device=payload.device, dtype=dtype)
    wind = payload.new_zeros((*v_ground.shape, 2)) if wind_xy is None else wind_xy.to(device=payload.device, dtype=dtype)
    return momentum_uav_power_torch(payload, v_ground, wind, heading, parameters=UAV_PARAMETERS)


def uav_customer_service_energy_kj(load_before, load_after, wind_vec, *,
                                   phase_policy=None, terminal_delivery=False):
    """Two powered phases using the same law at zero horizontal ground speed.

    A declared hover-equivalent visit approximation, not descent/climb dynamics.
    Food-ready waiting and battery-swap waiting are not included.
    """
    policy = phase_policy or UAVPhasePolicy(UAV_LANDING_MIN, UAV_TAKEOFF_MIN)
    landing = _uav_power(load_before, 0., wind_vec) * policy.customer_landing_min
    takeoff_min = (0. if terminal_delivery and policy.terminal_delivery == 'landed'
                   else policy.customer_takeoff_min)
    takeoff = _uav_power(load_after, 0., wind_vec) * takeoff_min
    return (landing + takeoff) * SECONDS_PER_MINUTE / JOULES_PER_KJ


def uav_customer_service_energy_torch(load_before, load_after, wind_xy=None, *,
                                      phase_policy=None, terminal_delivery=False):
    """Torch counterpart of the customer-visit proxy; returns kJ."""
    import torch
    policy = phase_policy or UAVPhasePolicy(UAV_LANDING_MIN, UAV_TAKEOFF_MIN)
    takeoff_min = policy.customer_takeoff_min
    if policy.terminal_delivery == 'landed':
        terminal = torch.as_tensor(terminal_delivery, device=load_after.device, dtype=torch.bool)
        takeoff_min = torch.where(terminal, 0., takeoff_min)
    landing = uav_power_torch(load_before, torch.zeros_like(load_before), wind_xy) * policy.customer_landing_min
    takeoff = uav_power_torch(load_after, torch.zeros_like(load_after), wind_xy) * takeoff_min
    return (landing + takeoff) * SECONDS_PER_MINUTE / JOULES_PER_KJ


def uav_depot_phase_energy_kj(payload, wind_vec, *, departure, phase_policy):
    """Arrival is paid by the old pack; departure by the recharged/initial pack.

    None is an explicit omitted phase. Do not charge departure when an aircraft
    only lands and swaps: it is charged once on the next non-idle outbound leg.
    """
    minutes = phase_policy.depot_takeoff_min if departure else phase_policy.depot_landing_min
    return _uav_power(payload, 0., wind_vec) * (minutes or 0.) * SECONDS_PER_MINUTE / JOULES_PER_KJ


def uav_depot_phase_energy_torch(payload, wind_xy=None, *, departure, phase_policy):
    import torch
    minutes = phase_policy.depot_takeoff_min if departure else phase_policy.depot_landing_min
    return uav_power_torch(payload, torch.zeros_like(payload), wind_xy) * (minutes or 0.) * SECONDS_PER_MINUTE / JOULES_PER_KJ

def _adr_power(payload_kg: float, v_ground: float) -> float:
    return C_R_ADR * G * (M_ADR + payload_kg) * v_ground / NU_ADR

def _numpy(value):
    return value.detach().cpu().numpy() if hasattr(value, "detach") else np.asarray(value)


def arc_heading(instance, from_node, to_node):
    coordinates = instance.get("coordinates")
    if coordinates is None:
        wind = _numpy(instance.get("wind", [0.0, 0.0])).reshape(-1)
        if wind.size and abs(float(wind[0])) > 1e-12:
            raise ValueError("Wind requires coordinates; regenerate legacy instances")
        return 0.0
    xy = _numpy(coordinates)
    delta = xy[to_node] - xy[from_node]
    return math.atan2(float(delta[1]), float(delta[0]))


def coord_edge_energy_kj(mode, distance, payload, wind_vec, speed, heading_rad=0.0):
    metres = float(distance) * SCALE_M_PER_COORD
    if mode.lower() == "uav":
        return uav_edge_energy(metres, payload, wind_vec, speed_to_mps(speed),
                               heading_rad) / JOULES_PER_KJ
    if mode.lower() == "adr":
        return adr_edge_energy(metres, payload) / JOULES_PER_KJ
    raise ValueError(f"Unknown vehicle mode: {mode}")

UAV_LT_SECS = UAV_SERVICE_MIN * SECONDS_PER_MINUTE

def uav_energy_matrix(d_matrix: np.ndarray,
                       demand_vals: np.ndarray,
                       demand_full: np.ndarray,
                       n_depots: int,
                       n_req: int,
                       wind_vec: np.ndarray,
                       v_uav_max: float,
                       coordinates=None) -> np.ndarray:
    distances = np.asarray(d_matrix, dtype=float)
    if v_uav_max <= 0:
        raise ValueError("Cruise speed must be positive")
    wind = np.asarray(wind_vec, dtype=float).reshape(2)
    if coordinates is None:
        if np.any(np.abs(wind) > 1e-12):
            raise ValueError("Wind energy features require coordinates")
        heading = np.zeros_like(distances)
    else:
        xy = np.asarray(coordinates, dtype=float)
        delta = xy[None, :, :] - xy[:, None, :]
        heading = np.arctan2(delta[..., 1], delta[..., 0])
    power = momentum_uav_power(0., v_uav_max, wind, heading, parameters=UAV_PARAMETERS)
    energy = distances / v_uav_max * power
    energy[(~np.isfinite(distances)) | (distances >= 1e9)] = np.inf
    np.fill_diagonal(energy, 0.0)
    return energy.astype(np.float32)

def adr_energy_matrix(d_matrix: np.ndarray,
                       demand_vals: np.ndarray,
                       demand_full: np.ndarray,
                       n_depots: int,
                       n_req: int,
                       v_adr_max: float) -> np.ndarray:
    distances = np.asarray(d_matrix, dtype=float)
    energy = C_R_ADR * G * M_ADR * distances / NU_ADR
    energy[(~np.isfinite(distances)) | (distances >= 1e9)] = np.inf
    np.fill_diagonal(energy, 0.0)
    return energy.astype(np.float32)

def uav_edge_energy(d: float, payload: float, wind_vec: np.ndarray, v_uav: float,
                    heading_rad: float = 0.0) -> float:
    if not np.isfinite([d, payload, v_uav]).all() or d < 0 or payload < 0 or v_uav < 0:
        raise ValueError("Invalid arc energy input")
    if d == 0:
        return 0.0
    if v_uav == 0:
        raise ValueError("A positive-distance flight requires positive speed")
    t_travel = d / v_uav
    P = _uav_power(payload, v_uav, wind_vec, heading_rad)
    return P * t_travel

def adr_edge_energy(d: float, payload: float) -> float:
    if not np.isfinite([d, payload]).all() or d < 0 or payload < 0:
        raise ValueError("Invalid arc energy input")
    return C_R_ADR * G * (M_ADR + payload) * d / NU_ADR
