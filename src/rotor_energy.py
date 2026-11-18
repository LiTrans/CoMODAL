"""Reference momentum model, also used by the provisional active energy law.

Implements Stolaroff et al. (2018), doi:10.1038/s41467-017-02411-5,
Eqs. 1-3 and 6-8, using directional airspeed in steady level flight.
Explicit vehicle parameters are mandatory. Tests establish numerical
conformance, not calibration of a particular UAV. All inputs use SI units.
"""

from dataclasses import asdict, dataclass
import hashlib
import json
import math

import numpy as np


@dataclass(frozen=True)
class RotorParameters:
    empty_mass_kg: float  # Includes the installed battery.
    rotor_count: int
    rotor_diameter_m: float
    air_density_kg_m3: float
    efficiency: float
    drag_cd_area_m2: float  # Sum of component Cd * projected area.
    gravity_m_s2: float = 9.81

    def fingerprint(self):
        """Bind calibration evidence to the actual coefficients, not a label."""
        encoded = json.dumps(asdict(self), sort_keys=True, allow_nan=False).encode('ascii')
        return hashlib.sha256(encoded).hexdigest()

    def __post_init__(self):
        values = (self.empty_mass_kg, self.rotor_diameter_m,
                  self.air_density_kg_m3, self.efficiency,
                  self.drag_cd_area_m2, self.gravity_m_s2)
        if not np.isfinite(values).all() or any(x <= 0 for x in values):
            raise ValueError("Rotor parameters must be finite and positive")
        if (not isinstance(self.rotor_count, int)
                or isinstance(self.rotor_count, bool) or self.rotor_count <= 0):
            raise ValueError("Rotor count must be a positive integer")
        if self.efficiency > 1:
            raise ValueError("Power efficiency must not exceed one")


def _disk_coefficient(parameters):
    return (2.0 / (math.pi * parameters.rotor_count
                   * parameters.rotor_diameter_m ** 2
                   * parameters.air_density_kg_m3))


def induced_velocity(thrust_n, airspeed_m_s, pitch_rad, *, parameters):
    """Solve Eq. 7 for arrays without per-arc SciPy calls.

    Scale vi by its hover value h. The positive root of
    y * hypot(r*cos(alpha), r*sin(alpha)+y) = 1 lies in (0, 1].
    For level-flight pitch in [0, pi/2], the residual is increasing and
    convex, so Newton steps from the upper bound converge from above.
    """
    thrust, speed, pitch = np.broadcast_arrays(
        np.asarray(thrust_n, dtype=np.float64),
        np.asarray(airspeed_m_s, dtype=np.float64),
        np.asarray(pitch_rad, dtype=np.float64))
    if (not all(np.isfinite(x).all() for x in (thrust, speed, pitch))
            or np.any(thrust <= 0) or np.any(speed < 0)
            or np.any(pitch < 0) or np.any(pitch > math.pi / 2)):
        raise ValueError("Invalid steady level-flight inputs")
    hover = np.sqrt(_disk_coefficient(parameters) * thrust)
    r = speed / hover
    horizontal, vertical = r * np.cos(pitch), r * np.sin(pitch)
    y = np.ones_like(r)
    for _ in range(12):
        s = np.hypot(horizontal, vertical + y)
        residual = y * s - 1.0
        derivative = s + y * (vertical + y) / s
        y = np.clip(y - residual / derivative, np.finfo(y.dtype).tiny, 1.0)
    residual = y * np.hypot(horizontal, vertical + y) - 1.0
    if not np.isfinite(residual).all() or np.any(np.abs(residual) > 1e-12):
        raise ArithmeticError("Induced-velocity solver did not converge")
    return hover * y


def induced_velocity_torch(thrust_n, airspeed_m_s, pitch_rad, *, parameters):
    """Device-local float32/float64 counterpart; no host root solver."""
    import torch

    inputs = (thrust_n, airspeed_m_s, pitch_rad)
    if any(not isinstance(x, torch.Tensor) for x in inputs):
        raise TypeError("All root-solver inputs must be tensors")
    if thrust_n.dtype not in (torch.float32, torch.float64):
        raise ValueError("Physics requires float32 or float64")
    if any(x.dtype != thrust_n.dtype or x.device != thrust_n.device for x in inputs):
        raise ValueError("Root-solver tensors must share dtype and device")
    thrust, speed, pitch = torch.broadcast_tensors(*inputs)
    invalid = (~torch.isfinite(thrust) | ~torch.isfinite(speed)
               | ~torch.isfinite(pitch) | (thrust <= 0) | (speed < 0)
               | (pitch < 0) | (pitch > math.pi / 2))
    if bool(invalid.any()):
        raise ValueError("Invalid steady level-flight inputs")
    hover = torch.sqrt(_disk_coefficient(parameters) * thrust)
    r = speed / hover
    horizontal, vertical = r * torch.cos(pitch), r * torch.sin(pitch)
    y = torch.ones_like(r)
    for _ in range(12):
        s = torch.hypot(horizontal, vertical + y)
        residual = y * s - 1.0
        derivative = s + y * (vertical + y) / s
        y = (y - residual / derivative).clamp(torch.finfo(y.dtype).tiny, 1.0)
    residual = y * torch.hypot(horizontal, vertical + y) - 1.0
    tolerance = max(1e-12, 32 * torch.finfo(y.dtype).eps)
    if bool((~torch.isfinite(residual) | (residual.abs() > tolerance)).any()):
        raise ArithmeticError("Induced-velocity solver did not converge")
    return hover * y


def momentum_uav_power(payload_kg, ground_speed_m_s, wind_xy_m_s,
                       heading_rad, *, parameters):
    """Steady propulsion watts, with wind vector pointing downwind.

    This replicates the source's additive thrust approximation (Eq. 1).
    It excludes acceleration, altitude change and service/hover duration.
    """
    payload, speed, heading = np.broadcast_arrays(
        np.asarray(payload_kg, dtype=np.float64),
        np.asarray(ground_speed_m_s, dtype=np.float64),
        np.asarray(heading_rad, dtype=np.float64))
    wind = np.asarray(wind_xy_m_s, dtype=np.float64)
    if wind.ndim == 0 or wind.shape[-1] != 2:
        raise ValueError("Wind must have a final xy dimension of two")
    if (not all(np.isfinite(x).all() for x in (payload, speed, heading, wind))
            or np.any(payload < 0) or np.any(speed < 0)):
        raise ValueError("Invalid payload, ground speed, heading or wind")
    airspeed = np.hypot(speed * np.cos(heading) - wind[..., 0],
                        speed * np.sin(heading) - wind[..., 1])
    drag = parameters.air_density_kg_m3 * parameters.drag_cd_area_m2 * airspeed ** 2 / 2
    weight = parameters.gravity_m_s2 * (parameters.empty_mass_kg + payload)
    thrust, pitch = drag + weight, np.arctan2(drag, weight)
    vi = induced_velocity(thrust, airspeed, pitch, parameters=parameters)
    return thrust * (airspeed * np.sin(pitch) + vi) / parameters.efficiency


def momentum_uav_power_torch(payload_kg, ground_speed_m_s, wind_xy_m_s,
                             heading_rad, *, parameters):
    """Vectorized tensor version of the reference power calculation."""
    import torch

    inputs = (payload_kg, ground_speed_m_s, wind_xy_m_s, heading_rad)
    if any(not isinstance(x, torch.Tensor) for x in inputs):
        raise TypeError("All power inputs must be tensors")
    if any(x.dtype != payload_kg.dtype or x.device != payload_kg.device for x in inputs):
        raise ValueError("Power tensors must share dtype and device")
    if wind_xy_m_s.ndim == 0 or wind_xy_m_s.shape[-1] != 2:
        raise ValueError("Wind must have a final xy dimension of two")
    if (any(bool((~torch.isfinite(x)).any()) for x in inputs)
            or bool((payload_kg < 0).any()) or bool((ground_speed_m_s < 0).any())):
        raise ValueError("Invalid payload, ground speed, heading or wind")
    airspeed = torch.hypot(ground_speed_m_s * torch.cos(heading_rad) - wind_xy_m_s[..., 0],
                           ground_speed_m_s * torch.sin(heading_rad) - wind_xy_m_s[..., 1])
    drag = parameters.air_density_kg_m3 * parameters.drag_cd_area_m2 * airspeed.square() / 2
    weight = parameters.gravity_m_s2 * (parameters.empty_mass_kg + payload_kg)
    thrust, pitch = drag + weight, torch.atan2(drag, weight)
    vi = induced_velocity_torch(thrust, airspeed, pitch, parameters=parameters)
    return thrust * (airspeed * torch.sin(pitch) + vi) / parameters.efficiency
