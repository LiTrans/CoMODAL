"""Source-backed hardware fields, separate from uncalibrated power parameters.

Profiles are selected from published vehicle specs. Rated electrical energy is
not a promise of usable flight range. Do not infer efficiency from these fields.
"""
from dataclasses import asdict, dataclass, field
import math
from typing import Mapping, Optional


@dataclass(frozen=True)
class HardwareProfile:
    identifier: str
    installed_mass_kg: float
    rated_energy_wh: float
    payload_limit_kg: float
    route_speed_mps: float
    maximum_speed_mps: float
    source_url: str
    rotor_count: int = 0
    rotor_diameter_m: float = 0.0

    def __post_init__(self):
        values = (self.installed_mass_kg, self.rated_energy_wh,
                  self.payload_limit_kg, self.route_speed_mps, self.maximum_speed_mps)
        if not all(math.isfinite(x) and x > 0 for x in values):
            raise ValueError('Hardware quantities must be finite and positive')
        if self.route_speed_mps > self.maximum_speed_mps:
            raise ValueError('Route speed exceeds the hardware ceiling')

    @property
    def rated_energy_kj(self):
        return self.rated_energy_wh * 3.6


FLYCART30_SINGLE = HardwareProfile(
    identifier='dji-flycart30-single-db2000',
    installed_mass_kg=42.5 + 11.3,
    rated_energy_wh=1984.4,
    payload_limit_kg=40.0,
    route_speed_mps=15.0,  # DJI's maximum automatic-route speed, constant here.
    maximum_speed_mps=20.0,
    rotor_count=8, rotor_diameter_m=1.375,
    source_url='https://www.dji.com/flycart-30/specs',
)

STARSHIP_2017 = HardwareProfile(
    identifier='starship-swiss-post-2017',
    installed_mass_kg=23.0,
    rated_energy_wh=8.0 * 18.5,  # 8000 mAh, 18.5 V in the operator factsheet.
    payload_limit_kg=10.0,
    route_speed_mps=6.0 / 3.6,
    maximum_speed_mps=6.0 / 3.6,
    source_url='https://www.post.ch/-/media/post/ueber-uns/medienmitteilungen/2017/factsheet-lieferroboter.pdf?la=en&vs=1',
)

HARDWARE_PROFILES = {'uav': FLYCART30_SINGLE, 'adr': STARSHIP_2017}


@dataclass(frozen=True)
class BatteryPolicy:
    """SOC is relative to derated capacity, not a fraction of the route budget.

    Remaining battery states include the protected reserve. Recharge restores
    ``usable_energy_kj * recharge_soc``; every powered phase must retain
    ``usable_energy_kj * reserve_soc`` before a swap. These defaults reproduce
    the rated-energy diagnostic, NOT a measured usable-energy configuration.
    """
    profile_id: str
    usable_fraction: float = 1.0
    initial_soc: float = 1.0
    recharge_soc: float = 1.0
    reserve_soc: float = 0.0
    provenance: str = 'simulation_assumption_unvalidated'
    evidence_id: Optional[str] = None

    def __post_init__(self):
        if self.profile_id not in {p.identifier for p in HARDWARE_PROFILES.values()}:
            raise ValueError('Unknown hardware profile')
        values = (self.usable_fraction, self.initial_soc, self.recharge_soc, self.reserve_soc)
        if not all(math.isfinite(x) for x in values):
            raise ValueError('Non-finite battery policy')
        if not (0 < self.usable_fraction <= 1 and 0 <= self.reserve_soc < self.recharge_soc <= 1
                and self.reserve_soc <= self.initial_soc <= self.recharge_soc):
            raise ValueError('Invalid usable fraction or SOC window')
        if self.provenance not in {'simulation_assumption_unvalidated', 'independent_measurement'}:
            raise ValueError('Unknown battery provenance')
        if self.provenance == 'independent_measurement' and not self.evidence_id:
            raise ValueError('Measured battery policy needs an evidence identifier')

    @property
    def profile(self):
        return next(p for p in HARDWARE_PROFILES.values() if p.identifier == self.profile_id)

    @property
    def rated_energy_kj(self):
        return self.profile.rated_energy_kj

    @property
    def usable_energy_kj(self):
        return self.rated_energy_kj * self.usable_fraction

    @property
    def full_charge_kj(self):
        return self.usable_energy_kj * self.recharge_soc

    @property
    def initial_charge_kj(self):
        return self.usable_energy_kj * self.initial_soc

    @property
    def reserve_energy_kj(self):
        return self.usable_energy_kj * self.reserve_soc

    @property
    def dispatch_budget_kj(self):
        return self.initial_charge_kj - self.reserve_energy_kj


@dataclass(frozen=True)
class UAVPhasePolicy:
    """Explicit hover-equivalent assumptions, not measured vertical dynamics.

    Completed customer visits are airborne: landing uses pre-service payload,
    takeoff uses post-service payload. Depot arrivals are landed; a subsequent
    departure (including initial dispatch) charges takeoff to its new pack.
    ``None`` means a documented accounting gap, not measured zero consumption.
    A landed terminal delivery omits takeoff; it never entails a depot return.
    """
    customer_landing_min: float = 1.0
    customer_takeoff_min: float = 1.0
    depot_landing_min: Optional[float] = None
    depot_takeoff_min: Optional[float] = None
    terminal_delivery: str = 'customer_pair'
    provenance: str = 'simulation_assumption_unvalidated'
    evidence_id: Optional[str] = None

    def __post_init__(self):
        values = (self.customer_landing_min, self.customer_takeoff_min,
                  self.depot_landing_min, self.depot_takeoff_min)
        if any(x is not None and (not math.isfinite(x) or x < 0) for x in values):
            raise ValueError('Phase durations must be finite and nonnegative')
        if self.terminal_delivery not in {'customer_pair', 'landed'}:
            raise ValueError('Unknown terminal delivery accounting')
        if self.provenance not in {'simulation_assumption_unvalidated', 'independent_measurement'}:
            raise ValueError('Unknown phase provenance')
        if self.provenance == 'independent_measurement' and not self.evidence_id:
            raise ValueError('Measured phase policy needs an evidence identifier')

    @property
    def has_accounting_gaps(self):
        return self.depot_landing_min is None or self.depot_takeoff_min is None

    def customer_minutes(self, terminal=False):
        takeoff = 0. if terminal and self.terminal_delivery == 'landed' else self.customer_takeoff_min
        return self.customer_landing_min + takeoff


@dataclass(frozen=True)
class PhysicalConfiguration:
    uav_battery: BatteryPolicy = field(default_factory=lambda: BatteryPolicy(FLYCART30_SINGLE.identifier))
    adr_battery: BatteryPolicy = field(default_factory=lambda: BatteryPolicy(STARSHIP_2017.identifier))
    uav_phases: UAVPhasePolicy = field(default_factory=UAVPhasePolicy)
    continuation_policy: str = 'depot_or_terminal'

    def __post_init__(self):
        if self.uav_battery.profile_id != FLYCART30_SINGLE.identifier or self.adr_battery.profile_id != STARSHIP_2017.identifier:
            raise ValueError('Battery policy does not match the selected mode/profile')
        if self.continuation_policy not in {'depot_or_terminal', 'legacy_leg_only'}:
            raise ValueError('Unknown continuation policy')

    def battery_for(self, mode):
        if mode.lower() not in HARDWARE_PROFILES:
            raise ValueError(f'Unknown mode: {mode}')
        return self.uav_battery if mode.lower() == 'uav' else self.adr_battery

    def to_dict(self):
        return asdict(self)


def physical_configuration(value=None):
    """Strict structured loader; missing metadata remains visibly provisional.

    Callers store ``config.to_dict()`` under an instance's ``physical_config``.
    Unknown fields are rejected rather than silently ignored.
    """
    if value is None:
        return PhysicalConfiguration()
    if isinstance(value, PhysicalConfiguration):
        return value
    if not isinstance(value, Mapping):
        raise TypeError('physical_config must be a mapping or PhysicalConfiguration')
    data = dict(value)
    for key, cls in (('uav_battery', BatteryPolicy), ('adr_battery', BatteryPolicy),
                     ('uav_phases', UAVPhasePolicy)):
        if key in data and not isinstance(data[key], cls):
            data[key] = cls(**data[key])
    return PhysicalConfiguration(**data)
