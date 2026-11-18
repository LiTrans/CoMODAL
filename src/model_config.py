"""Shared units and nominal parameters for the model."""

from hardware_profiles import FLYCART30_SINGLE, STARSHIP_2017, PhysicalConfiguration

MODEL_VERSION = "common-contract-8"
ENERGY_MODEL_VERSION = "steady-momentum-fc30-visit-provisional-2"
PHYSICAL_SAFETY_VERSION = "usable-soc-depot-or-terminal-1"
# Rated energy is retained as a hardware field. Unconfigured old pack arrays
# remain diagnostic inputs; only explicit physical_config binds their SOC window.
DEFAULT_PHYSICAL_CONFIG = PhysicalConfiguration()
GENERATOR_VERSION = "generator-3"
SCALE_M_PER_COORD = 200.0
SECONDS_PER_MINUTE = 60.0
JOULES_PER_KJ = 1000.0
KJ_PER_KWH = 3600.0
UAV_PROFILE = FLYCART30_SINGLE.identifier
ADR_PROFILE = STARSHIP_2017.identifier
UAV_BATTERY_KJ = FLYCART30_SINGLE.rated_energy_kj
UAV_SWAP_MIN = 10.0
ADR_SWAP_MIN = 20.0
ADR_BATTERY_KJ = STARSHIP_2017.rated_energy_kj
UAV_CAPACITY_KG = FLYCART30_SINGLE.payload_limit_kg
ADR_CAPACITY_KG = STARSHIP_2017.payload_limit_kg
UAV_CRUISE_MPS = FLYCART30_SINGLE.route_speed_mps
ADR_CRUISE_MPS = STARSHIP_2017.route_speed_mps
# Synthetic split of the existing two-minute visit, not measured flight phases.
UAV_LANDING_MIN = 1.0
UAV_TAKEOFF_MIN = 1.0
UAV_SERVICE_MIN = UAV_LANDING_MIN + UAV_TAKEOFF_MIN
ADR_SERVICE_MIN = 0.0


def speed_to_mps(coordinates_per_minute):
    return coordinates_per_minute * SCALE_M_PER_COORD / SECONDS_PER_MINUTE
