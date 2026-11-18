"""Compatibility entry point for the OR-Tools common-model baseline.

The old optional-service, transit-only CP-SAT circuit has been retired. It
omitted native battery, capacity and sharing constraints, forced a return arc
and could silently truncate during replay. Keeping its historical class name
does NOT imply that the replacement runs CP-SAT or is nonlinear-exact.
"""
from ce_cpdptw_rolling_baselines import (
    ExactRollingConfig, ORToolsRollingHorizonSolver,
    solve_static_instance_with_ortools_rolling,
    _make_forced_delivery_or_depot_route,
)

class CPSATVRPRollingHorizonSolver:
    """Historical API name; delegates explicitly to common linear SCIP/CBC.

    Uses exact affine ADR energy and conservative UAV payload intervals;
    independent scalar replay is mandatory. Inspect last_report for scope.
    """
    backend = "common_linear_milp_not_cp_sat"

    def __init__(self, time_limit_seconds=8., latest_pickup_slack=0.,
                 latest_delivery_slack=0., *, config=None, solver_id="SCIP"):
        if latest_pickup_slack or latest_delivery_slack:
            raise ValueError("Common five-cost model uses raw targets, not hard lateness windows")
        self.time_limit_seconds = time_limit_seconds
        self.config, self.solver_id = config, solver_id
        self.last_report = None

    def solve(self, residual):
        config = self.config
        if config is None:
            full, fleet = residual["full_instance"], residual["vehicles"]
            if "n_depots_uav" not in full:
                raise ValueError("Explicit config or n_depots_uav metadata required")
            nu = sum(v.normalized_mode() == "uav" for v in fleet.values())
            du = int(full["n_depots_uav"])
            config = ExactRollingConfig(nu, len(fleet) - nu, du, int(full["n_depots"]) - du,
                                       time_limit_seconds=self.time_limit_seconds)
        solver = ORToolsRollingHorizonSolver(config, self.solver_id)
        routes = solver.solve(residual)
        self.last_report = solver.last_report
        if self.last_report is not None:
            self.last_report["backend"] = self.backend
        return routes

def solve_static_instance_with_ortools_vrp_rolling(full_instance, *, n_uav, n_adr,
        n_depots_uav, n_depots_adr, depot_sharing=True, delta_minutes=5., shift_minutes=None,
        time_limit_seconds=8., latest_pickup_slack=0., latest_delivery_slack=0., solver_id="SCIP"):
    if latest_pickup_slack or latest_delivery_slack:
        raise ValueError("Historical hard-window CP-SAT benchmark is not a common-model comparator")
    config = ExactRollingConfig(n_uav, n_adr, n_depots_uav, n_depots_adr,
        depot_sharing=depot_sharing, delta_minutes=delta_minutes, shift_minutes=shift_minutes,
        time_limit_seconds=time_limit_seconds)
    return solve_static_instance_with_ortools_rolling(full_instance, config, solver_id=solver_id)
