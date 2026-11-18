"""Gurobi backend for the shared, bounded actual-load routing formulation.

There is intentionally one constraint builder, shared with OR-Tools. Solver
optimality cannot certify nonlinear UAV optimality or physical calibration.
Importing this module does not initialize a license or contact a server.
"""
import time
import gurobipy as gp
from gurobipy import GRB
import ce_cpdptw_ortools as common
from ce_cpdptw_ortools import (
    Request, Vehicle, ArcPhysics, CECPDPTWData, DepotCopyMeta,
    SyntheticAdapterConfig, BuiltORToolsCECPDPTWModel,
    V_UAV_MAX, V_ADR_MAX, V_UAV_DEPOT, V_ADR_DEPOT,
    V_UAV_MIN_PICKUP, V_ADR_MIN_PICKUP, UAV_LAND_TAKEOFF_MIN,
    ADR_CUSTOMER_SERVICE_MIN, UAV_RECHARGE_MIN, ADR_RECHARGE_MIN,
    UAV_LOW_BATTERY_FRAC, ADR_LOW_BATTERY_FRAC, EXACT_CONTRACT_VERSION,
    energy_interval_bounds, build_ce_cpdptw_data_from_synthetic,
    select_instance, to_numpy, pc_uav_numpy, pc_adr_numpy,
    energy_joules_from_power, _validate_data,
)

BuiltCECPDPTWModel = BuiltORToolsCECPDPTWModel

class _GurobiLinearSolver:
    """Small backend adapter; algebra and constraints live in common builder."""
    def __init__(self, model):
        self.model = model
        self.objective = gp.LinExpr()

    def BoolVar(self, name):
        return self.model.addVar(vtype=GRB.BINARY, name=name)

    def NumVar(self, lo, hi, name):
        return self.model.addVar(lb=lo, ub=hi, vtype=GRB.CONTINUOUS, name=name)

    def Sum(self, terms):
        return gp.quicksum(terms)

    def Add(self, constraint):
        return self.model.addConstr(constraint)

    def Minimize(self, objective):
        self.objective = gp.LinExpr(objective)
        self.model.setObjective(self.objective, GRB.MINIMIZE)

    def Objective(self):
        return self

    def Value(self):
        return self.objective.getValue()

    def BestBound(self):
        return self.model.ObjBound

def build_ce_cpdptw_model(data, *, use_adaptive_speed=False,
                         tighten_battery_equalities=True, model_name="CE_CPDPTW_Common",
                         gurobi_env=None):
    # Reject invalid inputs before any license initialization.
    _validate_data(data)
    if use_adaptive_speed and data.pickup_time_by_speed:
        raise NotImplementedError("Speed-choice search not common-model certified")
    if not tighten_battery_equalities:
        raise ValueError("Common transitions require battery equalities")
    model = gp.Model(model_name, env=gurobi_env) if gurobi_env is not None else gp.Model(model_name)
    built = common.build_ce_cpdptw_ortools_model(data, use_adaptive_speed=use_adaptive_speed,
        tighten_battery_equalities=tighten_battery_equalities, _solver=_GurobiLinearSolver(model))
    built.model = model
    model.update()
    return built

def solve_ce_cpdptw(data, *, use_adaptive_speed=False, tighten_battery_equalities=True,
                    time_limit_seconds=3600., mip_gap=.01, log_to_console=False,
                    n_threads=0, gurobi_env=None, **kwargs):
    built = build_ce_cpdptw_model(data, use_adaptive_speed=use_adaptive_speed,
        tighten_battery_equalities=tighten_battery_equalities, gurobi_env=gurobi_env, **kwargs)
    model = built.model
    model.Params.LogToConsole = int(log_to_console)
    if time_limit_seconds is not None:
        model.Params.TimeLimit = float(time_limit_seconds)
    if mip_gap is not None:
        model.Params.MIPGap = 0. if data.service_policy == "residual_defer" else float(mip_gap)
    if n_threads:
        model.Params.Threads = int(n_threads)
    start = time.monotonic()
    model.optimize()
    if data.service_policy == "residual_defer" and model.SolCount:
        built.deferral_cardinality_proven = model.Status == GRB.OPTIMAL
        common.finish_residual_cardinality(built, lambda var: var.X)
        if mip_gap is not None:
            model.Params.MIPGap = float(mip_gap)
        if time_limit_seconds is not None:
            remaining = time_limit_seconds - (time.monotonic() - start)
            if remaining <= 0:
                built.status = None
                return built
            model.Params.TimeLimit = remaining
        model.optimize()
    built.status = model.Status
    return built

def solve_synthetic_instance(instance, config, *, batch_index=0, **kwargs):
    data = build_ce_cpdptw_data_from_synthetic(select_instance(instance, batch_index), config)
    built = solve_ce_cpdptw(data, **kwargs)
    return built, extract_solution(built)

def extract_solution(built, *, tolerance=.5):
    return common.extract_solution(built, tolerance=tolerance, value_fn=lambda var: var.X,
        feasible=built.status is not None and built.model.SolCount > 0)

def write_debug_files(built, *, prefix="ce_cpdptw_debug"):
    built.model.write(f"{prefix}.lp")
