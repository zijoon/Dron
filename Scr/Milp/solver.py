"""Solver entry-point for exact MILP optimization."""
from __future__ import annotations

import time
from typing import List

from ..parameters import SearchConfig
from ..data_models import InstanceData, SolutionData
from .extract_solution import extract_solution
from .model_builder import build_model
from .backend import gp


def solve_instance(instance: InstanceData, config: SearchConfig) -> SolutionData:
    """Build and solve one MILP instance."""
    artifacts = build_model(instance, config)
    model = artifacts.model

    if config.milp.threads > 0:
        model.Params.Threads = config.milp.threads
    if config.milp.time_limit_seconds > 0:
        model.Params.TimeLimit = config.milp.time_limit_seconds
    if config.milp.mip_gap > 0:
        model.Params.MIPGap = config.milp.mip_gap

    t0 = time.time()
    model.optimize()
    runtime = time.time() - t0

    if model.SolCount == 0:
        return SolutionData(
            instance_name=instance.name,
            status="no_solution",
            objective=float("inf"),
            components={},
            run_time_seconds=runtime,
        )

    status = str(model.Status)
    if model.Status not in (gp.GRB.OPTIMAL, gp.GRB.TIME_LIMIT, gp.GRB.SUBOPTIMAL, gp.GRB.FEASIBLE, gp.GRB.INTERRUPTED):
        status = f"unsolved_{model.Status}"

    sol = extract_solution(instance, artifacts, status=status, runtime_seconds=runtime)
    sol.components["runtime_seconds"] = runtime
    if hasattr(model, "MIPGap"):
        sol.components["optimality_gap"] = float(model.MIPGap)
    sol.objective = float(model.ObjVal) if hasattr(model, "ObjVal") else sol.objective
    return sol


def solve_multiple(instances: List[InstanceData], config: SearchConfig) -> List[SolutionData]:
    """Solve many instances one by one."""
    return [solve_instance(instance, config) for instance in instances]
