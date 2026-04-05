"""MILP backend abstraction layer for the CBC/PuLP exact solver."""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

try:  # pragma: no cover - depends on environment
    import pulp
except Exception:  # pragma: no cover
    pulp = None


class _PulpGRB:
    """GRB-style status and variable-type constants used by existing MILP code."""

    BINARY = "Binary"
    INTEGER = "Integer"
    CONTINUOUS = "Continuous"

    MINIMIZE = 1
    MAXIMIZE = -1

    OPTIMAL = 2
    FEASIBLE = 1
    SUBOPTIMAL = 12
    TIME_LIMIT = 9
    INTERRUPTED = 11


@dataclass
class _PulpParams:
    """Solver parameter container compatible with the previous project's Param pattern."""

    OutputFlag: int = 0
    TimeLimit: int | float = 3600
    MIPGap: float = 0.0
    Threads: int = 1


class _PulpModel:
    """Minimal model object exposing the methods used by the project."""

    def __init__(self, name: str):
        if pulp is None:
            raise RuntimeError("PULP is not installed. Install `pulp` to enable CBC backend.")

        self._problem = pulp.LpProblem(name=name, sense=pulp.LpMinimize)
        self._objective = None
        self.Params = _PulpParams()
        self.Status = 0
        self.SolCount = 0
        self.ObjVal = math.nan
        self.MIPGap = math.nan

    def addVar(
        self,
        vtype: str | None = None,
        lb: float = 0.0,
        ub: float | None = None,
        name: str = "",
    ):
        if pulp is None:
            raise RuntimeError("PULP is not installed")
        cat = pulp.LpContinuous
        if vtype == _PulpGRB.BINARY:
            cat = pulp.LpBinary
        elif vtype == _PulpGRB.INTEGER:
            cat = pulp.LpInteger
        return pulp.LpVariable(name, lowBound=lb, upBound=ub, cat=cat)

    def addConstr(self, expr: Any, name: str | None = None) -> None:
        if name is not None:
            self._problem += expr, str(name)
        else:
            self._problem += expr

    def setObjective(self, expr: Any, sense: int) -> None:
        if sense == _PulpGRB.MINIMIZE:
            self._problem.sense = pulp.LpMinimize
        elif sense == _PulpGRB.MAXIMIZE:
            self._problem.sense = pulp.LpMaximize
        else:
            self._problem.sense = pulp.LpMinimize
        self._objective = expr
        self._problem.setObjective(expr)

    def optimize(self) -> None:
        if pulp is None:
            raise RuntimeError("PULP is not installed")
        time_limit = self.Params.TimeLimit if self.Params.TimeLimit > 0 else None
        threads = self.Params.Threads if self.Params.Threads and self.Params.Threads > 0 else None
        gap = self.Params.MIPGap if self.Params.MIPGap and self.Params.MIPGap > 0 else None
        solver = pulp.PULP_CBC_CMD(
            msg=bool(self.Params.OutputFlag),
            timeLimit=time_limit,
            threads=threads,
            gapRel=gap,
        )
        self._problem.solve(solver)

        status_name = pulp.LpStatus[self._problem.status]
        status_lower = status_name.lower()
        if status_lower == "optimal":
            self.Status = _PulpGRB.OPTIMAL
        elif status_lower in {"not solved", "suboptimal"}:
            self.Status = _PulpGRB.TIME_LIMIT
        elif status_lower in {"infeasible", "unbounded", "undefined"}:
            self.Status = 0
        elif status_lower == "integer infeasible":
            self.Status = 0
        else:
            self.Status = 0

        self.SolCount = 1 if any(v.varValue is not None for v in self._problem.variables()) else 0
        if self.SolCount and self._problem.objective is not None:
            self.ObjVal = float(pulp.value(self._problem.objective))
        else:
            self.ObjVal = math.nan

        if self.Status in {_PulpGRB.OPTIMAL, _PulpGRB.FEASIBLE, _PulpGRB.SUBOPTIMAL}:
            self.MIPGap = 0.0
        else:
            self.MIPGap = math.nan


class _PulpBackend:
    """Minimal PuLP-compatible object exposing the same surface used by the code."""

    Model = _PulpModel
    GRB = _PulpGRB

    @staticmethod
    def quicksum(values):
        if pulp is None:
            raise RuntimeError("PULP is not installed.")
        return pulp.lpSum(values)


def _build_backend(name: str):
    key = (name or "pulp_cbc").strip().lower()
    if key in {"pulp", "cbc", "pulp_cbc"}:
        if pulp is None:
            raise RuntimeError("pulp backend requested but pulp is not installed.")
        return _PulpBackend()
    if key in {"gurobi", "gurobi_persistent", "cplex", "scip"}:
        raise RuntimeError(
            f"backend '{key}' is not available in this build. Set milp.solver to 'pulp_cbc' "
            "or add a dedicated backend adapter."
        )
    raise ValueError(f"Unsupported MILP backend: {name}")


_ACTIVE_BACKEND = _build_backend("pulp_cbc")


def set_active_backend(name: str):
    """Select the backend used by subsequent model construction."""
    global _ACTIVE_BACKEND
    _ACTIVE_BACKEND = _build_backend(name)


def get_active_backend():
    """Return currently configured backend object."""
    return _ACTIVE_BACKEND


class _BackendProxy:
    """Dynamic attribute proxy so `from .backend import gp` follows active backend."""

    def __getattr__(self, name: str):
        return getattr(get_active_backend(), name)


gp = _BackendProxy()
