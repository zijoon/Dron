"""Small instance runner: exact MILP + heuristic for comparison."""
from __future__ import annotations

import math
from datetime import datetime
from pathlib import Path
from typing import List

import pandas as pd

from ..data_models import ExperimentResult
from ..feasibility import is_feasible
from ..heuristics import run_nils
from ..instance_generator import InstanceGenerator
from ..milp import solve_instance
from ..parameters import SearchConfig
from ..heuristics.nils import summarize_drone_usage


def _build_progress_writer(output_dir: str | None) -> Path:
    out_dir = Path(output_dir or "outputs")
    log_dir = out_dir / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    return log_dir / f"run_small_progress_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"


def _append_progress(path: Path, message: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%H:%M:%S")
    line = f"[{timestamp}] {message}"
    print(line)
    with path.open("a", encoding="utf-8") as fh:
        fh.write(f"{line}\n")


def _progress_bar(current: int, total: int, width: int = 24) -> str:
    if total <= 0:
        return "[------------------------] 0/0"
    safe_total = max(1, int(total))
    ratio = min(1.0, max(0.0, current / safe_total))
    fill = int(ratio * width)
    return "[" + "#" * fill + "-" * (width - fill) + f"] {current}/{safe_total}"


def _safe_solve_exact(instance: object, config: SearchConfig):
    if not config.milp.enabled:
        return None
    try:
        return solve_instance(instance, config)
    except RuntimeError as ex:
        message = str(ex).lower()
        if "gurobi is not" in message or "unavailable" in message or "not installed" in message:
            return None
        raise


def _to_sizes(config: SearchConfig) -> List[int]:
    explicit = config.experiment.get("run_small_sizes")
    if explicit:
        return [int(v) for v in explicit]
    if config.generation.sizes:
        return [int(v) for v in config.generation.sizes]
    return [20, 50, 100, 150, 200]


def run_small_exact(config: SearchConfig, output_dir: str | None = None) -> List[ExperimentResult]:
    exp_cfg = dict(config.experiment or {})
    sizes = _to_sizes(config)
    reps = int(exp_cfg.get("instance_reps_per_size", config.generation.instance_reps_per_size))

    generator = InstanceGenerator.from_search_config(config)
    target_dir = output_dir or str(exp_cfg.get("output_dir", "outputs"))
    instances = generator.generate_batch(config.seed, sizes, reps, target_dir, tag="small")
    progress_path = _build_progress_writer(target_dir)
    _append_progress(progress_path, f"run_small_exact start | instances={len(instances)} | output_dir={target_dir}")

    rows: List[ExperimentResult] = []
    table_rows = []

    for idx, inst in enumerate(instances, start=1):
        bar = _progress_bar(idx, len(instances))
        _append_progress(
            progress_path,
            f"{bar} | instance={inst.name} | customers={inst.num_customers} | drones={inst.num_drones} | trucks={inst.num_trucks}",
        )
        heuristic = run_nils(
            inst,
            seed=config.seed,
            max_iter=int(config.heuristics.max_outer_iter),
            max_no_improve=int(config.heuristics.max_no_improve),
            time_limit=int(config.heuristics.time_limit_seconds),
        )
        heuristic_usage = summarize_drone_usage(heuristic)
        _append_progress(
            progress_path,
            f"heuristic_done | {inst.name} | status={heuristic.status} | obj={heuristic.objective:.4f} | time={heuristic.run_time_seconds:.2f}s",
        )

        exact = _safe_solve_exact(inst, config)
        if exact is not None:
            _append_progress(
                progress_path,
                f"exact_done | {inst.name} | status={exact.status} | obj={exact.objective:.4f} | time={exact.run_time_seconds:.2f}s",
            )
        else:
            _append_progress(progress_path, f"exact_unavailable | {inst.name} | status={'exact_unavailable'}")
        exact_gap = None
        if exact is not None and math.isfinite(exact.objective) and exact.objective > 1e-9:
            exact_gap = (heuristic.objective - exact.objective) / exact.objective

        rows.append(
            ExperimentResult(
                instance_name=inst.name,
                method="heuristic",
                status=heuristic.status,
                runtime_seconds=heuristic.run_time_seconds,
                objective=float(heuristic.objective),
                gap_to_baseline=exact_gap if exact is not None else None,
                notes=(
                    "exact_enabled"
                    if (config.milp.enabled and exact is not None)
                    else ("exact_disabled" if not config.milp.enabled else "exact_unavailable")
                ),
                stats={
                    "drone_served_customers": heuristic_usage["drone_served_customers"],
                    "drone_arcs": heuristic_usage["drone_arcs"],
                    "reload_events": heuristic_usage["reload_events"],
                    "battery_swaps": heuristic_usage["battery_swaps"],
                    "nonempty_drone_routes": heuristic_usage["nonempty_drone_routes"],
                    "heuristic_tardiness_cost": float(heuristic.components.get("tardiness_cost", 0.0)),
                    "exact_objective": float(exact.objective) if exact else math.nan,
                },
            )
        )

        if exact is not None:
            exact_usage = summarize_drone_usage(exact)
            rows.append(
                ExperimentResult(
                    instance_name=inst.name,
                    method="exact",
                    status=exact.status,
                    runtime_seconds=exact.run_time_seconds,
                    objective=float(exact.objective),
                    stats={
                        "drone_served_customers": exact_usage["drone_served_customers"],
                        "drone_arcs": exact_usage["drone_arcs"],
                        "reload_events": exact_usage["reload_events"],
                        "battery_swaps": exact_usage["battery_swaps"],
                        "nonempty_drone_routes": exact_usage["nonempty_drone_routes"],
                        "heuristic_objective": float(heuristic.objective),
                        "exact_gap": exact_gap if exact_gap is not None else float("nan"),
                    },
                )
            )
        else:
            unknown = {
                "drone_served_customers": math.nan,
                "drone_arcs": math.nan,
                "reload_events": math.nan,
                "battery_swaps": math.nan,
                "nonempty_drone_routes": math.nan,
            }
            rows.append(
                ExperimentResult(
                    instance_name=inst.name,
                    method="exact",
                    status="disabled" if not config.milp.enabled else "unavailable",
                    runtime_seconds=0.0,
                    objective=float("nan"),
                    notes="milp_disabled" if not config.milp.enabled else "milp_unavailable",
                    stats=unknown,
                )
            )

        if exact is not None:
            exact_usage = summarize_drone_usage(exact)
        else:
            exact_usage = {
                "drone_served_customers": math.nan,
                "drone_arcs": math.nan,
                "reload_events": math.nan,
                "battery_swaps": math.nan,
                "nonempty_drone_routes": math.nan,
            }

        table_rows.append(
            {
                "instance": inst.name,
                "method": "exact" if exact else "exact_unavailable",
                "exact_obj": float(exact.objective) if exact else math.nan,
                "heur_obj": float(heuristic.objective),
                "seed": inst.seed,
                "size": inst.num_customers,
                "feasible_exact": bool(exact and exact.status != "no_solution" and is_feasible(inst, exact)),
                "feasible_heur": is_feasible(inst, heuristic),
                "gap_exact_minus_heur": float(heuristic.objective - exact.objective) if exact else math.nan,
                "runtime_exact": float(exact.run_time_seconds) if exact else 0.0,
                "runtime_heuristic": float(heuristic.run_time_seconds),
                "exact_available": bool(config.milp.enabled and exact is not None),
                "exact_status": exact.status if exact else "unavailable",
                "heur_drone_served_customers": heuristic_usage["drone_served_customers"],
                "heur_drone_arcs": heuristic_usage["drone_arcs"],
                "heur_reload_events": heuristic_usage["reload_events"],
                "heur_battery_swaps": heuristic_usage["battery_swaps"],
                "heur_nonempty_drone_routes": heuristic_usage["nonempty_drone_routes"],
                "exact_drone_served_customers": exact_usage["drone_served_customers"],
                "exact_drone_arcs": exact_usage["drone_arcs"],
                "exact_reload_events": exact_usage["reload_events"],
                "exact_battery_swaps": exact_usage["battery_swaps"],
                "exact_nonempty_drone_routes": exact_usage["nonempty_drone_routes"],
            }
        )
    _append_progress(progress_path, "run_small_exact complete")

    out_dir = Path(target_dir) / "tables"
    out_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(table_rows).to_csv(out_dir / "small_exact_summary.csv", index=False)
    return rows
