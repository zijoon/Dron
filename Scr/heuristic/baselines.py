"""Baseline methods for ablation and comparison runs."""
from __future__ import annotations

from dataclasses import dataclass
from typing import List

from ..data_models import InstanceData, SolutionData
from ..parameters import SearchConfig
from .construction import build_initial_solution
from .nils import (
    _eligible_customer_ids,
    default_drone_to_truck,
    evaluate_solution,
    move_to_drone,
    run_nils,
)


@dataclass
class BaselineResult:
    name: str
    solution: SolutionData


def _clone_solution(solution: SolutionData) -> SolutionData:
    return SolutionData(**{k: v for k, v in solution.__dict__.items()})


def truck_only_baseline(instance: InstanceData, config: SearchConfig) -> BaselineResult:
    sol = build_initial_solution(instance, config.seed)
    obj, comp = evaluate_solution(instance, sol)
    sol.objective = obj
    sol.components = comp
    sol.status = "baseline_truck_only"
    return BaselineResult(name="truck_only", solution=sol)


def paired_baseline(instance: InstanceData, config: SearchConfig) -> BaselineResult:
    drone_to_truck = default_drone_to_truck(instance)
    obj = run_nils(
        instance,
        seed=config.seed + 9,
        max_iter=max(1, config.heuristics.max_outer_iter),
        max_no_improve=max(1, config.heuristics.max_no_improve),
        time_limit=max(10, min(120, config.heuristics.time_limit_seconds)),
        drone_to_truck=drone_to_truck,
        use_paired_initialization=True,
        max_initial_pairings=2,
    )
    obj.status = "baseline_paired"
    return BaselineResult(name="paired_baseline", solution=obj)


def no_priority_baseline(instance: InstanceData, config: SearchConfig) -> BaselineResult:
    # run with flattened penalties temporarily
    original = dict(instance.priority_penalties)
    try:
        for key in instance.priority_penalties:
            instance.priority_penalties[key] = 1.0
        sol = run_nils(instance, config.seed + 1, max_iter=max(1, config.heuristics.max_outer_iter // 2))
        sol.status = "baseline_no_priority"
        return BaselineResult(name="no_priority", solution=sol)
    finally:
        instance.priority_penalties.update(original)


def no_unpairing_baseline(instance: InstanceData, config: SearchConfig) -> BaselineResult:
    drone_to_truck = default_drone_to_truck(instance)
    obj = run_nils(
        instance,
        seed=config.seed + 2,
        max_iter=max(1, config.heuristics.max_outer_iter // 2),
        max_no_improve=max(1, config.heuristics.max_no_improve),
        time_limit=max(10, min(120, config.heuristics.time_limit_seconds)),
        drone_to_truck=drone_to_truck,
        use_paired_initialization=True,
        max_initial_pairings=2,
    )
    obj.status = "baseline_no_unpairing"
    return BaselineResult(name="no_unpairing", solution=obj)


def simple_drone_assignment_baseline(instance: InstanceData, config: SearchConfig) -> BaselineResult:
    """One-pass greedy reassignment from truck backbone to drones."""
    current = build_initial_solution(instance, config.seed + 31)
    current.objective, current.components = evaluate_solution(instance, current)
    eligible = _eligible_customer_ids(instance)
    customers = [c for c in instance.customer_ids if c in eligible and any(current.u_truck.get((t, c), 0) == 1 for t in instance.truck_ids())]
    customers.sort(key=lambda cid: (instance.customer_map()[cid].ub, cid))

    for customer in customers:
        best = None
        best_obj = current.objective
        for drone in instance.drone_ids():
            candidate = move_to_drone(instance, current, customer, drone)
            if candidate is None:
                continue
            cand_obj, cand_comp = evaluate_solution(instance, candidate)
            if cand_obj < best_obj - 1e-12:
                candidate.objective = cand_obj
                candidate.components = cand_comp
                best = candidate
                best_obj = cand_obj
        if best is not None:
            current = best

    current.objective, current.components = evaluate_solution(instance, current)
    current.status = "baseline_simple_drone"
    return BaselineResult(name="simple_drone", solution=current)


def random_feasible_reassignment_baseline(instance: InstanceData, config: SearchConfig) -> BaselineResult:
    """Weak benchmark: random feasible truck->drone reassignments."""
    import random

    rnd = random.Random(config.seed + 47)
    current = build_initial_solution(instance, config.seed + 47)
    current.objective, current.components = evaluate_solution(instance, current)
    eligible = list(_eligible_customer_ids(instance))
    rnd.shuffle(eligible)

    moves = max(1, min(len(eligible), 2 * max(1, instance.num_drones)))
    applied = 0
    for customer in eligible:
        if applied >= moves:
            break
        if not any(current.u_truck.get((t, customer), 0) == 1 for t in instance.truck_ids()):
            continue
        drone_order = instance.drone_ids()
        rnd.shuffle(drone_order)
        for drone in drone_order:
            candidate = move_to_drone(instance, current, customer, drone)
            if candidate is None:
                continue
            candidate.objective, candidate.components = evaluate_solution(instance, candidate)
            current = candidate
            applied += 1
            break

    current.objective, current.components = evaluate_solution(instance, current)
    current.status = "baseline_random_feasible"
    return BaselineResult(name="random_feasible", solution=current)


def nils_no_local_search_baseline(instance: InstanceData, config: SearchConfig) -> BaselineResult:
    sol = run_nils(
        instance,
        seed=config.seed + 73,
        max_iter=max(1, config.heuristics.max_outer_iter),
        max_no_improve=max(1, config.heuristics.max_no_improve),
        time_limit=max(10, min(180, config.heuristics.time_limit_seconds)),
        enable_local_search=False,
    )
    sol.status = "ablation_no_local_search"
    return BaselineResult(name="nils_no_local_search", solution=sol)


def nils_no_perturbation_baseline(instance: InstanceData, config: SearchConfig) -> BaselineResult:
    sol = run_nils(
        instance,
        seed=config.seed + 79,
        max_iter=max(1, config.heuristics.max_outer_iter),
        max_no_improve=max(1, config.heuristics.max_no_improve),
        time_limit=max(10, min(180, config.heuristics.time_limit_seconds)),
        enable_perturbation=False,
    )
    sol.status = "ablation_no_perturbation"
    return BaselineResult(name="nils_no_perturbation", solution=sol)


def nils_no_battery_screening_baseline(instance: InstanceData, config: SearchConfig) -> BaselineResult:
    sol = run_nils(
        instance,
        seed=config.seed + 83,
        max_iter=max(1, config.heuristics.max_outer_iter),
        max_no_improve=max(1, config.heuristics.max_no_improve),
        time_limit=max(10, min(180, config.heuristics.time_limit_seconds)),
        battery_aware_screening=False,
    )
    sol.status = "ablation_no_battery_screening"
    return BaselineResult(name="nils_no_battery_screening", solution=sol)


def run_all_baselines(instance: InstanceData, config: SearchConfig) -> List[BaselineResult]:
    return [
        truck_only_baseline(instance, config),
        simple_drone_assignment_baseline(instance, config),
        random_feasible_reassignment_baseline(instance, config),
        paired_baseline(instance, config),
        no_priority_baseline(instance, config),
        no_unpairing_baseline(instance, config),
        nils_no_local_search_baseline(instance, config),
        nils_no_perturbation_baseline(instance, config),
        nils_no_battery_screening_baseline(instance, config),
    ]
