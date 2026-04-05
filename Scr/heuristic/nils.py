"""Nested Iterative Local Search (NILS) implementation used in experiments."""
from __future__ import annotations

import math
import random
import time
from dataclasses import dataclass
from typing import Dict, List, Tuple

import numpy as np

from ..data_models import InstanceData, SolutionData
from ..distance_utils import euclidean_distance, manhattan_distance
from .acceptance import metropolis
from .construction import build_initial_solution
from .local_search import apply_two_opt_to_solution
from .perturbation import relocate_random_customer


def _travel_time(distance_km: float, speed_kmph: float) -> float:
    if speed_kmph <= 0:
        return math.inf
    return distance_km / speed_kmph * 60.0


def evaluate_solution(instance: InstanceData, solution: SolutionData) -> Tuple[float, Dict[str, float]]:
    coords = {0: (0.0, 0.0)}
    coords.update({c.node_id: (c.x, c.y) for c in instance.customers})

    arrival_truck: Dict[Tuple[int, int], float] = {}
    arrival_drone: Dict[Tuple[int, int], float] = {}
    served_time: Dict[int, float] = {j: 0.0 for j in range(1, instance.num_customers + 1)}

    truck_cost = 0.0
    drone_cost = 0.0
    solution.waiting_truck = {}
    solution.waiting_drone = {}

    for t, route in solution.truck_routes.items():
        truck = instance.trucks[t - 1]
        cur_t = 0.0
        for i in range(len(route) - 1):
            frm = route[i]
            to = route[i + 1]
            if i > 0:
                prev = route[i - 1]
                if prev != 0:
                    cur_t += instance.customer_map()[prev].service_time_min
            dist = manhattan_distance(coords[frm], coords[to])
            cur_t += _travel_time(dist, truck.speed_kmph)
            arrival_truck[(t, to)] = cur_t
            if to != 0:
                served_time[to] = max(served_time[to], cur_t)
                solution.a_truck[(t, to)] = cur_t
            solution.l_truck[(t, to)] = cur_t
            truck_cost += truck.cost_per_km * dist

    for d, route in solution.drone_routes.items():
        drone = instance.drones[d - 1]
        cur_t = 0.0
        for i in range(len(route) - 1):
            frm = route[i]
            to = route[i + 1]
            if i > 0:
                prev = route[i - 1]
                if prev != 0:
                    cur_t += instance.customer_map()[prev].service_time_min
            dist = euclidean_distance(coords[frm], coords[to])
            cur_t += _travel_time(dist, drone.speed_kmph)
            arrival_drone[(d, to)] = cur_t
            if to != 0:
                served_time[to] = max(served_time[to], cur_t)
                solution.a_drone[(d, to)] = cur_t
            solution.l_drone[(d, to)] = cur_t
            drone_cost += drone.cost_per_km * dist

    tardiness_cost = 0.0
    for c in range(1, instance.num_customers + 1):
        delivered_at = served_time[c]
        ci = instance.customer_map()[c]
        penalty = max(0.0, delivered_at - ci.ub)
        solution.tardiness[c] = penalty
        tardiness_cost += penalty * instance.priority_penalties[ci.priority]

    sync_delay_total = 0.0
    for (node, drone_id, truck_id), val in solution.z1.items():
        if val <= 0.5:
            continue
        at = solution.a_truck.get((truck_id, node), 0.0)
        ad = solution.a_drone.get((drone_id, node), 0.0)
        delay = abs(at - ad)
        sync_delay_total += delay
        if at < ad:
            solution.waiting_truck[(truck_id, node)] = delay
        elif ad < at:
            solution.waiting_drone[(drone_id, node)] = delay
    for (node, drone_id, truck_id), val in solution.z2.items():
        if val <= 0.5:
            continue
        at = solution.a_truck.get((truck_id, node), 0.0)
        ad = solution.a_drone.get((drone_id, node), 0.0)
        delay = abs(at - ad)
        sync_delay_total += delay
        if at < ad:
            solution.waiting_truck[(truck_id, node)] = max(solution.waiting_truck.get((truck_id, node), 0.0), delay)
        elif ad < at:
            solution.waiting_drone[(drone_id, node)] = max(solution.waiting_drone.get((drone_id, node), 0.0), delay)

    return truck_cost + drone_cost + tardiness_cost + sync_delay_total, {
        "truck_cost": truck_cost,
        "drone_cost": drone_cost,
        "tardiness_cost": tardiness_cost,
        "waiting_cost": sync_delay_total,
        "waiting_sync_cost": sync_delay_total,
    }


def summarize_drone_usage(solution: SolutionData) -> Dict[str, int]:
    """Return compact diagnostics for whether drones were actually used."""
    drone_served_customers = {
        c for (d, c), v in solution.u_drone.items() if c > 0 and v == 1
    }
    if not drone_served_customers:
        for route in solution.drone_routes.values():
            for c in route:
                if c != 0:
                    drone_served_customers.add(c)

    drone_arcs = 0
    nonempty_drone_routes = 0
    for route in solution.drone_routes.values():
        if any(node != 0 for node in route):
            nonempty_drone_routes += 1
            if len(route) >= 2:
                drone_arcs += len(route) - 1

    reload_events = sum(1 for (_, _, _), v in solution.z1.items() if v > 0.5)
    battery_swaps = sum(1 for (_, _, _), v in solution.z2.items() if v > 0.5)

    return {
        "drone_served_customers": len(drone_served_customers),
        "drone_arcs": int(drone_arcs),
        "reload_events": int(reload_events),
        "battery_swaps": int(battery_swaps),
        "nonempty_drone_routes": int(nonempty_drone_routes),
    }


def default_drone_to_truck(instance: InstanceData) -> Dict[int, int]:
    """Deterministic round-robin default fixed-pair assignment map: drone -> truck."""
    trucks = instance.truck_ids()
    return {d: trucks[(d - 1) % len(trucks)] for d in instance.drone_ids()}


def _normalize_drone_to_truck(
    instance: InstanceData,
    drone_to_truck: Dict[int, int] | None,
) -> Dict[int, int]:
    if not drone_to_truck:
        return {}
    valid_trucks = set(instance.truck_ids())
    normalized: Dict[int, int] = {}
    for drone_id in instance.drone_ids():
        truck_id = int(drone_to_truck.get(drone_id, 0))
        if truck_id not in valid_trucks:
            return {}
        normalized[drone_id] = truck_id
    if set(drone_to_truck.keys()) - set(instance.drone_ids()):
        return {}
    return normalized


def _eligible_customer_ids(instance: InstanceData) -> set[int]:
    """Return customers eligible for drone service under the active scenario metadata."""
    raw = instance.metadata.get("drone_eligible_customers")
    if isinstance(raw, list) and raw:
        valid = {int(v) for v in raw if int(v) in set(instance.customer_ids)}
        if valid:
            return valid
    max_drone_cap = max((d.capacity_kg for d in instance.drones), default=0.0)
    return {c.node_id for c in instance.customers if c.demand_kg <= max_drone_cap}


def _service_mask(solution: SolutionData) -> Dict[int, str]:
    mask: Dict[int, str] = {}
    for (t, c), v in solution.u_truck.items():
        if v == 1:
            mask[c] = f"truck_{t}"
    for (d, c), v in solution.u_drone.items():
        if v == 1:
            mask[c] = f"drone_{d}"
    return mask


def _empty_solution_clone(solution: SolutionData) -> SolutionData:
    return SolutionData(
        instance_name=solution.instance_name,
        status=solution.status,
        objective=solution.objective,
        components=dict(solution.components),
        run_time_seconds=solution.run_time_seconds,
        x_truck=dict(solution.x_truck),
        x_drone=dict(solution.x_drone),
        u_truck=dict(solution.u_truck),
        u_drone=dict(solution.u_drone),
        z1=dict(solution.z1),
        z2=dict(solution.z2),
        y_loaded=dict(solution.y_loaded),
        a_truck=dict(solution.a_truck),
        l_truck=dict(solution.l_truck),
        a_drone=dict(solution.a_drone),
        l_drone=dict(solution.l_drone),
        waiting_truck=dict(solution.waiting_truck),
        waiting_drone=dict(solution.waiting_drone),
        load_truck=dict(solution.load_truck),
        battery_drone=dict(solution.battery_drone),
        tardiness=dict(solution.tardiness),
        sync_events=list(solution.sync_events),
        truck_routes={k: v.copy() for k, v in solution.truck_routes.items()},
        drone_routes={k: v.copy() for k, v in solution.drone_routes.items()},
        seed=solution.seed,
        tags=solution.tags,
    )


def _paired_trip_feasible(
    instance: InstanceData,
    drone: int,
    launch_node: int,
    customer: int,
    recovery_node: int,
    coords: Dict[int, Tuple[float, float]],
) -> bool:
    drone_obj = instance.drones[drone - 1]
    lc = _travel_time(euclidean_distance(coords[launch_node], coords[customer]), drone_obj.speed_kmph)
    cr = _travel_time(euclidean_distance(coords[customer], coords[recovery_node]), drone_obj.speed_kmph)
    if lc == math.inf or cr == math.inf:
        return False
    burn = drone_obj.energy_per_min_when_loaded if drone_obj.energy_per_min_when_loaded > 0 else 20.0
    energy = (lc + cr) * burn
    return energy <= drone_obj.max_battery_wh + 1e-9


def _drone_route_energy(instance: InstanceData, drone: int, route: List[int], coords: Dict[int, Tuple[float, float]]) -> float:
    drone_obj = instance.drones[drone - 1]
    consumed = 0.0
    for idx in range(len(route) - 1):
        frm = route[idx]
        to = route[idx + 1]
        arc_time = _travel_time(euclidean_distance(coords[frm], coords[to]), drone_obj.speed_kmph)
        loaded = to != 0
        burn = drone_obj.energy_per_min_when_loaded if loaded else drone_obj.energy_per_min_when_empty
        if burn <= 0:
            burn = 20.0 if loaded else 15.0
        consumed += burn * arc_time
    return consumed


def move_to_drone(
    instance: InstanceData,
    solution: SolutionData,
    customer: int,
    drone: int,
    *,
    launch_node: int | None = None,
    recovery_node: int | None = None,
    source_truck: int | None = None,
    enforce_battery: bool = True,
) -> SolutionData | None:
    """Generate candidate where one customer is moved from truck to a drone route."""
    service = _service_mask(solution)
    src = None
    if service.get(customer, "").startswith("truck_"):
        src = int(service[customer].split("_")[1])
    if source_truck is not None:
        src = source_truck
    if src is None or src <= 0 or src > instance.num_trucks:
        return None
    if customer <= 0 or drone <= 0 or drone > instance.num_drones:
        return None

    ci = instance.customer_map()[customer]
    if ci.demand_kg > instance.drones[drone - 1].capacity_kg:
        return None

    candidate = _empty_solution_clone(solution)

    if src in candidate.truck_routes and customer in candidate.truck_routes[src]:
        candidate.truck_routes[src] = [j for j in candidate.truck_routes[src] if j != customer]
        if len(candidate.truck_routes[src]) == 1:
            candidate.truck_routes[src] = [0, 0]
        candidate.u_truck[(src, customer)] = 0

    for d in instance.drone_ids():
        candidate.u_drone[(d, customer)] = 1 if d == drone else 0

    route = candidate.drone_routes.get(drone, [0, 0])
    if not route:
        route = [0, 0]
    if route[0] != 0:
        route = [0] + route
    if route[-1] != 0:
        route = route + [0]
    if customer not in route:
        route.insert(len(route) - 1, customer)
    if enforce_battery:
        coords = instance.coordinate_map()
        total_energy = _drone_route_energy(instance, drone, route, coords)
        if total_energy > instance.drones[drone - 1].max_battery_wh + 1e-9:
            return None
    candidate.drone_routes[drone] = route
    candidate.x_drone = {}

    if source_truck is not None and launch_node is not None and recovery_node is not None:
        candidate.z1[(launch_node, drone, source_truck)] = 1
        candidate.z2[(recovery_node, drone, source_truck)] = 1

    return candidate


def _extract_best_displacement(
    instance: InstanceData,
    solution: SolutionData,
    *,
    coords: Dict[int, Tuple[float, float]],
    eligible_customers: set[int],
    battery_aware_screening: bool,
    stats: Dict[str, int] | None = None,
) -> SolutionData | None:
    best_sol = None
    best_obj = float("inf")

    for customer in range(1, instance.num_customers + 1):
        if customer not in eligible_customers:
            continue
        truck_served = any(solution.u_truck.get((t, customer), 0) == 1 for t in instance.truck_ids())
        if not truck_served:
            continue
        for d in range(1, instance.num_drones + 1):
            if battery_aware_screening and not _paired_trip_feasible(instance, d, 0, customer, 0, coords):
                if stats is not None:
                    stats["battery_infeasible_rejections"] += 1
                continue
            candidate = move_to_drone(
                instance,
                solution,
                customer,
                d,
                enforce_battery=battery_aware_screening,
            )
            if candidate is None:
                if battery_aware_screening and stats is not None:
                    stats["battery_infeasible_rejections"] += 1
                continue
            if stats is not None:
                stats["candidate_reassignments_evaluated"] += 1
            obj, comp = evaluate_solution(instance, candidate)
            if obj < best_obj:
                candidate.objective = obj
                candidate.components = comp
                best_obj = obj
                best_sol = candidate
    return best_sol


def _extract_best_paired_displacement(
    instance: InstanceData,
    solution: SolutionData,
    drone_to_truck: Dict[int, int],
    coords: Dict[int, Tuple[float, float]],
    *,
    eligible_customers: set[int],
    battery_aware_screening: bool,
    stats: Dict[str, int] | None = None,
) -> SolutionData | None:
    best_sol = None
    best_obj = float("inf")

    for drone, truck in drone_to_truck.items():
        route = solution.truck_routes.get(truck, [0, 0])
        if len(route) < 3:
            continue

        for launch_idx in range(0, len(route) - 2):
            launch_node = route[launch_idx]
            for customer_idx in range(launch_idx + 1, len(route) - 1):
                customer = route[customer_idx]
                if customer == 0:
                    continue
                if customer not in eligible_customers:
                    continue
                if solution.u_truck.get((truck, customer), 0) != 1:
                    continue
                for recovery_idx in range(customer_idx + 1, len(route)):
                    recovery_node = route[recovery_idx]
                    if recovery_node == launch_node or recovery_node == customer:
                        continue
                    if battery_aware_screening:
                        if not _paired_trip_feasible(
                            instance,
                            drone,
                            launch_node,
                            customer,
                            recovery_node,
                            coords,
                        ):
                            if stats is not None:
                                stats["battery_infeasible_rejections"] += 1
                            continue
                    candidate = move_to_drone(
                        instance,
                        solution,
                        customer,
                        drone,
                        launch_node=launch_node,
                        recovery_node=recovery_node,
                        source_truck=truck,
                        enforce_battery=battery_aware_screening,
                    )
                    if candidate is None:
                        if battery_aware_screening and stats is not None:
                            stats["battery_infeasible_rejections"] += 1
                        continue
                    if stats is not None:
                        stats["candidate_reassignments_evaluated"] += 1
                    obj, comp = evaluate_solution(instance, candidate)
                    if obj < best_obj:
                        candidate.objective = obj
                        candidate.components = comp
                        best_obj = obj
                        best_sol = candidate
    return best_sol


def _apply_initial_paired_sorties(
    instance: InstanceData,
    solution: SolutionData,
    drone_to_truck: Dict[int, int],
    max_pairs: int,
    *,
    eligible_customers: set[int],
    battery_aware_screening: bool,
    stats: Dict[str, int] | None = None,
) -> SolutionData:
    if max_pairs <= 0:
        return solution
    coords = instance.coordinate_map()
    current = solution
    for _ in range(max_pairs):
        candidate = _extract_best_paired_displacement(
            instance,
            current,
            drone_to_truck,
            coords,
            eligible_customers=eligible_customers,
            battery_aware_screening=battery_aware_screening,
            stats=stats,
        )
        if candidate is None:
            break
        current = candidate
    return current


@dataclass
class NILSResult:
    solution: SolutionData
    iterations: int
    improved: int


def run_nils(
    instance: InstanceData,
    seed: int,
    max_iter: int = 30,
    max_no_improve: int = 5,
    time_limit: int = 600,
    drone_to_truck: Dict[int, int] | None = None,
    use_paired_initialization: bool = False,
    max_initial_pairings: int = 2,
    enable_local_search: bool = True,
    enable_perturbation: bool = True,
    battery_aware_screening: bool = True,
) -> SolutionData:
    """Run the NILS routine.

    If `drone_to_truck` is provided, the algorithm uses fixed-pair neighborhoods where
    each drone can only launch and recover on its assigned truck.
    """
    rng = np.random.default_rng(seed)
    rnd = random.Random(seed)

    current = build_initial_solution(instance, seed)
    current.objective, current.components = evaluate_solution(instance, current)

    pairing_map = _normalize_drone_to_truck(instance, drone_to_truck)
    if pairing_map and len(pairing_map) != instance.num_drones:
        pairing_map = {}
    if pairing_map:
        pairing_map = dict(pairing_map)
    else:
        pairing_map = {}

    eligible_customers = _eligible_customer_ids(instance)
    stats = {
        "iterations": 0,
        "improving_moves_accepted": 0,
        "candidate_reassignments_evaluated": 0,
        "battery_infeasible_rejections": 0,
        "perturbation_moves": 0,
        "local_search_calls": 0,
    }

    if use_paired_initialization and pairing_map:
        current = _apply_initial_paired_sorties(
            instance,
            current,
            pairing_map,
            max_pairs=max_initial_pairings,
            eligible_customers=eligible_customers,
            battery_aware_screening=battery_aware_screening,
            stats=stats,
        )
        current.objective, current.components = evaluate_solution(instance, current)

    start = time.time()
    best = _empty_solution_clone(current)
    best.status = "nils_best"
    no_improve = 0

    coords = instance.coordinate_map()
    for it in range(1, max_iter + 1):
        if time.time() - start > time_limit:
            break
        stats["iterations"] = it

        if pairing_map:
            candidate = _extract_best_paired_displacement(
                instance,
                current,
                pairing_map,
                coords,
                eligible_customers=eligible_customers,
                battery_aware_screening=battery_aware_screening,
                stats=stats,
            )
        else:
            candidate = _extract_best_displacement(
                instance,
                current,
                coords=coords,
                eligible_customers=eligible_customers,
                battery_aware_screening=battery_aware_screening,
                stats=stats,
            )

        if candidate is None:
            if enable_perturbation:
                candidate = relocate_random_customer(current, instance, rng)
                stats["perturbation_moves"] += 1
            else:
                no_improve += 1
                if no_improve >= max_no_improve:
                    break
                continue
        candidate.objective, candidate.components = evaluate_solution(instance, candidate)
        if enable_perturbation and it % 4 == 0:
            perturbed = relocate_random_customer(candidate, instance, rng)
            stats["perturbation_moves"] += 1
            perturbed.objective, perturbed.components = evaluate_solution(instance, perturbed)
            if perturbed.objective < candidate.objective or rnd.random() < 0.2:
                candidate = perturbed

        temp = max(1e-6, float(max_iter - it + 1))
        if metropolis(current.objective, candidate.objective, temp, rnd):
            if enable_local_search:
                candidate = apply_two_opt_to_solution(candidate, instance)
                candidate.objective, candidate.components = evaluate_solution(instance, candidate)
                stats["local_search_calls"] += 1
            current = candidate
            no_improve = 0
            if candidate.objective < best.objective - 1e-12:
                best = _empty_solution_clone(candidate)
                stats["improving_moves_accepted"] += 1
        else:
            no_improve += 1

        if no_improve >= max_no_improve:
            break

    best.status = "nils_complete"
    best.run_time_seconds = time.time() - start
    best.components = dict(best.components)
    best.components["iterations"] = float(stats["iterations"])
    best.components["improving_moves_accepted"] = float(stats["improving_moves_accepted"])
    best.components["candidate_reassignments_evaluated"] = float(stats["candidate_reassignments_evaluated"])
    best.components["battery_infeasible_rejections"] = float(stats["battery_infeasible_rejections"])
    best.components["perturbation_moves"] = float(stats["perturbation_moves"])
    best.components["local_search_calls"] = float(stats["local_search_calls"])
    best.components["battery_screening_enabled"] = 1.0 if battery_aware_screening else 0.0
    best.components["local_search_enabled"] = 1.0 if enable_local_search else 0.0
    best.components["perturbation_enabled"] = 1.0 if enable_perturbation else 0.0
    return best
