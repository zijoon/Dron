"""Heuristic route construction utilities."""
from __future__ import annotations

import math
from typing import Dict, List, Sequence, Tuple

import numpy as np

from ..data_models import InstanceData, SolutionData
from ..distance_utils import euclidean_distance, manhattan_distance

Priority = str

def _priority_rank(priority: Priority) -> int:
    return {"high": 0, "medium": 1, "low": 2}.get(priority, 2)


def _distance(a: Tuple[float, float], b: Tuple[float, float], mode: str = "manhattan") -> float:
    return manhattan_distance(a, b) if mode == "manhattan" else euclidean_distance(a, b)


def _route_from_order(order: Sequence[int], nodes: Dict[int, Tuple[float, float]], speed: float) -> Tuple[List[Tuple[int, float]], float]:
    arr: List[Tuple[int, float]] = [(0, 0.0)]
    current = 0
    t = 0.0
    for j in order:
        d = _distance(nodes[current], nodes[j], mode="manhattan")
        t += d / speed * 60.0
        arr.append((j, t))
        current = j
    # return to depot
    d = _distance(nodes[current], nodes[0], mode="manhattan")
    t += d / speed * 60.0
    arr.append((0, t))
    return arr, t


def construct_greedy_truck_routes(instance: InstanceData, rng: np.random.Generator) -> Dict[int, List[int]]:
    customer_ids = [c.node_id for c in instance.customers]
    customers_by_class = sorted(
        customer_ids,
        key=lambda c: _priority_rank(instance.customer_map()[c].priority),
    )
    trucks = [t.truck_id for t in instance.trucks]

    routes = {t: [0] for t in trucks}
    residual_load = {t: instance.trucks[t - 1].capacity_kg for t in trucks}
    coords = {0: (0.0, 0.0)}
    coords.update({c.node_id: (c.x, c.y) for c in instance.customers})

    unserved = set(customers_by_class)
    while unserved:
        c_id = next(iter(unserved))
        candidate = instance.customer_map()[c_id]

        best_pair = None
        best_extra = math.inf
        for t in trucks:
            if residual_load[t] < candidate.demand_kg:
                continue
            r = routes[t]
            last = r[-1]
            d_to = _distance(coords[last], coords[c_id], mode="manhattan")
            d_home = _distance(coords[c_id], (0.0, 0.0), mode="manhattan")
            # small greedy insertion score
            score = d_to + d_home + 0.1 * _priority_rank(candidate.priority)
            if score < best_extra:
                best_extra = score
                best_pair = (t, c_id)

        if best_pair is None:
            # open a new start with least-loaded truck due to soft violation
            t = rng.choice(trucks)
            routes[t].append(c_id)
        else:
            t, cid = best_pair
            routes[t].append(cid)
            residual_load[t] -= instance.customer_map()[cid].demand_kg
        unserved.remove(c_id)

    for t in routes:
        routes[t].append(0)
    return routes


def _fill_solution_from_routes(
    instance: InstanceData,
    truck_routes: Dict[int, List[int]],
) -> SolutionData:
    nodes = {0: (0.0, 0.0)}
    for c in instance.customers:
        nodes[c.node_id] = (c.x, c.y)

    sol = SolutionData(
        instance_name=instance.name,
        status="heuristic_feasible",
        objective=0.0,
        components={},
        seed=None,
        truck_routes={k: v.copy() for k, v in truck_routes.items()},
    )

    for t, route in truck_routes.items():
        truck = instance.trucks[t - 1]
        arr, _ = _route_from_order(route[1:-1], nodes, truck.speed_kmph)
        for i in range(len(route)):
            frm = route[i]
            arr_t = arr[i][1]
            sol.l_truck[(t, frm)] = arr_t
            sol.a_truck[(t, frm)] = arr_t
            if i == len(route) - 1:
                continue
            nxt = route[i + 1]
            sol.x_truck[(frm, nxt, t)] = 1
            if nxt != 0:
                sol.u_truck[(t, nxt)] = 1

        for j in range(1, len(route) - 1):
            node = route[j]
            ci = instance.customer_map()[node]
            tard = max(0.0, arr[j][1] - ci.ub)
            sol.tardiness[node] = tard

    for d in [dr.drone_id for dr in instance.drones]:
        for j in range(1, instance.num_customers + 1):
            sol.u_drone[(d, j)] = 0
            for i in range(0, instance.num_customers + 1):
                sol.x_drone[(i, j, d)] = 0
                sol.x_drone[(j, i, d)] = 0

    for d in instance.drone_ids():
        # initialize empty route return to depot
        if d not in sol.drone_routes:
            sol.drone_routes[d] = [0, 0]

    # keep default drone timing fields if no drone routes are constructed
    for d in instance.drone_ids():
        for j in range(0, instance.num_customers + 1):
            sol.a_drone.setdefault((d, j), 0.0)
            sol.l_drone.setdefault((d, j), 0.0)

    return sol


def build_initial_solution(instance: InstanceData, seed: int) -> SolutionData:
    rng = np.random.default_rng(seed)
    truck_routes = construct_greedy_truck_routes(instance, rng)

    # simple post-processing: remove empty or repeated nodes
    cleaned = {}
    for t, r in truck_routes.items():
        seen = set()
        out = [0]
        for j in r[1:]:
            if j in seen or j == 0:
                continue
            seen.add(j)
            out.append(j)
        if out[-1] != 0:
            out.append(0)
        cleaned[t] = out

    return _fill_solution_from_routes(instance, cleaned)
