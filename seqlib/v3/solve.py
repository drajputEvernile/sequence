"""v3 solvers: the evidence costs plus the scan prior (the move margin), then one order.

Nodes: 0 = START / END (the order is a loop through it), 1..n = the pages in scan order.
cost(a, b) = -log P_evidence(b follows a) + margin, unless b is a's scan successor
(a -> a + 1, START -> 1, n -> END), which pays no margin. A pair the candidate step never scored
gets P = FLOOR.

* ``search`` -- the v2 decoder on these costs: exact (Held-Karp) up to 11 pages, else block moves
  and reversals from the scan order while the total cost drops.

* ``cover`` -- design doc 7.5: the assignment problem (each page picks one successor and one
  predecessor, globally cheapest; scipy) gives the loop through START plus, possibly, separate
  cycles; each cycle is broken at its dearest link into a chain and inserted where it costs least;
  then the block-move search polishes the result.

Anchors (v4+, design doc 7.4.1): a forced link a -> b costs nothing and every other link out of a
or into b costs ANCHOR_PENALTY more (finite, so the search can still compare orders); the search
then starts from the scan order with each anchored chain kept together.
"""
from __future__ import annotations

import numpy as np

from seqlib.v2.decode import EXACT_MAX, cost_matrix, held_karp, local_search, path_cost

ANCHOR_PENALTY = 25.0


def v3_costs(n: int, edge: dict[tuple[int, int], float], margin: float) -> np.ndarray:
    cost = cost_matrix(n, edge)
    if margin:
        scan = np.zeros_like(cost, dtype=bool)
        for a in range(n + 1):
            scan[a, (a + 1) % (n + 1)] = True  # 0->1, 1->2, ..., n->0
        cost = np.where(scan | ~np.isfinite(cost), cost, cost + margin)
    return cost


def _cover(cost: np.ndarray) -> list[int]:
    from scipy.optimize import linear_sum_assignment

    n = cost.shape[0] - 1
    big = np.where(np.isfinite(cost), cost, 1e9)
    rows, cols = linear_sum_assignment(big)
    succ = dict(zip(rows.tolist(), cols.tolist()))
    # the loop through START
    main, k, seen = [], succ[0], {0}
    while k != 0 and k not in seen:
        main.append(k)
        seen.add(k)
        k = succ[k]
    # other cycles -> chains, broken at their dearest link
    chains = []
    for start in range(1, n + 1):
        if start in seen:
            continue
        cycle, k = [], start
        while k not in seen:
            cycle.append(k)
            seen.add(k)
            k = succ[k]
        dearest = max(range(len(cycle)), key=lambda i: cost[cycle[i], cycle[(i + 1) % len(cycle)]])
        chains.append(cycle[dearest + 1:] + cycle[:dearest + 1])
    # cheapest insertion, chains taken in the scan order of their first page
    for chain in sorted(chains, key=lambda c: min(c)):
        full = [0, *main, 0]
        best, at = None, 0
        for i in range(len(full) - 1):
            x, y = full[i], full[i + 1]
            delta = cost[x, chain[0]] + cost[chain[-1], y] - cost[x, y]
            if best is None or delta < best:
                best, at = delta, i
        main = main[:at] + chain + main[at:]
    return main


def apply_anchors(cost: np.ndarray, anchors) -> np.ndarray:
    cost = cost.copy()
    for a, b in anchors:
        keep = cost[a, b]
        cost[a, :] = np.where(np.isfinite(cost[a, :]), cost[a, :] + ANCHOR_PENALTY, cost[a, :])
        cost[:, b] = np.where(np.isfinite(cost[:, b]), cost[:, b] + ANCHOR_PENALTY, cost[:, b])
        cost[a, b] = 1e-6 if np.isfinite(keep) else keep
    return cost


def decode_v3(n: int, edge: dict[tuple[int, int], float], margin: float, solver: str = "search",
              max_block: int = 40, anchors=()) -> list[int]:
    if n == 0:
        return []
    cost = v3_costs(n, edge, margin)
    begin = list(range(1, n + 1))
    if anchors:
        from seqlib.ladder.extra import anchor_order

        cost = apply_anchors(cost, anchors)
        begin = anchor_order(n, anchors)
    if n <= EXACT_MAX:
        return held_karp(cost)
    if solver == "cover":
        start = _cover(cost)
        polished = local_search(cost, start, max_block=max_block)
        return polished if path_cost(polished, cost) <= path_cost(start, cost) else start
    return local_search(cost, begin, max_block=max_block)
