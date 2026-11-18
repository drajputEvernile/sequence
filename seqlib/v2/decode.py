"""Turn pair probabilities into one order.

The order is a path START -> p1 -> ... -> pN -> START. Its cost is the sum of
``-log P(next | previous)`` over its links; the best order is the cheapest path (an asymmetric
travelling-salesman problem). Charts are small enough that:

* up to ``EXACT_MAX`` pages: exact (Held-Karp dynamic programme);
* above that: local search from the scan order, moving any block of up to ``max_block`` pages to
  any other place, forwards or reversed, while the cost drops. Starting from the scan order and
  accepting only improvements keeps the scan order wherever the model has no strong reason to
  change it.
"""
from __future__ import annotations

import math
from typing import Sequence

import numpy as np

EXACT_MAX = 11
FLOOR = 1e-6  # probability of a pair the candidate step never scored


def cost_matrix(n: int, probs: dict[tuple[int, int], float]) -> np.ndarray:
    """(n+1) x (n+1) costs over node 0 = START and nodes 1..n = pages in scan order."""
    cost = np.full((n + 1, n + 1), -math.log(FLOOR))
    for (a, b), p in probs.items():
        cost[a, b] = -math.log(max(FLOOR, min(1.0, p)))
    np.fill_diagonal(cost, np.inf)
    return cost


def path_cost(order: Sequence[int], cost: np.ndarray) -> float:
    full = [0, *order, 0]
    return float(sum(cost[a, b] for a, b in zip(full, full[1:])))


def held_karp(cost: np.ndarray) -> list[int]:
    n = cost.shape[0] - 1
    if n <= 1:
        return list(range(1, n + 1))
    full = 1 << n
    dp = np.full((full, n), np.inf)
    parent = np.full((full, n), -1, dtype=np.int64)
    for j in range(n):
        dp[1 << j, j] = cost[0, j + 1]
    for mask in range(full):
        row = dp[mask]
        for j in range(n):
            if not (mask >> j) & 1 or row[j] == np.inf:
                continue
            base = row[j]
            for k in range(n):
                if (mask >> k) & 1:
                    continue
                m2 = mask | (1 << k)
                value = base + cost[j + 1, k + 1]
                if value < dp[m2, k]:
                    dp[m2, k] = value
                    parent[m2, k] = j
    last_mask = full - 1
    end = int(np.argmin(dp[last_mask] + cost[1:, 0]))
    order, mask, j = [], last_mask, end
    while j != -1:
        order.append(j + 1)
        prev = int(parent[mask, j])
        mask ^= 1 << j
        j = prev
    return order[::-1]


def local_search(cost: np.ndarray, start: Sequence[int], max_block: int = 40, max_rounds: int = 200) -> list[int]:
    """Block moves (with optional reversal) from ``start`` while the path gets cheaper."""
    order = list(start)
    n = len(order)
    for _ in range(max_rounds):
        full = [0, *order, 0]
        # prefix sums of forward and reversed link costs along the current path
        fwd = np.zeros(n + 2)
        rev = np.zeros(n + 2)
        for k in range(1, n + 2):
            fwd[k] = fwd[k - 1] + cost[full[k - 1], full[k]]
            rev[k] = rev[k - 1] + cost[full[k], full[k - 1]]
        arr = np.asarray(full)
        xs, ys = arr[:-1], arr[1:]  # every link k: full[k] -> full[k+1]
        gaps = cost[xs, ys]
        best = (-1e-9, None)
        for i in range(1, n + 1):  # block = full[i..j]
            for j in range(i, min(n, i + max_block - 1) + 1):
                p, q = full[i - 1], full[j + 1]
                head, tail = full[i], full[j]
                removed = cost[p, head] + cost[tail, q]
                inner_f = fwd[j] - fwd[i]
                inner_r = rev[j] - rev[i]
                closed = cost[p, q]
                # reverse in place
                delta = (cost[p, tail] + cost[head, q] + inner_r) - (removed + inner_f)
                if j > i and delta < best[0]:
                    best = (delta, ("rev", i, j, None))
                # move the block between full[k] and full[k+1], k outside [i-1, j]
                base = closed - removed
                d_fwd = base + cost[xs, head] + cost[tail, ys] - gaps
                d_fwd[i - 1 : j + 1] = np.inf
                k = int(np.argmin(d_fwd))
                if d_fwd[k] < best[0]:
                    best = (float(d_fwd[k]), ("move", i, j, k, False))
                if j > i:
                    d_rev = base + cost[xs, tail] + cost[head, ys] - gaps + inner_r - inner_f
                    d_rev[i - 1 : j + 1] = np.inf
                    k = int(np.argmin(d_rev))
                    if d_rev[k] < best[0]:
                        best = (float(d_rev[k]), ("move", i, j, k, True))
        if best[1] is None:
            break
        move = best[1]
        block_idx = (move[1] - 1, move[2])  # slice into order
        block = order[block_idx[0]:block_idx[1]]
        if move[0] == "rev":
            order[block_idx[0]:block_idx[1]] = block[::-1]
            continue
        _, i, j, k, reverse = move
        rest = order[: i - 1] + order[j:]
        # k counted positions in full (0 = START); map to an index in rest
        insert_at = k if k < i - 1 else k - (j - i + 1)
        rest[insert_at:insert_at] = block[::-1] if reverse else block
        order = rest
    return order


def decode(n: int, probs: dict[tuple[int, int], float], max_block: int = 40) -> list[int]:
    """Best order of nodes 1..n (scan order = 1..n)."""
    if n == 0:
        return []
    cost = cost_matrix(n, probs)
    if n <= EXACT_MAX:
        return held_karp(cost)
    return local_search(cost, list(range(1, n + 1)), max_block=max_block)
