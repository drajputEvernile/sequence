"""Extra pair evidence for v4 to v6, and the anchors (forced links) of v4+.

Feature groups (columns added to v3's evidence):

* ``banner`` -- same fax / print transmission; its counter runs on (n -> n+1); the step; a is its
  transmission's last page; b its first.
* ``dates``  -- both pages dated from their own text; same date; days from a to b (signed);
  going back in time; either dated. The model learns from the truth whether reading order
  follows dates -- nothing assumes it does.
* ``hash``   -- header image similarity (1 - Hamming distance / 64).
* ``seam``   -- the seam cross-encoder's probability that b's opening continues a's ending.
* ``v1``     -- v1's rules (printed page-number families) put b right after a *against* the scan
               order (v8). Only departures from scan order count, so the feature says nothing
               about where a page was scanned.

Anchors (design doc 7.4.1): a -> b is forced when a prints "page n of m", b prints "page n+1 of m"
(same total) and no other page claims either number; likewise for a fax / print banner counter
in one transmission, unless the pages' printed numbers say otherwise. A page with two candidate
anchors in either direction gets none (conflict -> left to the evidence).
"""
from __future__ import annotations

import math
from collections import defaultdict
from datetime import date
from typing import Optional, Sequence

import numpy as np

from .facts import ChartFacts, hash_similarity

START = -1
GROUP_FEATURES: dict[str, tuple[str, ...]] = {
    "banner": ("bn_same_tx", "bn_follows", "bn_step", "bn_a_last", "bn_b_first"),
    "dates": ("dt_both", "dt_same", "dt_days", "dt_backwards", "dt_a_dated", "dt_b_dated"),
    "hash": ("hash_sim",),
    "seam": ("seam_p",),
    "v1": ("v1_jump",),
}
GROUP_ORDER = ("banner", "dates", "hash", "seam", "v1")


def feature_names(groups: Sequence[str]) -> tuple[str, ...]:
    return tuple(name for g in GROUP_ORDER if g in groups for name in GROUP_FEATURES[g])


def _days(a: str, b: str) -> Optional[float]:
    try:
        return float((date.fromisoformat(b) - date.fromisoformat(a)).days)
    except (TypeError, ValueError):
        return None


def extra_matrix(pairs: Sequence[tuple[int, int]], facts: ChartFacts, groups: Sequence[str],
                 seam: Optional[dict[tuple[int, int], float]] = None,
                 v1: Optional[set[tuple[int, int]]] = None) -> np.ndarray:
    names = feature_names(groups)
    col = {n: i for i, n in enumerate(names)}
    X = np.full((len(pairs), len(names)), np.nan, dtype=np.float32)
    for r, (a, b) in enumerate(pairs):
        if "v1" in groups:
            X[r, col["v1_jump"]] = (a, b) in (v1 or set())
        if a == START or b == START:
            continue
        x = X[r]
        if "banner" in groups:
            ba, bb = facts.banner.get(a), facts.banner.get(b)
            if ba is not None:
                x[col["bn_a_last"]] = ba.total is not None and ba.counter == ba.total
            if bb is not None:
                x[col["bn_b_first"]] = bb.counter == 1
            if ba is not None and bb is not None:
                same = ba.identity == bb.identity
                x[col["bn_same_tx"]] = same
                if same:
                    step = bb.counter - ba.counter
                    x[col["bn_step"]] = max(-20, min(20, step))
                    x[col["bn_follows"]] = step == 1
        if "dates" in groups:
            da, db = facts.dos.get(a), facts.dos.get(b)
            x[col["dt_a_dated"]] = bool(da)
            x[col["dt_b_dated"]] = bool(db)
            if da and db:
                x[col["dt_both"]] = 1
                x[col["dt_same"]] = da == db
                days = _days(da, db)
                if days is not None:
                    x[col["dt_days"]] = max(-3650, min(3650, days))
                    x[col["dt_backwards"]] = days < 0
            else:
                x[col["dt_both"]] = 0
        if "hash" in groups:
            sim = hash_similarity(facts.header_hash.get(a), facts.header_hash.get(b))
            if sim is not None:
                x[col["hash_sim"]] = sim
        if "seam" in groups and seam is not None:
            p = seam.get((a, b))
            if p is not None:
                x[col["seam_p"]] = p
    return X


def find_anchors(printed: dict[int, tuple[int, int]], facts: Optional[ChartFacts],
                 use_banner: bool = True) -> list[tuple[int, int]]:
    """Forced links (page numbers), from unambiguous "n of m" -> "n+1 of m" (and banner) runs."""
    proposals: list[tuple[int, int]] = []
    holders: dict[tuple[int, int], list[int]] = defaultdict(list)
    for page, (n, m) in printed.items():
        holders[(m, n)].append(page)
    for (m, n), pages in holders.items():
        nxt = holders.get((m, n + 1), [])
        if len(pages) == 1 and len(nxt) == 1:
            proposals.append((pages[0], nxt[0]))
    if use_banner and facts is not None and facts.banner:
        tx: dict[tuple[str, int], list[int]] = defaultdict(list)
        for page, b in facts.banner.items():
            tx[(b.identity, b.counter)].append(page)
        for (ident, counter), pages in tx.items():
            nxt = tx.get((ident, counter + 1), [])
            if len(pages) == 1 and len(nxt) == 1:
                a, b = pages[0], nxt[0]
                pa, pb = printed.get(a), printed.get(b)
                if pa and pb and pa[1] == pb[1] and pb[0] != pa[0] + 1:
                    continue  # the document's own page numbers say otherwise
                proposals.append((a, b))
    # a page with two different anchored successors / predecessors gets none
    outs: dict[int, set[int]] = defaultdict(set)
    ins: dict[int, set[int]] = defaultdict(set)
    for a, b in proposals:
        outs[a].add(b)
        ins[b].add(a)
    anchors = sorted({(a, b) for a, b in proposals if len(outs[a]) == 1 and len(ins[b]) == 1})
    # never a loop
    succ = dict(anchors)
    clean = []
    for a, b in anchors:
        k, seen = b, {a}
        while k in succ and k not in seen:
            seen.add(k)
            k = succ[k]
        if k != a:
            clean.append((a, b))
    return clean


def anchor_order(n: int, anchors: Sequence[tuple[int, int]]) -> list[int]:
    """Nodes 1..n in scan order with every anchored chain kept together, placed where its
    earliest-scanned page was: a better start for the search than the bare scan order."""
    succ = dict(anchors)
    has_pred = {b for _, b in anchors}
    chains: list[list[int]] = []
    for k in range(1, n + 1):
        if k in has_pred:
            continue
        chain, x = [k], k
        while x in succ:
            x = succ[x]
            chain.append(x)
        chains.append(chain)
    placed = {x for c in chains for x in c}
    chains += [[k] for k in range(1, n + 1) if k not in placed]  # anchor loops broken earlier: safety
    chains.sort(key=min)
    return [x for c in chains for x in c]


def sigmoid(x: float) -> float:
    return 1.0 / (1.0 + math.exp(-x))


def v1_jumps(chart, pages: Sequence[int]) -> list[tuple[int, int]]:
    """The links v1's engines make against the scan order of ``pages`` (page numbers, START = -1
    for the first / last page); [] when v1 keeps scan order or cannot run."""
    from seqlib.v1.runner import run_v1

    keep = set(pages)
    saved = {p.page_number: p.group_override for p in chart.pages}
    try:
        for page in chart.pages:
            page.group_override = "main" if page.page_number in keep else "junk"
        order = [p for p in run_v1(chart).main_order if p in keep]
    except Exception:  # noqa: BLE001 - v1 is one piece of evidence among many
        return []
    finally:
        for page in chart.pages:
            page.group_override = saved.get(page.page_number)
    scan = sorted(keep)
    if sorted(order) != scan:
        return []
    nxt = dict(zip(scan, scan[1:]))
    jumps = [(a, b) for a, b in zip(order, order[1:]) if nxt.get(a) != b]
    if order and order[0] != scan[0]:
        jumps.append((START, order[0]))
    if order and order[-1] != scan[-1]:
        jumps.append((order[-1], START))
    return jumps
