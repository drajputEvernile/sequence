"""How right an order is, for one chart and over many.

Two families of numbers, because they answer different questions:

**Strict (what review-ui shows today).** Each page whose ground truth is a number (Main) scores
a match when the predicted ``actual_sequence`` is the same number. A page the method set aside
as blank / junk has no number and scores a miss here (review-ui shows it as "unknown" and leaves
it out; ``strict_scored_only`` is that variant). A single misplaced page early in a chart shifts
every number after it, so this is the harshest measure and it also mixes in blank / junk errors.

**Order (junk taken from the ground truth).** The prediction's order is restricted to the
ground truth's Main pages (a Main page the method set aside is put back at its scan position),
so only the ordering is measured:

* ``position``   -- share of pages at exactly the right place (renumbered among true Main pages);
* ``successor``  -- share of true neighbour pairs (A then B) that are neighbours in the prediction:
                   robust to a block being shifted, the usual measure for page ordering;
* ``in_place``   -- longest common subsequence / pages: share of pages that need not move
                   (1 - fewest pages to pick up and re-insert);
* ``kendall``    -- Kendall's tau between the two orders (1 identical, -1 reversed);
* ``perfect``    -- the whole chart's order is right.

``junk_precision`` / ``junk_recall`` score the set-aside pages against the J series.
"""
from __future__ import annotations

from bisect import bisect_left
from dataclasses import dataclass, field
from typing import Iterable, Optional

from seqlib.io.ground_truth import ChartTruth


@dataclass
class ChartScore:
    chart_name: str
    pages: int  # true Main pages
    strict_correct: int = 0
    strict_total: int = 0
    strict_scored: int = 0  # pages review-ui would score (prediction has a number)
    strict_scored_correct: int = 0
    position_correct: int = 0
    successor_correct: int = 0
    successor_total: int = 0
    in_place: int = 0
    kendall: Optional[float] = None
    perfect: bool = False
    junk_tp: int = 0
    junk_fp: int = 0
    junk_fn: int = 0
    scan_order_correct: bool = False  # the truth is plain scan order
    errors: list[str] = field(default_factory=list)  # first few wrong pages, for the error dump


def _lcs_length(a: list[int], b: list[int]) -> int:
    """LCS of two permutations of the same items = LIS of b's positions in a (n log n)."""
    where = {x: i for i, x in enumerate(a)}
    seq = [where[x] for x in b if x in where]
    tails: list[int] = []
    for v in seq:
        k = bisect_left(tails, v)
        if k == len(tails):
            tails.append(v)
        else:
            tails[k] = v
    return len(tails)


def _kendall(a: list[int], b: list[int]) -> Optional[float]:
    n = len(a)
    if n < 2:
        return None
    rank = {x: i for i, x in enumerate(a)}
    seq = [rank[x] for x in b]
    # inversions by merge sort
    def count(arr: list[int]) -> tuple[list[int], int]:
        if len(arr) <= 1:
            return arr, 0
        mid = len(arr) // 2
        left, x = count(arr[:mid])
        right, y = count(arr[mid:])
        merged, inv, i, j = [], x + y, 0, 0
        while i < len(left) and j < len(right):
            if left[i] <= right[j]:
                merged.append(left[i]); i += 1
            else:
                merged.append(right[j]); j += 1; inv += len(left) - i
        merged += left[i:] + right[j:]
        return merged, inv
    _, inversions = count(seq)
    pairs = n * (n - 1) / 2
    return 1.0 - 2.0 * inversions / pairs


def restrict(pred_main: list[int], truth_main: list[int]) -> list[int]:
    """The prediction's order over the true Main pages; a true Main page the prediction set
    aside goes back in at its scan position (before the first predicted page scanned after it)."""
    truth = set(truth_main)
    order = [p for p in pred_main if p in truth]
    for page in sorted(truth - set(order)):
        k = next((i for i, q in enumerate(order) if q > page), len(order))
        order.insert(k, page)
    return order


def score_chart(truth: ChartTruth, pred_main: list[int], pred_junk: Iterable[int]) -> ChartScore:
    t_main = truth.main_order
    t_junk = set(truth.junk_pages)
    pred_junk = set(pred_junk)
    s = ChartScore(truth.chart_name, len(t_main))
    s.scan_order_correct = t_main == sorted(t_main)

    # strict
    pred_label = {p: i for i, p in enumerate(pred_main, start=1)}
    for i, page in enumerate(t_main, start=1):
        s.strict_total += 1
        got = pred_label.get(page)
        if got is not None:
            s.strict_scored += 1
        if got == i:
            s.strict_correct += 1
            if got is not None:
                s.strict_scored_correct += 1

    # order, junk from the truth
    order = restrict(pred_main, t_main)
    s.position_correct = sum(1 for a, b in zip(order, t_main) if a == b)
    nxt = {a: b for a, b in zip(order, order[1:])}
    pairs = list(zip(t_main, t_main[1:]))
    s.successor_total = len(pairs)
    s.successor_correct = sum(1 for a, b in pairs if nxt.get(a) == b)
    s.in_place = _lcs_length(t_main, order)
    s.kendall = _kendall(t_main, order)
    s.perfect = order == t_main
    for i, (a, b) in enumerate(zip(order, t_main)):
        if a != b and len(s.errors) < 8:
            s.errors.append(f"pos {i + 1}: got page {a}, want page {b}")

    # blank / junk
    s.junk_tp = len(pred_junk & t_junk)
    s.junk_fp = len(pred_junk - t_junk)
    s.junk_fn = len(t_junk - pred_junk)
    return s


def _ratio(n: float, d: float) -> Optional[float]:
    return round(n / d, 4) if d else None


def summarize(scores: list[ChartScore]) -> dict[str, Optional[float]]:
    """Micro (pooled over pages) and macro (mean over charts) of every measure."""
    scored = [s for s in scores if s.pages]
    if not scored:
        return {"charts": 0}
    pages = sum(s.pages for s in scored)
    out: dict[str, Optional[float]] = {
        "charts": len(scored),
        "pages": pages,
        "charts_already_in_scan_order": _ratio(sum(s.scan_order_correct for s in scored), len(scored)),
        "strict_page_accuracy": _ratio(sum(s.strict_correct for s in scored), sum(s.strict_total for s in scored)),
        "strict_scored_only": _ratio(sum(s.strict_scored_correct for s in scored), sum(s.strict_scored for s in scored)),
        "position_accuracy": _ratio(sum(s.position_correct for s in scored), pages),
        "successor_accuracy": _ratio(sum(s.successor_correct for s in scored), sum(s.successor_total for s in scored)),
        "in_place": _ratio(sum(s.in_place for s in scored), pages),
        "perfect_charts": _ratio(sum(s.perfect for s in scored), len(scored)),
        "macro_position_accuracy": round(sum(s.position_correct / s.pages for s in scored) / len(scored), 4),
        "macro_successor_accuracy": round(
            sum((s.successor_correct / s.successor_total) if s.successor_total else 1.0 for s in scored) / len(scored), 4),
    }
    taus = [s.kendall for s in scored if s.kendall is not None]
    out["mean_kendall_tau"] = round(sum(taus) / len(taus), 4) if taus else None
    tp, fp, fn = (sum(getattr(s, k) for s in scored) for k in ("junk_tp", "junk_fp", "junk_fn"))
    out["junk_precision"] = _ratio(tp, tp + fp)
    out["junk_recall"] = _ratio(tp, tp + fn)
    return out
