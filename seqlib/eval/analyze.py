"""What the ground truth says about the problem, before any method is scored.

The answers decide what v2 has to be good at:

* How many charts are already in scan order? (If most are, the job is to find the few pages
  that moved, not to rebuild every order.)
* When pages move, how? One page out of place, a block moved, a block reversed (a stack scanned
  upside down), or documents shuffled?
* How far do they move? (v1 never moves a page more than 5 places.)
* Do the moved pages print "page N of M"? (The share v1's marker rule can ever fix.)
"""
from __future__ import annotations

from collections import Counter
from typing import Optional

from seqlib.io.chart import Chart
from seqlib.io.ground_truth import ChartTruth


def runs(order: list[int]) -> list[list[int]]:
    """Maximal runs of the true order that are consecutive in scan order (+1 or -1 steps)."""
    scan = sorted(order)
    rank = {p: i for i, p in enumerate(scan)}
    out: list[list[int]] = []
    for page in order:
        if out and abs(rank[page] - rank[out[-1][-1]]) == 1 and (
            len(out[-1]) == 1 or (rank[page] - rank[out[-1][-1]]) == (rank[out[-1][-1]] - rank[out[-1][-2]])
        ):
            out[-1].append(page)
        else:
            out.append([page])
    return out


def describe(truth: ChartTruth, chart: Optional[Chart] = None) -> dict:
    order = truth.main_order
    scan = sorted(order)
    rank = {p: i for i, p in enumerate(scan)}
    moves = [abs(i - rank[p]) for i, p in enumerate(order)]
    pieces = runs(order)
    reversed_pieces = [r for r in pieces if len(r) > 1 and rank[r[1]] < rank[r[0]]]
    out = {
        "chart": truth.chart_name,
        "main_pages": len(order),
        "junk_pages": len(truth.junk_pages),
        "unlabelled_pages": len(truth.unlabelled),
        "scan_order": order == scan,
        "pages_out_of_place": sum(1 for m in moves if m),
        "max_move": max(moves) if moves else 0,
        "runs": len(pieces),  # 1 = scan order; each extra run is a cut in the scan order
        "reversed_runs": len(reversed_pieces),
        "fully_reversed": len(order) > 1 and order == scan[::-1],
        # true neighbour pairs that are also neighbours in scan order (scan order's successor hits)
        "scan_successor_hits": sum(1 for a, b in zip(order, order[1:]) if rank[b] == rank[a] + 1),
        "problems": truth.problems(),
    }
    if chart is not None:
        pages = chart.by_number()
        moved = [p for p, m in zip(order, moves) if m]
        def printed(p: int) -> bool:
            page = pages.get(p)
            return bool(page and str(page.get("page_details", "printed_page_total", "") or "").strip())
        out["moved_with_printed_total"] = sum(1 for p in moved if printed(p))
        out["pages_with_printed_total"] = sum(1 for p in order if printed(p))
        out["junk_disagree"] = sorted(
            p for p in pages if (pages[p].group != "main") != (p in set(truth.junk_pages))
        )[:20]
    return out


def overview(rows: list[dict]) -> dict:
    n = len(rows)
    if not n:
        return {}
    pages = sum(r["main_pages"] for r in rows)
    moved_pages = sum(r["pages_out_of_place"] for r in rows)
    dist = Counter(min(r["max_move"], 50) // 5 * 5 for r in rows if r["max_move"])
    out = {
        "charts": n,
        "main_pages": pages,
        "charts_in_scan_order": sum(r["scan_order"] for r in rows),
        "charts_fully_reversed": sum(r["fully_reversed"] for r in rows),
        "charts_with_reversed_run": sum(1 for r in rows if r["reversed_runs"]),
        "pages_out_of_place": moved_pages,
        "share_pages_out_of_place": round(moved_pages / pages, 4) if pages else None,
        "scan_order_successor_accuracy": None,
        "max_move_histogram (bucket of 5 -> charts)": dict(sorted(dist.items())),
        "median_runs_when_moved": _median([r["runs"] for r in rows if not r["scan_order"]]),
    }
    pairs = sum(max(0, r["main_pages"] - 1) for r in rows)
    hits = sum(r["scan_successor_hits"] for r in rows)
    out["scan_order_successor_accuracy"] = round(hits / pairs, 4) if pairs else None
    if any("moved_with_printed_total" in r for r in rows):
        out["moved_pages_with_printed_page_n_of_m"] = sum(r.get("moved_with_printed_total", 0) for r in rows)
        out["main_pages_with_printed_page_n_of_m"] = sum(r.get("pages_with_printed_total", 0) for r in rows)
    return out


def _median(values: list[int]) -> Optional[float]:
    if not values:
        return None
    values = sorted(values)
    mid = len(values) // 2
    return float(values[mid]) if len(values) % 2 else (values[mid - 1] + values[mid]) / 2
