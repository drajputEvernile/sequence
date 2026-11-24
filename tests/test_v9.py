"""v9's evidence and solver switches, on made-up text: page markers, content clues, the chart's
direction in time, cheap reversed runs, swaps."""
from __future__ import annotations

import numpy as np


def _page(top: str, body: str, bottom: str) -> str:
    return f"{top}\nPatient: Test Person\n{body}\nsome more text here\nand a closing line\n{bottom}"


def test_markers_read_every_printed_form():
    from seqlib.ladder.markers import read_marker

    m = read_marker(_page("Clinic Report", "body", "Page 23 of 41"))
    assert (m.number, m.total, m.kind, m.band) == (23, 41, "of", "bottom")  # v1 stops at totals of 20
    m = read_marker(_page("Printed 03/04/2024 10:22  Pg. 7/9", "body", "footer"))
    assert (m.number, m.total, m.kind, m.band) == (7, 9, "of", "top")
    m = read_marker(_page("Clinic Report", "body", "Page 5"))
    assert (m.number, m.total, m.kind) == (5, None, "page")
    m = read_marker(_page("Clinic Report", "body", "- 12 -"))
    assert (m.number, m.kind) == (12, "bare")
    m = read_marker(_page("Clinic Report", "body", "3 of 4"))
    assert (m.number, m.total, m.kind) == (3, 4, "of")
    # the most explicit form wins; a number that is not alone on its line is not a page number
    assert read_marker(_page("Page 2 of 3", "body", "2")).kind == "of"
    assert read_marker(_page("Clinic Report", "body", "Hemoglobin 12")) is None
    assert read_marker(_page("Clinic Report", "body", "Page 4 of 3")) is None  # number past the total
    # the same document's pages share a style; another way of numbering does not
    a, b = read_marker(_page("x", "body", "Page 1 of 3")), read_marker(_page("y", "other", "Page 2 of 3"))
    c = read_marker(_page("x", "body", "Page 2"))
    assert a.style == b.style and a.style != c.style


def test_marker_anchors_successors_and_pair_features():
    from seqlib.ladder.markers import marker_anchors, marker_pair, marker_successors, page_markers

    texts = {10: _page("t", "b", "- 2 -"), 11: _page("t", "b", "- 1 -"), 12: _page("t", "b", "- 3 -"),
             20: _page("t", "b", "Page 1"), 21: _page("t", "b", "Page 1"), 22: _page("t", "b", "Page 2")}
    marks = page_markers(texts)
    assert marker_anchors(marks) == [(10, 12), (11, 10)]  # "Page 1" twice: no anchor into page 22
    assert set(marker_successors(marks)) == {(11, 10), (10, 12), (20, 22), (21, 22)}
    x = marker_pair(marks[11], marks[10])
    assert x["mk_follows"] == 1.0 and x["mk_step"] == 1.0 and x["mk_b_first"] == 0.0
    assert marker_pair(marks[10], marks[20])["mk_same_style"] == 0.0
    assert marker_pair(None, None) == {}


def test_content_clues():
    from seqlib.ladder.content import content_pair, item_successors, read_content

    a = read_content("Encounter Id: AB1234\nChief Complaint\ncough\nHistory of Present Illness\n1. fever\n2. chills\n"
                     "3. tired\nline\nline\nline")
    b = read_content("Encounter Id: AB1234\n(continued)\n4. night sweats\nAssessment\nPlan\nline\nline\nline\nline")
    c = read_content("Encounter Id: ZZ9999\nVisit Date: 03/14/2024\nReview of Systems\nline\nline\nline\nline\nline")
    assert (a.sec_first, a.sec_last, a.item_first, a.item_last, a.enc_id) == (1, 2, 1, 3, "ab1234")
    assert b.cont_from and b.sec_first == 7 and c.enc_date == "2024-03-14"
    x = content_pair(a, b)
    assert x["ct_item_next"] == 1.0 and x["ct_enc_same"] == 1.0 and x["ct_sec_delta"] == 5.0 and x["ct_b_cont_from"] == 1.0
    assert content_pair(b, c)["ct_enc_same"] == 0.0 and content_pair(b, c)["ct_sec_delta"] < 0
    assert item_successors({1: a, 2: b, 3: c}) == [(1, 2)]
    t = read_content("Lab Results\nx\ny\nz\nw\nWBC 5.2 K/uL 4.0-10.0\nRBC 4.6 M/uL 4.2-5.4\nHGB 13.9 g/dL 12-16\n"
                     "end\nend\nend\nend")
    assert t.table_tail


def test_date_trend_and_pattern():
    from seqlib.ladder.content import date_trend, pattern_pair

    dos = {1: "2024-01-01", 2: "2024-02-01", 3: "2024-03-01", 4: "2023-12-01", 5: "2024-04-01"}
    trend = date_trend(dos, [1, 2, 3, 4, 5])
    assert trend == (3 - 1) / 4
    assert pattern_pair("2024-01-01", "2024-02-01", trend)["pt_with_trend"] == trend
    assert pattern_pair("2024-02-01", "2024-01-01", trend)["pt_with_trend"] == -trend
    assert date_trend({}, [1, 2]) is None and pattern_pair("a", "b", None) == {}


def test_reversed_run_is_cheap_to_undo_with_reverse_share():
    from seqlib.v3.solve import decode_v3, v3_costs

    n = 15
    truth = [1, 2, 3, 4, *range(12, 4, -1), 13, 14, 15]  # pages 5..12 were scanned in reverse
    full = [0, *truth, 0]
    probs = {(a, b): 0.3 for a in range(n + 1) for b in range(n + 1) if a != b}
    probs.update({(a, b): 0.9 for a, b in zip(full, full[1:])})
    cost = v3_costs(n, probs, 1.5, reverse_share=0.25)
    assert np.isclose(cost[9, 8], -np.log(0.9) + 1.5 * 0.25) and np.isclose(cost[8, 10], -np.log(0.3) + 1.5)
    assert decode_v3(n, probs, 1.5, "search") == list(range(1, n + 1))  # every reversed link pays a margin
    for solver in ("search", "cover", "best"):
        assert decode_v3(n, probs, 1.5, solver, reverse_share=0.25) == truth


def test_swap_move():
    from seqlib.v2.decode import _best_swap, cost_matrix, local_search, path_cost

    n = 20
    truth = list(range(1, n + 1))
    truth[2], truth[14] = truth[14], truth[2]
    full = [0, *truth, 0]
    probs = {(a, b): 0.05 for a in range(n + 1) for b in range(n + 1) if a != b}
    probs.update({(a, b): 0.95 for a, b in zip(full, full[1:])})
    cost = cost_matrix(n, probs)
    scan_full = [0, *range(1, n + 1), 0]
    delta, i, j = _best_swap(cost, scan_full)
    assert (i, j) == (3, 15) and np.isclose(delta, path_cost(truth, cost) - path_cost(list(range(1, n + 1)), cost))
    assert local_search(cost, list(range(1, n + 1)), swaps=True) == truth
    assert path_cost(local_search(cost, list(range(1, n + 1)), swaps=True), cost) <= \
        path_cost(local_search(cost, list(range(1, n + 1))), cost) + 1e-9


def test_position_measure_for_tuning():
    from seqlib.v3.model import position_accuracy, successor_accuracy

    truth = list(range(1, 11))
    moved = [1, 2, 9, 10, 3, 4, 5, 6, 7, 8]  # one block in the wrong place: links mostly right, places not
    assert successor_accuracy(moved, truth) > 0.6 and position_accuracy(moved, truth) == 0.2
