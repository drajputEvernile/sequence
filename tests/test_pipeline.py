"""Readers, metrics, split, v1 runner and v2 end to end on synthetic charts."""
from __future__ import annotations

import shutil

import pytest

from seqlib.settings import cfg
from synthetic import make_dataset


@pytest.fixture(scope="module")
def dataset():
    shutil.rmtree(cfg.CHARTS_ROOT.parent, ignore_errors=True)
    return make_dataset(cfg.CHARTS_ROOT, cfg.GROUND_TRUTH_PATH, n_charts=24)


def test_ground_truth_reader_and_j_series(dataset):
    from seqlib.io.ground_truth import load_ground_truth

    truths = load_ground_truth(cfg.GROUND_TRUTH_PATH)
    assert set(truths) == set(dataset)
    for name, order in dataset.items():
        assert truths[name].main_order == order
        assert len(truths[name].junk_pages) == 1
        assert not truths[name].problems()


def test_ground_truth_problems_are_reported():
    from seqlib.io.ground_truth import ChartTruth

    assert ChartTruth("x", {1: "1", 2: "1", 3: "J1"}).problems()
    assert ChartTruth("x", {1: "2", 2: "1"}).problems() == []


def test_chart_loader_reads_ocr_and_csvs(dataset):
    from seqlib.io.chart import load_chart

    chart = load_chart("SYN000")
    assert [p.page_number for p in chart.pages] == list(range(1, len(chart.pages) + 1))
    assert sum(p.group == "junk" for p in chart.pages) == 1
    assert all(p.ocr_source == "final2" for p in chart.pages)
    assert "dos" in chart.missing  # optional input absent: reported, not fatal


def test_metrics():
    from seqlib.eval.metrics import score_chart, summarize
    from seqlib.io.ground_truth import ChartTruth

    truth = ChartTruth("c", {1: "1", 2: "2", 3: "3", 4: "4", 5: "J1"})
    perfect = score_chart(truth, [1, 2, 3, 4], [5])
    assert perfect.perfect and perfect.successor_correct == 3 and perfect.junk_tp == 1
    # one page moved from the front to the back: every strict number is wrong, most links hold
    moved = score_chart(truth, [2, 3, 4, 1], [5])
    assert moved.strict_correct == 0 and moved.successor_correct == 2 and moved.in_place == 3
    # a Main page set aside goes back at its scan position for the order measures
    dropped = score_chart(truth, [1, 2, 4], [3, 5])
    assert dropped.position_correct == 4 and dropped.strict_correct == 2 and dropped.junk_fp == 1
    summary = summarize([perfect, moved])
    assert summary["perfect_charts"] == 0.5


def test_split_is_by_chart_and_stratified(dataset):
    from seqlib.eval.split import make_split
    from seqlib.io.ground_truth import load_ground_truth

    split = make_split(load_ground_truth(cfg.GROUND_TRUTH_PATH), 0.25, 13)
    assert not set(split["train"]) & set(split["test"])
    assert len(split["train"]) + len(split["test"]) == len(dataset)
    assert any(k.startswith("moved") for k in split["strata"])


def test_v1_runs_from_local_files_and_uses_page_numbers(dataset):
    from seqlib.eval.metrics import score_chart, summarize
    from seqlib.io.chart import load_chart
    from seqlib.io.ground_truth import load_ground_truth
    from seqlib.v1.runner import run_scan_order, run_v1

    truths = load_ground_truth(cfg.GROUND_TRUTH_PATH)
    v1, scan = [], []
    for name in dataset:
        chart = load_chart(name)
        r = run_v1(chart)
        assert sorted(r.main_order) == sorted(truths[name].main_order)
        v1.append(score_chart(truths[name], r.main_order, r.junk_pages))
        s = run_scan_order(chart)
        scan.append(score_chart(truths[name], s.main_order, s.junk_pages))
    # printed "Page i of k" lets v1 repair some damage scan order leaves
    assert summarize(v1)["position_accuracy"] >= summarize(scan)["position_accuracy"]


def test_decoder_exact_and_local_search_agree_on_a_clear_chain():
    from seqlib.v2.decode import cost_matrix, held_karp, local_search

    n = 8
    truth = [3, 1, 4, 2, 8, 6, 5, 7]
    probs = {(a, b): 0.01 for a in range(n + 1) for b in range(n + 1) if a != b}
    full = [0, *truth, 0]
    probs.update({(a, b): 0.95 for a, b in zip(full, full[1:])})
    cost = cost_matrix(n, probs)
    assert held_karp(cost) == truth
    assert local_search(cost, list(range(1, n + 1))) == truth


def test_v2_trains_and_beats_scan_order(dataset):
    from seqlib.eval.metrics import score_chart, summarize
    from seqlib.eval.split import make_split
    from seqlib.io.chart import load_chart
    from seqlib.io.ground_truth import load_ground_truth
    from seqlib.v2.model import PairScorer, train
    from seqlib.v2.runner import run_v2

    truths = load_ground_truth(cfg.GROUND_TRUTH_PATH)
    split = make_split(truths, 0.25, 13)
    scorer = train((load_chart(n), truths[n]) for n in split["train"])
    assert scorer.report["candidate_recall"] > 0.9
    scorer.save(cfg.V2_MODEL_PATH)
    scorer = PairScorer.load(cfg.V2_MODEL_PATH)
    v2, scan = [], []
    for name in split["test"]:
        chart = load_chart(name)
        r = run_v2(chart, scorer)
        v2.append(score_chart(truths[name], r.main_order, r.junk_pages))
        scan.append(score_chart(truths[name], sorted(truths[name].main_order), []))
    assert summarize(v2)["successor_accuracy"] >= summarize(scan)["successor_accuracy"]


@pytest.mark.skipif(not cfg.TESSERACT_CMD.is_file(), reason="Tesseract not installed")
def test_local_ocr_writes_a_readable_chart(tmp_path):
    from PIL import Image, ImageDraw, ImageFont

    from seqlib.io.chart import load_chart
    from seqlib.local_ocr.tesseract_ocr import ocr_chart

    photos = []
    for i, words in enumerate(["Progress Note Page 1 of 2", "Assessment and Plan Page 2 of 2"], start=1):
        image = Image.new("RGB", (1700, 2200), "white")
        draw = ImageDraw.Draw(image)
        try:
            font = ImageFont.truetype("arial.ttf", 60)
        except OSError:
            font = ImageFont.load_default()
        draw.text((150, 200), words, fill="black", font=font)
        path = tmp_path / f"IMG_{i}.jpg"
        image.save(path)
        photos.append(path)
    out = ocr_chart(photos, "photo_case", tmp_path / "charts", cfg.TESSERACT_CMD)
    chart = load_chart(out)
    assert len(chart.pages) == 2
    assert "page 1 of 2" in chart.pages[0].text.lower()
    assert chart.pages[0].final2["pagesMeta"][0]["lines"]
