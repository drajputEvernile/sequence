"""The engine's evidence and versions (v5, v8, v9): facts, markers, content clues, anchors, the solver,
training, the seam model, the gate."""
from __future__ import annotations

import shutil

import pytest

from seqlib.settings import cfg
from synthetic import make_raw_dataset

ROOT = cfg.CHARTS_ROOT.parent / "ladder_data"
GT = ROOT / "ground_truth.csv"


@pytest.fixture(scope="module")
def data():
    shutil.rmtree(ROOT, ignore_errors=True)
    from seqlib.eval.inventory import source_files
    from seqlib.extract.import_ocr import import_document

    raw, ocr = ROOT / "raw", ROOT / "ocr"
    truths = make_raw_dataset(raw, ocr, ROOT / "pipeline.csv", GT, n_docs=14, extra_docs=0)
    images, files = source_files(raw, ocr, cfg.DOC_KEY_REGEX)
    for key in truths:
        shutil.rmtree(cfg.CHARTS_ROOT / key, ignore_errors=True)
        import_document(key, images[key], files[key])
    from seqlib.io.ground_truth import load_ground_truth

    return load_ground_truth(GT, sequence_column="page_sequence")


def test_banner_parser():
    from seqlib.ladder.facts import parse_banner

    b = parse_banner(["11/03/2025 14:22 FROM: 555-123-4567 TO: 555-987-6543 P.004/012"])
    assert (b.counter, b.total) == (4, 12)
    c = parse_banner(["11/03/2025 14:22 FROM: 555-123-4567 TO: 555-987-6543 P.005/012"])
    assert c.identity == b.identity  # one transmission
    assert parse_banner(["Nov 3 2025 2:22PM  HP Fax  Page 4 of 12"]).counter == 4
    assert parse_banner(["Patient seen on 03/14/2024 for follow up"]) is None  # a date is not a counter


def test_header_hash(tmp_path):
    from PIL import Image, ImageDraw

    from seqlib.ladder.facts import hash_similarity, header_dhash

    def page(text, path):
        image = Image.new("L", (850, 1100), 255)
        ImageDraw.Draw(image).rectangle((40, 20, 40 + 30 * len(text), 90), fill=0)
        image.save(path)
        return header_dhash(path)

    a, b, c = page("letterhead", tmp_path / "a.png"), page("letterhead", tmp_path / "b.png"), page("x", tmp_path / "c.png")
    assert hash_similarity(a, b) == 1.0 and hash_similarity(a, c) < 1.0
    Image.new("L", (850, 1100), 255).save(tmp_path / "blank.png")
    assert header_dhash(tmp_path / "blank.png") is None


def test_page_dates_from_text_are_cached(data):
    from seqlib.io.chart import load_chart
    from seqlib.ladder.facts import CACHE_FILE, chart_facts

    chart = load_chart(sorted(data)[0])
    facts = chart_facts(chart, ["dates", "banner", "hash"])
    assert (chart.root / CACHE_FILE).is_file()
    assert facts.dos and all(v[:2] == "20" for v in facts.dos.values())  # synthetic banners carry a "dos" date
    assert facts.header_hash == {}  # the synthetic page images are blank: no header fingerprint
    again = chart_facts(chart, ["dates"])
    assert again.dos == facts.dos


def test_anchors():
    from seqlib.ladder.extra import anchor_order, find_anchors

    printed = {10: (2, 3), 11: (1, 3), 12: (3, 3), 13: (1, 2), 14: (1, 2)}
    assert find_anchors(printed, None) == [(10, 12), (11, 10)]  # "1 of 2" twice: no anchor for them
    assert anchor_order(5, [(2, 3), (4, 1)]) == [4, 1, 2, 3, 5]


def test_solver_restores_a_reversed_numbered_document():
    from seqlib.v3.solve import decode_v3

    n = 14
    probs = {(a, b): 0.3 for a in range(n + 1) for b in range(n + 1) if a != b}
    truth = [*range(8, 0, -1), *range(9, n + 1)]  # pages 8..1 are "page 1..8 of 8"
    anchors = [(truth[i], truth[i + 1]) for i in range(7)]
    for solver in ("search", "cover"):
        assert decode_v3(n, probs, 1.0, solver, anchors=anchors)[:8] == truth[:8]


def test_v5_trains_and_runs(data):
    from seqlib.io.chart import load_chart
    from seqlib.v3.model import V3Model, train
    from seqlib.v3.runner import run_v3
    from seqlib.versions import model_path, settings_for

    version = "v5"

    keys = sorted(data)
    model = train(((load_chart(k), data[k]) for k in keys[:10]), settings_for(version, margins=(0.0, 2.0)))
    assert model.report["version"] == version and model.report["anchors_found"] > 0
    assert model.report["anchors_correct"] == 1.0  # synthetic "Page i of k" never lies
    path = model_path(version)
    model.save(path)
    model = V3Model.load(path)
    for key in keys[10:]:
        r = run_v3(load_chart(key), model, pages=data[key].main_order)
        assert sorted(r.main_order) == sorted(data[key].main_order) and r.method.startswith(version)


def _tiny_base(path):
    """A tiny random BERT and a word-level tokenizer, built locally (no download)."""
    from transformers import BertConfig, BertModel, BertTokenizerFast

    words = sorted({w for t in ("cardiac rhythm atrial fibrillation diabetes insulin glucose kidney asthma "
                                "hypertension depression knee copd continued progress note page of patient test "
                                "person visit dos clinic main street fax cover sheet records confidential").split()
                    for w in [t]})
    path.mkdir(parents=True, exist_ok=True)
    (path / "vocab.txt").write_text("\n".join(["[PAD]", "[UNK]", "[CLS]", "[SEP]", "[MASK]", *words]), encoding="utf-8")
    BertTokenizerFast(vocab_file=str(path / "vocab.txt"), do_lower_case=True).save_pretrained(str(path))
    BertModel(BertConfig(vocab_size=5 + len(words), hidden_size=32, num_hidden_layers=1, num_attention_heads=2,
                         intermediate_size=64, max_position_embeddings=256)).save_pretrained(str(path))


def test_seam_model(data):
    pytest.importorskip("torch")
    pytest.importorskip("transformers")
    from seqlib.io.chart import load_chart
    from seqlib.ladder import load_seam
    from seqlib.ladder.seam import seam_half, train_seam
    from seqlib.v3.model import train
    from seqlib.v3.runner import run_v3
    from seqlib.versions import settings_for

    shutil.rmtree(cfg.SEAM_MODEL_PATH, ignore_errors=True)
    _tiny_base(cfg.SEAM_BASE_MODEL_PATH)
    keys = sorted(data)
    half = sorted(seam_half(keys[:10])) or keys[:5]
    report = train_seam(cfg.SEAM_BASE_MODEL_PATH, cfg.SEAM_MODEL_PATH, ((load_chart(k), data[k]) for k in half),
                        epochs=1, batch=16, max_length=64)
    assert report["pairs"] > 0
    seam = load_seam(required=True)
    assert set(seam.meta["trained_on"]) == set(half)
    model = train(((load_chart(k), data[k]) for k in keys[:10]), settings_for("v5", groups=("banner", "dates", "hash", "seam"), margins=(0.0, 2.0)),
                  seam=seam, fit_exclude=set(half))
    assert model.report["fitted_on_documents"] == 10 - len(half)
    r = run_v3(load_chart(keys[-1]), model, pages=data[keys[-1]].main_order, seam=seam)
    assert sorted(r.main_order) == sorted(data[keys[-1]].main_order)


def test_margin_rule_when_no_margin_meets_the_floor(data):
    from seqlib.io.chart import load_chart
    from seqlib.v3.model import V3Settings, train

    keys = sorted(data)
    model = train(((load_chart(k), data[k]) for k in keys[:10]),
                  V3Settings(margins=(0.0, 2.0, 6.0), harm_floor=1.01))  # unreachable floor
    curve = model.report["margin_curve"]
    key = lambda p: (p["held_out_in_order_successor"] or 0.0, p["held_out_shuffled_successor"] or 0.0)  # noqa: E731
    chosen = next(p for p in curve if p["margin"] == model.margin)
    assert key(chosen) == max(key(p) for p in curve)  # the least harm, then the most repair
    assert model.report["tuned_on_documents"] == 10  # every document, out of fold


def test_gate_policy_and_v8(data):
    import numpy as np

    from seqlib.io.chart import load_chart
    from seqlib.v3.tuning import choose_policy
    from seqlib.v7.model import GATE_FEATURES, V7Model, train_v7
    from seqlib.v7.runner import run_v7
    from seqlib.versions import model_path, settings_for

    # the policy: when the high margin is best everywhere, nothing gains from the gate
    succ = np.array([[0.5, 0.9], [0.5, 0.9], [1.0, 1.0], [0.9, 1.0]])  # margins (0, 2): docs x margins
    p = choose_policy(succ, np.array([0.9, 0.8, 0.1, 0.2]), np.array([False, False, True, True]), (0.0, 2.0),
                      harm_floor=1.0)
    assert p["meets_floor"] and (p["repair"], p["in_order"]) == (0.9, 1.0)
    # ... when the low margin repairs and only harms in-order documents, the gate keeps it off them
    p = choose_policy(np.array([[1.0, 0.5], [0.9, 1.0]]), np.array([0.9, 0.1]), np.array([False, True]),
                      (0.0, 2.0), harm_floor=1.0)
    assert (p["margin_low"], p["margin_high"], p["repair"], p["in_order"]) == (0.0, 2.0, 1.0, 1.0)

    keys = sorted(data)
    settings = settings_for("v8")
    assert "v1" in settings.groups and settings.solver == "best" and "v1_jump" in settings.feature_names
    assert (settings.reverse_share, settings.swaps, settings.objective) == (1.0, False, "successor")
    model = train_v7(((load_chart(k), data[k]) for k in keys[:10]), settings, folds=3, margins=(0.0, 2.0))
    assert model.report["tuned_on_documents"] == 10 and len(model.report["margin_curve_out_of_fold"]) == 2
    assert model.report["candidate_recall"] > 0.9 and model.report["anchors_correct"] == 1.0
    path = model_path("v8")
    model.save(path)
    model = V7Model.load(path)
    for key in keys[10:]:
        r = run_v7(load_chart(key), model, pages=data[key].main_order)
        assert sorted(r.main_order) == sorted(data[key].main_order) and r.method.startswith("v8:gate")
    assert len(GATE_FEATURES) == 15


def test_v1_jumps_are_v1_departures_from_scan(data):
    from seqlib.io.chart import load_chart
    from seqlib.ladder.extra import START, extra_matrix, v1_jumps
    from seqlib.ladder.facts import ChartFacts

    key = sorted(data)[0]
    chart = load_chart(key)
    pages = sorted(data[key].main_order)
    jumps = v1_jumps(chart, pages)
    scan_next = dict(zip(pages, pages[1:]))
    assert all(scan_next.get(a) != b for a, b in jumps if a != START)
    assert all(p.group_override is None for p in chart.pages)  # v1's run leaves the chart as it was
    X = extra_matrix([(1, 3), (2, 3)], ChartFacts(), ["v1"], v1={(1, 3)})
    assert X[:, 0].tolist() == [1.0, 0.0]


def test_best_solver_is_never_worse_than_either():
    import numpy as np

    from seqlib.v2.decode import path_cost
    from seqlib.v3.solve import decode_v3, v3_costs

    rng = np.random.default_rng(3)
    n = 16
    probs = {(a, b): float(rng.uniform(0.01, 0.99)) for a in range(n + 1) for b in range(n + 1) if a != b}
    cost = v3_costs(n, probs, 0.5)
    best = path_cost(decode_v3(n, probs, 0.5, "best"), cost)
    for solver in ("search", "cover"):
        assert best <= path_cost(decode_v3(n, probs, 0.5, solver), cost) + 1e-9


def test_v9_trains_on_position_and_runs(data):
    from seqlib.io.chart import load_chart
    from seqlib.v7.model import V7Model, train_v7
    from seqlib.v7.runner import run_v7
    from seqlib.versions import model_path, settings_for

    keys = sorted(data)
    settings = settings_for("v9")
    assert {"markers", "content", "pattern"} <= set(settings.groups)
    assert {"mk_follows", "ct_sec_delta", "pt_with_trend"} <= set(settings.feature_names)
    model = train_v7(((load_chart(k), data[k]) for k in keys[:10]), settings, folds=3, margins=(0.0, 2.0))
    rep = model.report
    assert rep["tuned_on"] == "position" and rep["decode"] == {"reverse_share": 0.25, "swaps": True}
    assert any(k.startswith("marker:") for k in rep["anchors_by_source"])  # synthetic "Page i of k"
    assert rep["anchors_correct"] == 1.0
    model.save(model_path("v9"))
    model = V7Model.load(model_path("v9"))
    for key in keys[10:]:
        r = run_v7(load_chart(key), model, pages=data[key].main_order)
        assert sorted(r.main_order) == sorted(data[key].main_order) and r.method.startswith("v9:gate")
