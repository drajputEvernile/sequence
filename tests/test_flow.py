"""The training machine's flow on synthetic data shaped like it: RecordID_DocumentID folders of
images and Azure OCR, the pipeline outputs CSV and the truth CSV (with documents the folders lack)
-> inventory -> import_ocr -> v1 / v1_csv / scan -> v2."""
from __future__ import annotations

import json
import shutil

import pytest

from seqlib.settings import cfg
from synthetic import make_raw_dataset

GT = cfg.PIPELINE_OUTPUT_CSV.parent / "ground_truth.csv"
N_DOCS, EXTRA = 10, 2


@pytest.fixture(scope="module")
def raw():
    shutil.rmtree(cfg.PIPELINE_OUTPUT_CSV.parent, ignore_errors=True)
    truths = make_raw_dataset(cfg.RAW_DOCS_ROOT, cfg.OCR_DOCS_ROOT, cfg.PIPELINE_OUTPUT_CSV, GT,
                              n_docs=N_DOCS, extra_docs=EXTRA)
    for key in truths:
        shutil.rmtree(cfg.CHARTS_ROOT / key, ignore_errors=True)
    return truths


def truths_of():
    from seqlib.io.ground_truth import load_ground_truth

    return load_ground_truth(GT, sequence_column="page_sequence")


def test_truth_export_is_read_by_page_number(raw):
    truths = truths_of()
    assert set(truths) == set(raw)
    key = sorted(raw)[0]
    assert truths[key].main_order == raw[key]  # page_number, not the row id
    assert len(truths[key].junk_pages) == 1 and not truths[key].problems()


def test_pipeline_csv_reader():
    from seqlib.io import pipeline_csv

    assert pipeline_csv.iso_date("1/6/2025") == "2025-01-06"
    assert pipeline_csv.iso_date("2/22/2222") == ""  # the non-visit default is no date
    assert pipeline_csv.group({"blank_junk_flag": "not_blank_junk"}) == "main"
    assert pipeline_csv.group({"blank_junk_flag": "duplicate"}) == "duplicate"
    assert pipeline_csv.type_path({"model_type": "Lab Report > Radiology Report > Others"}) == (
        "Lab Report", "Radiology Report", "Others")


def test_inventory_uses_only_documents_with_all_four(raw):
    from seqlib.eval.inventory import build
    from seqlib.io.pipeline_csv import load_pipeline_csv

    pipeline = load_pipeline_csv(cfg.PIPELINE_OUTPUT_CSV)
    docs, summary = build(cfg.RAW_DOCS_ROOT, cfg.OCR_DOCS_ROOT, truths_of(), cfg.DOC_KEY_REGEX, pipeline)
    assert summary["documents_in_pipeline_csv"] == N_DOCS + EXTRA
    assert summary["documents_in_truth_csv"] == N_DOCS + EXTRA
    assert summary["documents_with_images_and_ocr"] == N_DOCS
    assert summary["USABLE (all four)"] == N_DOCS
    assert summary["in_truth_but_no_images"] == EXTRA
    assert summary["usable_where_page_counts_agree"] == N_DOCS
    assert "azure document intelligence (analyzeResult)" in summary["ocr_formats (documents)"]
    assert "test person" not in json.dumps(summary)  # no page text in the report


def test_import_and_chart_reads_pipeline_values(raw):
    from seqlib.eval.inventory import source_files, usable_documents
    from seqlib.extract.import_ocr import import_document
    from seqlib.io.chart import load_chart
    from seqlib.io.pipeline_csv import load_pipeline_csv

    images, ocr = source_files(cfg.RAW_DOCS_ROOT, cfg.OCR_DOCS_ROOT, cfg.DOC_KEY_REGEX)
    usable = usable_documents(truths_of(), cfg.DOC_KEY_REGEX, cfg.RAW_DOCS_ROOT, cfg.OCR_DOCS_ROOT,
                              load_pipeline_csv(cfg.PIPELINE_OUTPUT_CSV))
    assert len(usable) == N_DOCS
    for key in usable:
        report = import_document(key, images[key], ocr[key])
        assert report["ocr_pages"] == report["images"]
        chart = load_chart(key)
        assert all(p.pipeline for p in chart.pages) and not chart.missing
        assert sum(p.group == "junk" for p in chart.pages) == 1  # from blank_junk_flag
        assert chart.pages[0].final2["pagesMeta"][0]["words"]


def test_only_the_ocr_subfolder_is_read(raw):
    from seqlib.eval.inventory import _entries, source_files

    _, ocr = source_files(cfg.RAW_DOCS_ROOT, cfg.OCR_DOCS_ROOT, cfg.DOC_KEY_REGEX)
    key = sorted(raw)[0]
    assert [p.parent.name for p in ocr[key]] == ["ocr"]  # not staging/extraction.json, not imaging/
    everything = _entries(cfg.OCR_DOCS_ROOT, cfg.DOC_KEY_REGEX)  # without the subfolder setting
    assert {p.parent.name for p in everything[key]} == {"ocr", "staging", "imaging"}


def test_per_page_azure_files_convert(tmp_path):
    from seqlib.extract.ocr_convert import convert_document

    make_raw_dataset(tmp_path / "raw", tmp_path / "ocr", tmp_path / "p.csv", tmp_path / "gt.csv",
                     n_docs=1, extra_docs=0, per_page_ocr=True)
    key = next((tmp_path / "ocr").iterdir()).name
    files = sorted((tmp_path / "ocr" / key / "ocr").iterdir())
    converted = convert_document(files, key, {})
    assert sorted(converted.final2) == list(range(1, len(files) + 1))


def test_unknown_ocr_format_names_its_keys(tmp_path):
    from seqlib.extract.ocr_convert import UnknownOcrFormat, convert_document

    path = tmp_path / "1_2.json"
    path.write_text(json.dumps({"foo": 1, "bar": []}), encoding="utf-8")
    with pytest.raises(UnknownOcrFormat, match="bar, foo"):
        convert_document([path], "1_2", {})


def test_v1_csv_v1_and_scan_score(raw):
    from seqlib.eval.metrics import score_chart, summarize
    from seqlib.io.chart import load_chart
    from seqlib.v1.runner import documents, run_scan_order, run_v1, run_v1_csv

    truths = truths_of()
    got = {"v1_csv": [], "v1": [], "scan": []}
    for key in sorted(raw)[:N_DOCS]:
        chart = load_chart(key)
        docs = documents(chart, "csv")
        assert set(v for v in docs.values() if v)  # the pipeline's document_seq
        for name, result in (("v1_csv", run_v1_csv(chart)), ("v1", run_v1(chart)), ("scan", run_scan_order(chart))):
            got[name].append(score_chart(truths[key], result.main_order, result.junk_pages))
    # the synthetic production seq is scan order: v1_csv must score exactly like scan
    assert summarize(got["v1_csv"]) == summarize(got["scan"])
    assert summarize(got["v1"])["position_accuracy"] >= summarize(got["scan"])["position_accuracy"]


def test_v2_trains_on_pipeline_features(raw):
    from seqlib.io.chart import load_chart
    from seqlib.v2.features import FEATURES
    from seqlib.v2.model import train
    from seqlib.v2.runner import run_v2

    truths = truths_of()
    keys = sorted(raw)[:N_DOCS]
    scorer = train((load_chart(k), truths[k]) for k in keys[:7])
    used = {FEATURES[i] for i in scorer.active}
    assert used and not {"same_doc_type", "b_type_continued"} & set(FEATURES)  # order-dependent: not read
    for key in keys[7:]:
        result = run_v2(load_chart(key), scorer)
        assert sorted(result.main_order) == sorted(truths[key].main_order)


def test_blank_junk_pages_run_their_own_series(tmp_path):
    import csv

    from seqlib.io.ground_truth import load_ground_truth

    export = tmp_path / "export.csv"
    with open(export, "w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(["id", "chart_name", "page_number", "blank_page", "junk_page", "is_invoice", "page_sequence"])
        w.writerow([9001, "1_2", 1, "No", "Yes", "No", 1])   # junk: its own series
        w.writerow([9002, "1_2", 2, "No", "No", "No", 1])
        w.writerow([9003, "1_2", 3, "No", "No", "No", 2])
        w.writerow([9004, "1_2", 4, "Yes", "No", "No", "J2"])
    t = load_ground_truth(export, sequence_column="page_sequence")["1_2"]
    assert t.main_order == [2, 3] and t.junk_pages == [1, 4] and not t.problems()

    gap = tmp_path / "gap.csv"
    with open(gap, "w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(["id", "chart_name", "page_number", "blank_page", "junk_page", "is_invoice", "page_sequence"])
        w.writerow([1, "1_2", 1, "No", "Yes", "No", 1])  # flagged, but it is Main number 1
        w.writerow([2, "1_2", 2, "No", "No", "No", 2])
    t = load_ground_truth(gap, sequence_column="page_sequence")["1_2"]
    assert t.problems() == []  # a gap leaves the order clear: kept
    assert t.warnings() == ["sequence numbers missing: [1]; page 1 has number 1 but is marked blank / junk / invoice"]
    assert t.main_order == [2]


def test_setaside_model_trains_and_applies(raw):
    from seqlib.eval.metrics import score_chart
    from seqlib.io.chart import load_chart
    from seqlib.v1.runner import run_scan_order
    from seqlib.v2.setaside import SetAsideModel, train

    truths = truths_of()
    keys = sorted(raw)[:N_DOCS]
    examples = [(load_chart(k), truths[k]) for k in keys]
    model = train(examples, folds=3)
    rep = model.report
    assert rep["documents"] == N_DOCS and rep["set_aside_in_truth"] == N_DOCS  # one cover page each
    assert "pipeline_verdict (train, for comparison)" in rep and 0.1 <= model.threshold <= 0.9
    model.save(cfg.SETASIDE_MODEL_PATH)
    model = SetAsideModel.load(cfg.SETASIDE_MODEL_PATH)
    chart = model.apply(load_chart(keys[0]))
    assert all(p.group_override in ("main", "junk", "blank", "duplicate") for p in chart.pages)
    s = score_chart(truths[keys[0]], *(lambda r: (r.main_order, r.junk_pages))(run_scan_order(chart)))
    assert s.pages == len(truths[keys[0]].main_order)


def test_truth_set_aside_makes_strict_equal_position(raw):
    import importlib.util
    import sys
    from pathlib import Path

    from seqlib.eval.metrics import score_chart
    from seqlib.io.chart import load_chart
    from seqlib.v1.runner import run_scan_order

    root = Path(__file__).resolve().parents[1]
    sys.path.insert(0, str(root / "scripts"))
    spec = importlib.util.spec_from_file_location("evaluate_script", root / "scripts" / "evaluate.py")
    evaluate = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(evaluate)
    truths = truths_of()
    for key in sorted(raw)[:N_DOCS]:
        chart = load_chart(key)
        chart.pages[0].rows["pipeline"]["blank_junk_flag"] = "junk"  # a pipeline error on page 1
        evaluate.set_aside_by_truth(chart, truths[key])
        r = run_scan_order(chart)
        s = score_chart(truths[key], r.main_order, r.junk_pages)
        assert s.strict_correct == s.position_correct and s.junk_fp == 0 and s.junk_fn == 0


def test_explain_rules_score_every_document(raw):
    from seqlib.eval.explain import RULES, explain, rule_effect
    from seqlib.io.chart import load_chart

    truths = truths_of()
    moved, in_order = [], []
    for key in sorted(raw)[:N_DOCS]:
        t = truths[key]
        e = explain(load_chart(key), t)
        assert set(e.scores) == set(RULES)
        if t.main_order == sorted(t.main_order):
            assert e.scores["scan"] == 1.0 and e.best == "scan"
            in_order.append(e)
        else:
            assert e.scores["scan"] < 1.0 and e.table and all("page" in r and "dos" in r for r in e.table)
            assert all(r["dos"] == "-" or r["dos"].startswith("D") for r in e.table)  # no real dates
            moved.append(e)
    effect = rule_effect(moved, in_order)
    assert effect["scan"]["in_order_docs_broken"] == 0 and effect["scan"]["moved_docs_fixed"] == 0


def test_shuffle_ops_always_change_the_order():
    import random

    from seqlib.eval.shuffle import shuffle_order

    rng = random.Random(1)
    for n in (4, 5, 12, 40):
        pages = list(range(1, n + 1))
        for _ in range(30):
            order, ops = shuffle_order(pages, rng, 3)
            assert sorted(order) == pages and order != pages and ops


def test_shuffle_test_copies_renames_and_hides_order(raw, tmp_path):
    import csv

    from seqlib.eval.inventory import source_files
    from seqlib.eval.shuffle import CONFIG_TEMPLATE, build, choose
    from seqlib.io.ground_truth import load_ground_truth
    from seqlib.io.pipeline_csv import load_pipeline_csv

    truths = truths_of()
    pipeline = load_pipeline_csv(cfg.PIPELINE_OUTPUT_CSV)
    images, ocr = source_files(cfg.RAW_DOCS_ROOT, cfg.OCR_DOCS_ROOT, cfg.DOC_KEY_REGEX)
    before = {p: p.stat().st_mtime_ns for v in images.values() for p in v}
    docs = choose(truths, sorted(raw)[:N_DOCS], 5, 4, seed=3)
    out = tmp_path / "shuffle"
    made = build(out, docs, truths, pipeline, cfg.PIPELINE_OUTPUT_CSV, GT, images, ocr, seed=3, max_ops=3)
    assert {p: p.stat().st_mtime_ns for v in images.values() for p in v} == before  # originals untouched
    assert len(made) == len(docs) and all(m.pages_moved for m in made)

    new_truth = load_ground_truth(out / "ground_truth.csv", sequence_column="page_sequence")
    for m in made:
        # the truth order, renamed, is the original truth order mapped through the renaming
        assert new_truth[m.doc].main_order == [m.new_of[p] for p in truths[m.doc].main_order]
        assert sorted(p.name for p in (out / "raw_images" / m.doc).iterdir()) == sorted(
            f"{n}.png" for n in m.new_of.values())
    with open(out / "imaging_output_full.csv", newline="", encoding="utf-8") as fh:
        rows = list(csv.DictReader(fh))
    assert rows and all(r["seq"] == "" and r["document_seq"] == "" and r["page_type_source"] == "" for r in rows)
    manifest = json.loads((out / "shuffle_manifest.json").read_text(encoding="utf-8"))
    assert len(manifest["documents"]) == len(docs)
    assert CONFIG_TEMPLATE.startswith("r\"\"\"")


def test_shuffle_test_end_to_end_with_its_config(raw, tmp_path):
    """The generated config points every reader at the shuffle folder; scan order is broken there
    and the truth still loads."""
    import subprocess
    import sys
    from pathlib import Path

    from seqlib.eval.inventory import source_files
    from seqlib.eval.shuffle import CONFIG_TEMPLATE, build, choose
    from seqlib.io.pipeline_csv import load_pipeline_csv

    root = Path(__file__).resolve().parents[1]
    truths = truths_of()
    images, ocr = source_files(cfg.RAW_DOCS_ROOT, cfg.OCR_DOCS_ROOT, cfg.DOC_KEY_REGEX)
    docs = choose(truths, sorted(raw)[:N_DOCS], 4, 4, seed=5)
    out = tmp_path / "sh"
    build(out, docs, truths, load_pipeline_csv(cfg.PIPELINE_OUTPUT_CSV), cfg.PIPELINE_OUTPUT_CSV, GT,
          images, ocr, seed=5, max_ops=2)
    conf = out / "config_shuffled.py"
    conf.write_text(CONFIG_TEMPLATE.format(main=root / "tests" / "config_test.py", root=out, this=conf),
                    encoding="utf-8")
    env = {**__import__("os").environ, "SEQ_CONFIG": str(conf)}
    py = sys.executable
    run = lambda *a: subprocess.run([py, str(root / "scripts" / a[0]), *a[1:]], env=env, capture_output=True,
                                    text=True, cwd=root)
    r = run("import_ocr.py")
    assert r.returncode == 0, r.stdout + r.stderr
    assert f"done: {len(docs)} imported, 0 failed" in r.stdout
    r = run("evaluate.py", "--method", "v1_csv")
    assert r.returncode != 0 and "shuffle test" in (r.stdout + r.stderr)
    r = run("evaluate.py", "--method", "scan", "--set-aside", "truth")
    assert r.returncode == 0, r.stdout + r.stderr
    summary = json.loads(r.stdout[r.stdout.index("{"):r.stdout.rindex("}") + 1])
    assert summary["charts"] == len(docs) and summary["failed_charts"] == 0
    assert summary["perfect_charts"] < 1.0  # the shuffle broke scan order


def test_v2_reads_no_order_dependent_value(raw):
    from seqlib.io.chart import load_chart
    from seqlib.v2.features import order_free

    chart = load_chart(sorted(raw)[0])
    assert any(p.pipeline.get("document_seq") for p in chart.pages)
    free = order_free(chart)
    assert all(not p.pipeline.get("document_seq") and not p.pipeline.get("seq") for p in free.pages)
    assert any(p.pipeline.get("document_seq") for p in chart.pages)  # the original is untouched


def test_shuffle_report_finds_latest_evaluations(tmp_path):
    import importlib.util
    import sys
    from pathlib import Path

    root = Path(__file__).resolve().parents[1]
    sys.path.insert(0, str(root / "scripts"))
    spec = importlib.util.spec_from_file_location("shuffle_report", root / "scripts" / "shuffle_report.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    for name in ("scan_test_20261119-100000", "scan_test_20261119-110000", "v1-csv-setaside_truth_test_20261119-090000"):
        (tmp_path / "eval" / name).mkdir(parents=True)
        (tmp_path / "eval" / name / "per_chart.csv").write_text("chart\n", encoding="utf-8")
    found = mod.latest_evals(tmp_path)
    assert found["scan"].name.endswith("110000") and "v1-csv-setaside_truth" in found
