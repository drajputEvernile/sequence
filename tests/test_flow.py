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
    assert {"same_doc_v1", "b_type_continued"} & used
    for key in keys[7:]:
        result = run_v2(load_chart(key), scorer)
        assert sorted(result.main_order) == sorted(truths[key].main_order)
