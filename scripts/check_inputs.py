"""Print every configured path, whether it exists, and what it holds.

    python scripts/check_inputs.py              # config summary + first 20 charts
    python scripts/check_inputs.py --all        # every chart
"""
import argparse
from collections import Counter

import _common  # noqa: F401

from seqlib.io.chart import chart_dirs, load_chart
from seqlib.io.ground_truth import load_ground_truth
from seqlib.settings import cfg


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--all", action="store_true")
    args = ap.parse_args()

    print(f"config: {cfg.__file__}\n")
    for key in ("RAW_DOCS_ROOT", "OCR_DOCS_ROOT", "PIPELINE_OUTPUT_CSV", "GROUND_TRUTH_PATH", "CHARTS_ROOT",
                "SPLIT_FILE", "SETASIDE_MODEL_PATH", "V2_MODEL_PATH", "EMBEDDING_MODEL_PATH", "V1_CANON_DIR", "TESSERACT_CMD", "OUTPUT_ROOT"):
        value = getattr(cfg, key, None)
        state = "-" if value is None else ("ok" if value.exists() else "MISSING")
        print(f"  {key:22} {state:8} {value}")
    print(f"  GT_SEQUENCE_COLUMN = {cfg.GT_SEQUENCE_COLUMN!r}, DOC_KEY_REGEX = {cfg.DOC_KEY_REGEX!r}")

    from seqlib.io.pipeline_csv import load_pipeline_csv

    pipeline = load_pipeline_csv(cfg.PIPELINE_OUTPUT_CSV)
    if pipeline:
        print(f"\npipeline outputs: {len(pipeline)} documents, {sum(len(v) for v in pipeline.values())} pages")

    dirs = chart_dirs()
    print(f"\ncharts under CHARTS_ROOT: {len(dirs)}")
    missing = Counter()
    pages = 0
    sources = Counter()
    for i, d in enumerate(dirs):
        try:
            chart = load_chart(d)
        except Exception as exc:  # noqa: BLE001
            print(f"  {d.name}: cannot load ({exc})")
            continue
        pages += len(chart.pages)
        missing.update(chart.missing)
        sources.update(p.ocr_source or "none" for p in chart.pages)
        if args.all or i < 20:
            print(f"  {chart.name}: {len(chart.pages)} pages; missing: {', '.join(chart.missing) or 'nothing'}")
    if dirs:
        print(f"\npages: {pages}; text from: {dict(sources)}")
        print(f"optional inputs missing (charts): {dict(missing) or 'none'}")

    if cfg.GROUND_TRUTH_PATH.exists():
        truths = load_ground_truth(cfg.GROUND_TRUTH_PATH)
        bad = {n: t.problems() for n, t in truths.items() if t.problems()}
        names = {d.name for d in dirs}
        gaps = {n: t.warnings() for n, t in truths.items() if t.warnings() and n not in bad}
        print(f"\nground truth: {len(truths)} charts, {len(bad)} left out (order unclear), "
              f"{len(gaps)} kept with gaps in their numbering, {len(set(truths) & names)} also under CHARTS_ROOT")
        flagged = sum(len(t.flagged_numbered) for t in truths.values())
        print(f"  blank / junk / invoice pages (moved to the J series): {flagged}")
        for name, problems in list(bad.items())[:10]:
            print(f"  LEFT OUT {name}: {'; '.join(problems)}")
        for name, warning in list(gaps.items())[:5]:
            print(f"  kept     {name}: {'; '.join(warning)}")
        without_ocr = sorted(set(truths) - names)
        if without_ocr:
            print(f"  ground-truth charts with no chart folder: {len(without_ocr)} (e.g. {without_ocr[:5]})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
