"""Gather everything worth bringing back into <OUTPUT_ROOT>/share/, as small separate files that
can be copied out one at a time, in any order:

    00_scores.txt                   every method on every set: position, successor, perfect (start here)
    model_<v5|v8|v9|seam>.json      each trained model's report
    <set>_built.json                how a shuffle set was made (documents, operations per document, kinds)
    <set>_scores.json               every method's evaluation summary on that set (<set> = real, shuffle_ops5 ...)
    <set>_shuffle_report.json       by operation kind / count / document size (when run)
    <set>_link_report.json          which links each method repairs, by evidence (when run)

    python scripts\\collect_results.py          (with SEQ_CONFIG unset: the main config)

Scores, counts and settings only: no document names, page numbers or text (the evaluations'
per_chart.csv and errors.txt stay on this machine).
"""
import json
from collections import Counter
from pathlib import Path

import _common  # noqa: F401

from seqlib.settings import cfg
from seqlib.versions import VERSIONS, model_path

SHOW = ("version", "groups", "evidence", "documents", "tuned_on_documents", "tuned_in_order",
        "out_of_order_documents", "in_order_documents", "tuned_on", "decode", "candidate_recall",
        "anchors_found", "anchors_correct", "anchors_by_source", "chosen_margin", "margin_curve",
        "gate_auc_out_of_fold", "gate_mean_p", "gate_signs_used", "chosen", "without_gate",
        "margin_curve_out_of_fold", "unused_evidence")


def latest_summaries(root: Path) -> dict:
    found: dict[str, tuple[str, dict]] = {}
    folder = root / "eval"
    for d in sorted(folder.iterdir()) if folder.is_dir() else []:
        parts = d.name.rsplit("_", 2)
        if len(parts) != 3 or not (d / "summary.json").is_file():
            continue
        tag, split, stamp = parts
        key = f"{tag} ({split})"
        if key not in found or stamp > found[key][0]:
            s = json.loads((d / "summary.json").read_text(encoding="utf-8"))
            s.pop("failures", None)  # names documents; the count stays (failed_charts)
            found[key] = (stamp, s)
    return {k: v for k, (_, v) in sorted(found.items())}


def shuffle_set(folder: Path) -> dict:
    out: dict = {}
    manifest = folder / "shuffle_manifest.json"
    if manifest.is_file():
        docs = json.loads(manifest.read_text(encoding="utf-8"))["documents"]
        ops = [d["operations"] for d in docs]
        out["built"] = {
            "documents": len(docs),
            "controls (left in order)": sum(o == ["unchanged (control)"] for o in ops),
            "operations_per_document": dict(sorted(Counter(len(o) for o in ops if o != ["unchanged (control)"]).items())),
            "operation_kinds": dict(Counter(" ".join(x.split()[:2]) if x.startswith(("move", "reverse")) else x.split()[0]
                                            for o in ops if o != ["unchanged (control)"] for x in o)),
            "main_pages": sum(d.get("main_pages", 0) for d in docs),
        }
    root = folder / "outputs"
    out["evaluations"] = latest_summaries(root)
    for name in ("shuffle_report.json", "link_report.json"):
        if (root / name).is_file():
            out[name.removesuffix(".json")] = json.loads((root / name).read_text(encoding="utf-8"))
    return out


def write(folder: Path, name: str, data) -> Path:
    path = folder / name
    path.write_text(data if isinstance(data, str) else json.dumps(data, indent=1), encoding="utf-8")
    return path


def main() -> int:
    if getattr(cfg, "SHUFFLE_TEST", False):
        raise SystemExit("run with the main config (Remove-Item Env:SEQ_CONFIG)")
    import joblib

    share = Path(cfg.OUTPUT_ROOT) / "share"
    share.mkdir(parents=True, exist_ok=True)
    for old in share.glob("*"):
        if old.is_file():
            old.unlink()  # a fresh set of files each run
    written = []
    for v in VERSIONS:
        path = model_path(v)
        if path.is_file():
            report = joblib.load(path).report
            written.append(write(share, f"model_{v}.json", {k: report[k] for k in SHOW if k in report}))
    seam_meta = Path(getattr(cfg, "SEAM_MODEL_PATH", "")) / "seam_meta.json"
    if seam_meta.is_file():
        meta = json.loads(seam_meta.read_text(encoding="utf-8"))
        seam = {k: v for k, v in meta.items() if k not in ("trained_on", "held_out_documents", "base_model")}
        seam["documents"] = len(meta.get("trained_on") or [])
        written.append(write(share, "model_seam.json", seam))

    sets = {"real": {"evaluations": latest_summaries(Path(cfg.OUTPUT_ROOT))}}
    for folder in sorted((Path(cfg.REPO) / "workspace").glob("shuffle_*")):
        if folder.is_dir():
            sets[folder.name] = shuffle_set(folder)
    lines = [f"{'set':16} {'method':28} {'docs':>5} {'position':>9} {'successor':>9} {'perfect':>8}"]
    for name, block in sets.items():
        if "built" in block:
            written.append(write(share, f"{name}_built.json", block["built"]))
        if block.get("evaluations"):
            written.append(write(share, f"{name}_scores.json", block["evaluations"]))
        for report in ("shuffle_report", "link_report"):
            if report in block:
                written.append(write(share, f"{name}_{report}.json", block[report]))
        for method, s in block.get("evaluations", {}).items():
            fmt = lambda k: "-" if s.get(k) is None else f"{100 * s[k]:.1f}%"  # noqa: E731
            lines.append(f"{name:16} {method:28} {s.get('charts', 0):>5} {fmt('position_accuracy'):>9} "
                         f"{fmt('successor_accuracy'):>9} {fmt('perfect_charts'):>8}")
    table = "\n".join(lines) + "\n"
    written.insert(0, write(share, "00_scores.txt", table))
    print(table)
    print(f"-> {share}  ({len(written)} files; each can be brought back on its own)")
    for path in written:
        print(f"   {path.name:40} {path.stat().st_size / 1024:7.1f} KB")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
