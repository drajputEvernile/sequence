"""Shuffle test, link by link: which kinds of true links the shuffle broke and each method repaired.

    $env:SEQ_CONFIG = "<repo>\\workspace\\shuffle_eval\\config_shuffled.py"
    python scripts\\link_report.py
    python scripts\\link_report.py --methods scan-setaside_truth v8-setaside_truth v9-setaside_truth

Every true link a -> b (b directly follows a in the truth) gets a kind, read from the ORIGINAL
pipeline outputs (computed on the unshuffled order -- an oracle view, for diagnosis only):

* ``within document``      -- a and b are in the same pipeline document (document_seq);
* ``boundary, b later``    -- b starts the next pipeline document and its date is later than a's;
  ``boundary, same date`` / ``boundary, b earlier`` / ``boundary, undated`` likewise;
* and, separately, ``anchored`` or not -- a printed "n of m" -> "n+1 of m" (or fax counter) link
  that v4+'s anchor rule finds;
* and ``evidence`` -- the strongest clue the two pages themselves give that b follows a
  (EVIDENCE, strongest first; "none of these" when no clue points to it). Its broken links that
  carry no clue are the ones no content model can be expected to repair.

A link is *broken* when the shuffled scan order does not have b right after a. Per kind: links,
broken, and for each method (its latest evaluation on this shuffle set) the share of broken links
it repaired and the share of unbroken links it kept. Within-document links can be read from the
pages; boundary links between separate documents often cannot -- their share of the broken links
is roughly what no content model can recover.

Writes <OUTPUT_ROOT>/link_report.json. Counts and shares only -- safe to bring back.
"""
import argparse
import csv
import json
import runpy
from collections import defaultdict
from pathlib import Path

import _common  # noqa: F401

from seqlib.io import pipeline_csv
from seqlib.settings import cfg


def latest_evals(root: Path) -> dict[str, Path]:
    out: dict[str, tuple[str, Path]] = {}
    folder = root / "eval"
    for d in sorted(folder.iterdir()) if folder.is_dir() else []:
        parts = d.name.rsplit("_", 2)
        if len(parts) != 3 or not (d / "per_chart.csv").is_file():
            continue
        tag, _split, stamp = parts
        if tag not in out or stamp > out[tag][0]:
            out[tag] = (stamp, d)
    return {tag: d for tag, (_, d) in out.items()}


def original_csv(arg: str) -> Path:
    if arg:
        return Path(arg)
    main = getattr(cfg, "_MAIN", None)
    if main is None:
        raise SystemExit("not a shuffle config: set SEQ_CONFIG to the shuffle test's config (or pass --original-csv)")
    return Path(runpy.run_path(str(main))["PIPELINE_OUTPUT_CSV"])


EVIDENCE = (
    "printed page number (n of m / Page n)",
    "bare page number",
    "fax / print counter",
    "numbered list goes on",
    "sentence / table carries over",
    "same encounter id / date",
    "same date of service",
    "section order fits",
    "none of these",
)


def link_evidence(cf, facts, markers, contents, a: int, b: int) -> str:
    """The strongest clue that b follows a (EVIDENCE)."""
    ma, mb = markers.get(a), markers.get(b)
    if ma is not None and mb is not None and ma.style == mb.style and mb.number == ma.number + 1:
        return EVIDENCE[0] if ma.kind in ("of", "page") else EVIDENCE[1]
    ba, bb = facts.banner.get(a), facts.banner.get(b)
    if ba is not None and bb is not None and ba.identity == bb.identity and bb.counter == ba.counter + 1:
        return EVIDENCE[2]
    ca, cb = contents[a], contents[b]
    if ca.item_last is not None and cb.item_first == ca.item_last + 1:
        return EVIDENCE[3]
    pa, pb = cf.pages[cf.index[a]], cf.pages[cf.index[b]]
    if (pa.cut_mid and pb.starts_lower) or pa.says_cont or cb.cont_from or (ca.table_tail and cb.table_head):
        return EVIDENCE[4]
    if (ca.enc_id and ca.enc_id == cb.enc_id) or (ca.enc_date and ca.enc_date == cb.enc_date):
        return EVIDENCE[5]
    if facts.dos.get(a) and facts.dos.get(a) == facts.dos.get(b):
        return EVIDENCE[6]
    if ca.sec_last is not None and cb.sec_first is not None and cb.sec_first > ca.sec_last:
        return EVIDENCE[7]
    return EVIDENCE[8]


def read_links(names: list[str], pages_of: dict[str, list[int]]) -> tuple[dict[str, set], dict[str, dict]]:
    """Per document: the anchored links, and each true link's evidence."""
    from seqlib.io.chart import load_chart
    from seqlib.ladder.content import page_contents
    from seqlib.ladder.extra import find_anchors
    from seqlib.ladder.facts import chart_facts
    from seqlib.ladder.markers import page_markers
    from seqlib.v2.features import chart_features

    anchors, evidence = {}, {}
    for name in names:
        try:
            chart = load_chart(name)
            order = pages_of[name]
            cf = chart_features(chart, order)
            facts = chart_facts(chart, ["banner", "dates"])
            texts = {p.page_number: p.inp.text for p in cf.pages}
            markers, contents = page_markers(texts), page_contents(texts)
            printed = {p.page_number: p.printed for p in cf.pages if p.printed}
            anchors[name] = set(find_anchors(printed, facts))
            evidence[name] = {(a, b): link_evidence(cf, facts, markers, contents, a, b)
                              for a, b in zip(order, order[1:])}
        except Exception as exc:  # noqa: BLE001
            print(f"  {name}: pages not read ({exc})")
            anchors[name], evidence[name] = set(), {}
    return anchors, evidence


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--methods", nargs="*", help="evaluation tags (folder names without _<split>_<time>)")
    ap.add_argument("--original-csv", default="", help="the unshuffled pipeline outputs CSV (default: the main config's)")
    ap.add_argument("--no-pages", action="store_true",
                    help="skip reading the pages (no anchored / evidence split; faster)")
    args = ap.parse_args()
    manifest_path = Path(cfg.GROUND_TRUTH_PATH).parent / "shuffle_manifest.json"
    if not manifest_path.is_file():
        raise SystemExit(f"no shuffle manifest at {manifest_path}: set SEQ_CONFIG to the shuffle test's config")
    manifest = {d["doc"]: d for d in json.loads(manifest_path.read_text(encoding="utf-8"))["documents"]}
    original = pipeline_csv.load_pipeline_csv(original_csv(args.original_csv))
    evals = latest_evals(Path(cfg.OUTPUT_ROOT))
    tags = args.methods or sorted(evals)
    predicted: dict[str, dict[str, list[int]]] = {}
    margins_used: dict[str, dict[str, int]] = {}
    truth: dict[str, list[int]] = {}
    for tag in tags:
        if tag not in evals:
            print(f"no evaluation found for {tag}")
            continue
        with open(evals[tag] / "per_chart.csv", newline="", encoding="utf-8") as fh:
            for r in csv.DictReader(fh):
                if r["chart"] not in manifest:
                    continue
                predicted.setdefault(tag, {})[r["chart"]] = [int(x) for x in r["predicted"].split()]
                truth.setdefault(r["chart"], [int(x) for x in r["truth"].split()])
                detail = r.get("method_detail") or ""
                if ":gate" in detail:  # gated: "v9:gate0.83:m0.5" -> documents per margin used
                    used = margins_used.setdefault(tag, defaultdict(int))
                    used[detail.rsplit(":m", 1)[-1]] += 1
    names = sorted(truth)
    anchors, evidence = ({}, {}) if args.no_pages else read_links(names, truth)

    def kind(doc: str, a: int, b: int) -> str:
        inv = {int(new): int(old) for old, new in manifest[doc].get("renamed (original -> new)", {}).items()}
        rows = original.get(doc, {})
        ra, rb = rows.get(inv.get(a, a)) or {}, rows.get(inv.get(b, b)) or {}
        da, db = str(ra.get("document_seq") or "").strip(), str(rb.get("document_seq") or "").strip()
        if not da or not db:
            return "no pipeline document"
        if da == db:
            return "within document"
        ta, tb = pipeline_csv.iso_date(ra.get("dos_from")), pipeline_csv.iso_date(rb.get("dos_from"))
        if not ta or not tb:
            return "boundary, undated"
        return "boundary, b later" if tb > ta else "boundary, same date" if tb == ta else "boundary, b earlier"

    # links: (kind, anchored, broken, {tag: right})
    table = defaultdict(lambda: {"links": 0, "broken": 0, "methods": defaultdict(lambda: [0, 0])})
    for doc in names:
        order = truth[doc]
        scan_next = dict(zip(sorted(order), sorted(order)[1:]))
        nexts = {tag: dict(zip(p[doc], p[doc][1:])) for tag, p in predicted.items() if doc in p}
        for a, b in zip(order, order[1:]):
            broken = scan_next.get(a) != b
            k = kind(doc, a, b)
            dims = [("kind", k), ("all", "all")]
            if not args.no_pages:
                dims.append(("anchor", "anchored" if (a, b) in anchors.get(doc, set()) else "not anchored"))
                dims.append(("evidence", evidence.get(doc, {}).get((a, b), "pages not read")))
            for dim, value in dims:
                cell = table[(dim, value)]
                cell["links"] += 1
                cell["broken"] += broken
                for tag, nxt in nexts.items():
                    right = nxt.get(a) == b
                    if broken:
                        cell["methods"][tag][0] += right  # repaired
                    else:
                        cell["methods"][tag][1] += right  # kept
    report = {}
    for (dim, value), cell in sorted(table.items()):
        unbroken = cell["links"] - cell["broken"]
        report.setdefault(dim, {})[value] = {
            "links": cell["links"], "broken": cell["broken"],
            "methods": {tag: {"repaired": round(rep / cell["broken"], 4) if cell["broken"] else None,
                              "kept": round(kept / unbroken, 4) if unbroken else None}
                        for tag, (rep, kept) in sorted(cell["methods"].items())},
        }
    if margins_used:
        report["gated: documents per margin"] = {t: dict(sorted(u.items())) for t, u in margins_used.items()}
    if not report:
        print("no evaluation of a shuffle-set document found")
        return 1
    all_broken = report["all"]["all"]["broken"]
    out = Path(cfg.OUTPUT_ROOT) / "link_report.json"
    out.write_text(json.dumps(report, indent=2), encoding="utf-8")
    width = max([len(t) for t in predicted] + [10])
    for dim in ("all", "kind", "anchor", "evidence"):
        cells = report.get(dim, {})
        if dim == "evidence":
            cells = {v: cells[v] for v in (*EVIDENCE, "pages not read") if v in cells}
        for value, cell in cells.items():
            share = cell["broken"] / all_broken if all_broken else 0
            print(f"\n{value}: {cell['links']} links, {cell['broken']} broken by the shuffle ({share:.0%} of all broken)")
            for tag, s in cell["methods"].items():
                rep = "-" if s["repaired"] is None else f"{s['repaired']:.3f}"
                kept = "-" if s["kept"] is None else f"{s['kept']:.3f}"
                print(f"   {tag:{width}}  repaired {rep}   kept {kept}")
    for tag, used in report.get("gated: documents per margin", {}).items():
        print(f"\n{tag}: documents per margin used " + ", ".join(f"m{m}: {c}" for m, c in used.items()))
    print(f"\n-> {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
