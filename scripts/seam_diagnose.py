"""Why the seam model learns so little: counts and AUCs on its own kind of pairs (no text leaves).

    $env:SEQ_CONFIG = "<repo>\\workspace\\shuffle_train\\config_shuffled.py"
    python scripts\\seam_diagnose.py              # ~20-30 min on CPU (scores every pair with the seam)
    python scripts\\seam_diagnose.py --no-model   # text statistics + the TF-IDF baseline only (minutes)

Builds the seam's training pairs (a's ending + b's opening; true: b is the next page; wrong: the scan
neighbour, the most similar-looking page, a random page) on

* unseen -- training documents the seam did not learn from (its held-out tenth and the half that
  v8 / v9 fit on);
* seen   -- a sample of the documents it learned from (seen much better than unseen = over-fitting;
  both poor = it cannot learn from these pairs as they are);

and reports, for each:

1. the pairs: how many of each kind, the share of true pairs, and the loss of a model that only
   guesses that share (compare with the training loss);
2. the text the seam reads: words in a's ending / b's opening, empty ones; tokens, and how many
   pairs max_length cuts off;
3. conflicts: wrong pairs whose input is identical to a true pair's (same two texts, opposite
   labels); wrong pages that open exactly like the true next page;
4. AUC of the seam and of the plain TF-IDF wording similarity (v1 / v2's feature) -- all pairs, true
   pairs against each kind of wrong pair, and true pairs split by link kind (within one pipeline
   document / a document boundary) and by the strongest clue the two pages give (link_report's
   evidence classes); and how often the true next page outscores every wrong page of its a;
5. the seam's loss on these pairs against the guessing loss.

Writes <OUTPUT_ROOT>/seam_diagnosis.json (collect_results.py copies it as
shuffle_train_seam_diagnosis.json). Counts and scores only.
"""
from __future__ import annotations

import argparse
import json
import logging
import math
import random
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

import _common  # noqa: F401

logging.getLogger("seqlib.v1").setLevel(logging.WARNING)

from link_report import EVIDENCE, link_evidence, link_kind, original_csv  # noqa: E402
from seqlib.eval.split import read_split  # noqa: E402
from seqlib.io import pipeline_csv  # noqa: E402
from seqlib.io.chart import load_chart  # noqa: E402
from seqlib.io.ground_truth import load_ground_truth  # noqa: E402
from seqlib.ladder.seam import META, NEGATIVE_KINDS, training_pairs  # noqa: E402
from seqlib.settings import cfg  # noqa: E402


def auc(pos, neg):
    if not len(pos) or not len(neg):
        return None
    from sklearn.metrics import roc_auc_score

    return round(float(roc_auc_score([1] * len(pos) + [0] * len(neg), list(pos) + list(neg))), 4)


def doc_rows(name, truth, manifest, original, rng) -> list[dict]:
    """The seam's pairs for one document, each with its TF-IDF score and (true pairs) link kind / evidence."""
    from seqlib.ladder.content import page_contents
    from seqlib.ladder.facts import chart_facts
    from seqlib.ladder.markers import page_markers
    from seqlib.v2.features import chart_features

    chart = load_chart(name)
    rows = training_pairs([(chart, truth)], rng, negatives=NEGATIVE_KINDS)  # every kind of wrong page
    if not rows:
        return []
    order = [p for p in truth.main_order if p in chart.by_number()]
    cf = chart_features(chart, order)
    facts = chart_facts(chart, ["banner", "dates"])
    texts = {p.page_number: p.inp.text for p in cf.pages}
    markers, contents = page_markers(texts), page_contents(texts)
    out = []
    for tail, head, label, _doc, kind, a, b in rows:
        r = {"doc": name, "a": a, "tail": tail, "head": head, "label": label, "kind": kind,
             "tfidf": float(cf.char_sim[cf.index[a], cf.index[b]])}
        if label:
            pa, pb = cf.pages[cf.index[a]], cf.pages[cf.index[b]]
            r["link"] = link_kind(manifest[name], original.get(name, {}), a, b) if name in manifest else "unknown"
            r["evidence"] = link_evidence(cf, facts, markers, contents, a, b)
            r["continues"] = bool((pa.cut_mid and pb.starts_lower) or pa.says_cont)
        out.append(r)
    return out


def word_stats(texts) -> dict:
    n = np.array([len(t.split()) for t in texts]) if texts else np.zeros(0)
    if not len(n):
        return {}
    return {"median words": float(np.median(n)), "10th percentile": float(np.percentile(n, 10)),
            "90th percentile": float(np.percentile(n, 90)), "empty share": round(float(np.mean(n == 0)), 4),
            "under 10 words share": round(float(np.mean(n < 10)), 4)}


def ranked_first(rows, key) -> float | None:
    """Share of a's whose true next page scores above every wrong page built for it."""
    by_a = defaultdict(lambda: [None, []])
    for r in rows:
        if key not in r:
            continue
        slot = by_a[(r["doc"], r["a"])]
        if r["label"]:
            slot[0] = r[key]
        else:
            slot[1].append(r[key])
    wins = [t > max(w) for t, w in by_a.values() if t is not None and w]
    return round(float(np.mean(wins)), 4) if wins else None


def summarise(rows: list[dict], documents: int, tokens: list[int] | None, max_length: int) -> dict:
    pos = [r for r in rows if r["label"]]
    neg = [r for r in rows if not r["label"]]
    share = len(pos) / max(1, len(rows))
    out: dict = {
        "documents": documents,
        "pairs": len(rows),
        "true pairs": len(pos),
        "wrong pairs by kind": dict(Counter(r["kind"] for r in neg)),
        "true share": round(share, 4),
        "loss of only guessing the share": round(-(share * math.log(share) + (1 - share) * math.log(1 - share)), 4)
        if 0 < share < 1 else None,
        "text the seam reads (true pairs)": {"a's ending": word_stats([r["tail"] for r in pos]),
                                             "b's opening": word_stats([r["head"] for r in pos])},
    }
    if tokens:
        t = np.array(tokens)
        out["tokens per pair"] = {"median": float(np.median(t)), "90th percentile": float(np.percentile(t, 90)),
                                  f"over max_length {max_length} share": round(float(np.mean(t > max_length)), 4)}
    true_inputs = {(r["tail"], r["head"]) for r in pos}
    true_head = {(r["doc"], r["a"]): r["head"] for r in pos}
    out["conflicts"] = {
        "wrong pairs with exactly a true pair's input": sum((r["tail"], r["head"]) in true_inputs for r in neg),
        "wrong pages opening exactly like the true next page": sum(
            r["head"] == true_head.get((r["doc"], r["a"])) for r in neg),
        "true pairs with an empty side": sum(not r["tail"].strip() or not r["head"].strip() for r in pos),
    }
    out["true pairs by link kind"] = dict(Counter(r["link"] for r in pos))
    out["true pairs by evidence"] = {e: c for e in EVIDENCE if (c := sum(r["evidence"] == e for r in pos))}
    out["true pairs that visibly continue (cut mid-sentence + lower-case start, or 'continued')"] = sum(
        r["continues"] for r in pos)

    for key in ("seam", "tfidf"):
        if not rows or key not in rows[0]:
            continue
        s = lambda rs: [r[key] for r in rs]  # noqa: E731
        block = {
            "all pairs": auc(s(pos), s(neg)),
            "true next ranked above all its wrong pages": ranked_first(rows, key),
            "true vs each kind of wrong": {k: auc(s(pos), s([r for r in neg if r["kind"] == k])) for k in NEGATIVE_KINDS},
            "by link kind (those true pairs vs all wrong)": {
                k: {"true pairs": n, "auc": auc(s([r for r in pos if r["link"] == k]), s(neg))}
                for k, n in sorted(Counter(r["link"] for r in pos).items())},
            "by evidence (those true pairs vs all wrong)": {
                e: {"true pairs": n, "auc": auc(s([r for r in pos if r["evidence"] == e]), s(neg))}
                for e in EVIDENCE if (n := sum(r["evidence"] == e for r in pos))},
            "visibly continues vs not (vs all wrong)": {
                "continues": auc(s([r for r in pos if r["continues"]]), s(neg)),
                "does not": auc(s([r for r in pos if not r["continues"]]), s(neg))},
        }
        out[f"auc: {key}"] = block
    if rows and "seam" in rows[0]:
        p = np.clip(np.array([r["seam"] for r in rows]), 1e-6, 1 - 1e-6)
        y = np.array([r["label"] for r in rows])
        out["seam loss on these pairs"] = round(float(-np.mean(y * np.log(p) + (1 - y) * np.log(1 - p))), 4)
        out["seam mean score"] = {"true": round(float(p[y == 1].mean()), 4), "wrong": round(float(p[y == 0].mean()), 4)}
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--no-model", action="store_true", help="skip scoring with the seam (text statistics + TF-IDF only)")
    ap.add_argument("--max-docs", type=int, default=0, help="at most this many unseen documents (0 = all)")
    ap.add_argument("--seen-docs", type=int, default=40, help="documents the seam learned from, to score as well")
    args = ap.parse_args()

    manifest_path = Path(cfg.GROUND_TRUTH_PATH).parent / "shuffle_manifest.json"
    if not manifest_path.is_file():
        raise SystemExit("run with the training shuffle set's config (workspace\\shuffle_train\\config_shuffled.py)")
    manifest = {d["doc"]: d for d in json.loads(manifest_path.read_text(encoding="utf-8"))["documents"]}
    original = pipeline_csv.load_pipeline_csv(original_csv(""))
    meta_path = Path(cfg.SEAM_MODEL_PATH) / META
    if not meta_path.is_file():
        raise SystemExit(f"no seam model at {cfg.SEAM_MODEL_PATH} (train_seam.py first)")
    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    trained_on, held = set(meta.get("trained_on") or []), set(meta.get("held_out_documents") or [])

    truths = load_ground_truth(cfg.GROUND_TRUTH_PATH)
    names = sorted(n for n in read_split(cfg.SPLIT_FILE)["train"] if n in truths and not truths[n].problems())
    rng = random.Random(0)
    unseen = [n for n in names if n not in trained_on or n in held]
    if args.max_docs:
        unseen = sorted(rng.sample(unseen, min(args.max_docs, len(unseen))))
    seen_pool = sorted(n for n in names if n in trained_on and n not in held)
    seen = sorted(rng.sample(seen_pool, min(args.seen_docs, len(seen_pool))))

    scorer = None
    if not args.no_model:
        from seqlib.ladder.seam import SeamScorer

        scorer = SeamScorer(cfg.SEAM_MODEL_PATH)
    max_length = int(meta.get("max_length") or 256)
    report: dict = {"seam model": {k: meta.get(k) for k in (
        "pairs", "held_out_pairs", "negatives", "held_out_auc", "held_out_auc_by_kind", "epochs", "epoch_curve",
        "lr", "max_length") if k in meta}}
    for group, docs in (("unseen", unseen), ("seen", seen)):
        rows, tokens, done = [], [], 0
        for i, name in enumerate(docs, start=1):
            try:
                got = doc_rows(name, truths[name], manifest, original, random.Random(i))
            except Exception as exc:  # noqa: BLE001
                print(f"  a document was not read ({type(exc).__name__})")
                continue
            if not got:
                continue
            done += 1
            if scorer is not None:
                for r, p in zip(got, scorer.score([r["tail"] for r in got], [r["head"] for r in got])):
                    r["seam"] = float(p)
                tokens += [len(scorer.tokenizer(r["tail"], r["head"])["input_ids"]) for r in got]
            rows += got
            if i % 20 == 0 or i == len(docs):
                print(f"{group}: {i}/{len(docs)} documents, {len(rows)} pairs")
        report[group] = summarise(rows, done, tokens or None, max_length)

    out = Path(cfg.OUTPUT_ROOT) / "seam_diagnosis.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=1), encoding="utf-8")
    for group in ("unseen", "seen"):
        g = report[group]
        print(f"\n{group}: {g['documents']} documents, {g['pairs']} pairs, true share {g['true share']}, "
              f"guessing loss {g['loss of only guessing the share']}")
        for key in ("seam", "tfidf"):
            b = g.get(f"auc: {key}")
            if b:
                kinds = ", ".join(f"{k} {v}" for k, v in b["true vs each kind of wrong"].items())
                print(f"  {key:5} AUC {b['all pairs']}  (vs {kinds}); true next ranked first "
                      f"{b['true next ranked above all its wrong pages']}")
        if "seam loss on these pairs" in g:
            print(f"  seam loss {g['seam loss on these pairs']}")
    print(f"\n-> {out}  (counts and scores only)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
