"""Why does v2 not repair an order: the pair model, or the search? (run with a shuffle set's config)

For every test document (the truth's Main pages, so blank / junk play no part):

1. **Model or search.** The decoder minimises the cost of an order (the sum of -log P(next | page)).
   It compares three orders under the model's own costs: the truth, what v2 decoded, and the scan
   order. truth cheaper than decoded -> the *search* missed a better order the model already
   prefers; truth dearer than decoded -> the *model* prefers a wrong order (the search did its job).
2. **The pair model on its own.** For each page (and the virtual START), is its true next page
   the model's first choice among its candidates? Split by whether the true next page is also the
   scan neighbour (the easy case) or was moved away by the shuffle (the case that matters). Also:
   the probability the model gives the true next page vs the scan neighbour in the moved case.
3. **What it leans on.** Permutation importance of the inputs on the test pairs.

    $env:SEQ_CONFIG = "<repo>\\workspace\\shuffle_eval\\config_shuffled.py"
    python scripts\\diagnose_v2.py

Writes <OUTPUT_ROOT>/diagnose_v2.json. Scores and counts only -- safe to bring back.
"""
import argparse
import json
import logging
import statistics

import _common  # noqa: F401

import numpy as np

logging.getLogger("seqlib.v1").setLevel(logging.WARNING)

from seqlib.eval.split import read_split  # noqa: E402
from seqlib.io.chart import load_chart  # noqa: E402
from seqlib.io.ground_truth import load_ground_truth  # noqa: E402
from seqlib.settings import cfg  # noqa: E402
from seqlib.v2.decode import cost_matrix, decode, path_cost  # noqa: E402
from seqlib.v2.features import FEATURES, candidates, pair_matrix  # noqa: E402
from seqlib.v2.model import PairScorer, true_pairs  # noqa: E402
from seqlib.v2.runner import score_pairs  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--split", choices=("test", "train"), default="test")
    ap.add_argument("--importance-rows", type=int, default=60000, help="pairs sampled for the importance")
    args = ap.parse_args()
    scorer = PairScorer.load(cfg.V2_MODEL_PATH)
    truths = load_ground_truth(cfg.GROUND_TRUTH_PATH)
    names = [n for n in read_split(cfg.SPLIT_FILE)[args.split] if n in truths and not truths[n].problems()]

    verdicts = {"solved": 0, "search missed a better order": 0, "model prefers a wrong order": 0}
    gaps = []  # (cost decoded - cost truth) per document
    first_choice = {"stays next to its scan neighbour": [0, 0], "moved away by the shuffle": [0, 0]}
    p_true_moved, p_scan_moved, recall = [], [], [0, 0]
    X_all, y_all = [], []
    for name in names:
        truth = truths[name]
        try:
            chart = load_chart(name)
        except Exception as exc:  # noqa: BLE001
            print(f"  skipped {name}: {exc}")
            continue
        pages = truth.main_order
        if len(pages) < 2:
            continue
        scored = score_pairs(chart, scorer, pages)
        n = len(scored.features.pages)
        node = scored.node
        cost = cost_matrix(n, scored.edge)
        truth_nodes = [node[p] for p in pages]
        decoded = decode(n, scored.edge, max_block=scorer.settings.max_block)
        c_truth, c_dec = path_cost(truth_nodes, cost), path_cost(decoded, cost)
        gaps.append(c_dec - c_truth)
        if decoded == truth_nodes:
            verdicts["solved"] += 1
        elif c_truth < c_dec - 1e-6:
            verdicts["search missed a better order"] += 1
        else:
            verdicts["model prefers a wrong order"] += 1

        # the pair model on its own
        out_edges: dict[int, list[tuple[float, int]]] = {}
        for (a, b), p in scored.edge.items():
            out_edges.setdefault(a, []).append((p, b))
        full = [0, *truth_nodes, 0]
        for a, b in zip(full, full[1:]):
            options = out_edges.get(a, [])
            recall[1] += 1
            if any(x == b for _, x in options):
                recall[0] += 1
            if a == 0 or b == 0:
                continue
            kind = "stays next to its scan neighbour" if b == a + 1 else "moved away by the shuffle"
            best = max(options)[1] if options else None
            first_choice[kind][0] += int(best == b)
            first_choice[kind][1] += 1
            if kind == "moved away by the shuffle":
                p_true_moved.append(scored.edge.get((a, b), 0.0))
                if a + 1 <= n:
                    p_scan_moved.append(scored.edge.get((a, a + 1), 0.0))

        # pairs for the importance
        cf = scored.features
        pairs = candidates(cf, scorer.settings.window, scorer.settings.top_k)
        positives = true_pairs(pages)
        X_all.append(pair_matrix(cf, pairs))
        y_all.append(np.array([p in positives for p in pairs], dtype=np.int8))

    report = {
        "documents": sum(verdicts.values()),
        "model or search": verdicts,
        "median cost gap (decoded - truth; < 0 = model prefers the decoded order)":
            round(statistics.median(gaps), 3) if gaps else None,
        "candidate recall (true next page among the scored candidates)":
            round(recall[0] / recall[1], 4) if recall[1] else None,
        "pair model: true next page is its first choice": {
            k: {"pairs": v[1], "first_choice": round(v[0] / v[1], 4) if v[1] else None}
            for k, v in first_choice.items()},
        "moved pairs: median probability of the true next page": round(statistics.median(p_true_moved), 4)
        if p_true_moved else None,
        "moved pairs: median probability of the (wrong) scan neighbour": round(statistics.median(p_scan_moved), 4)
        if p_scan_moved else None,
    }
    if X_all:
        from sklearn.inspection import permutation_importance

        X, y = np.vstack(X_all), np.concatenate(y_all)
        rng = np.random.default_rng(0)
        if len(y) > args.importance_rows:
            keep = rng.choice(len(y), args.importance_rows, replace=False)
            X, y = X[keep], y[keep]
        active = scorer.active
        result = permutation_importance(scorer.model, X[:, active], y, n_repeats=3, random_state=0,
                                        scoring="average_precision")
        order = np.argsort(-result.importances_mean)[:20]
        report["importance (drop in average precision when shuffled)"] = [
            (FEATURES[active[i]], round(float(result.importances_mean[i]), 4)) for i in order]
    out = cfg.OUTPUT_ROOT / "diagnose_v2.json"
    out.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))
    print(f"-> {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
