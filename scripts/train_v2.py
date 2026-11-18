"""Train the v2 pair scorer on the split's train charts; save it to config.V2_MODEL_PATH.

    python scripts/train_v2.py
    python scripts/train_v2.py --window 10 --top-k 8      # wider candidate net

Then: python scripts/evaluate.py --method v2
"""
import argparse
import json

import _common  # noqa: F401

from seqlib.eval.split import read_split
from seqlib.io.chart import load_chart
from seqlib.io.ground_truth import load_ground_truth
from seqlib.settings import cfg
from seqlib.v2.features import FEATURES, Embedder
from seqlib.v2.model import V2Settings, train


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--window", type=int, default=V2Settings.window)
    ap.add_argument("--top-k", type=int, default=V2Settings.top_k)
    ap.add_argument("--max-block", type=int, default=V2Settings.max_block)
    args = ap.parse_args()

    truths = load_ground_truth(cfg.GROUND_TRUTH_PATH)
    names = [n for n in read_split(cfg.SPLIT_FILE)["train"] if n in truths and not truths[n].problems()]
    embedder = Embedder(cfg.EMBEDDING_MODEL_PATH, cfg.EMBEDDING_DEVICE) if cfg.EMBEDDING_MODEL_PATH else None

    def examples():
        for name in names:
            try:
                yield load_chart(name), truths[name]
            except Exception as exc:  # noqa: BLE001
                print(f"  skipped {name}: {exc}")

    settings = V2Settings(window=args.window, top_k=args.top_k, max_block=args.max_block)
    scorer = train(examples(), settings, embedder)
    scorer.save(cfg.V2_MODEL_PATH)
    print(json.dumps(scorer.report, indent=2))
    print(f"features: {len(FEATURES)} -> {cfg.V2_MODEL_PATH}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
