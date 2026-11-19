"""Train v3 (evidence model with no scan inputs + a tuned move margin). Run with the training
shuffle set's config active.

    $env:SEQ_CONFIG = "<repo>\\workspace\\shuffle_train\\config_shuffled.py"
    python scripts\\train_v3.py
    python scripts\\train_v3.py --solver cover          # the design doc's path cover
    python scripts\\train_v3.py --harm-floor 0.995       # allow a little more movement

20% of the training documents are held out to tune the margin: the shuffled ones measure repair,
the ones the shuffle left in order measure harm. The model is saved to config.V3_MODEL_PATH (with
the solver and the margin) and the margin curve is printed -- scores only, safe to bring back.
"""
import argparse
import json
import logging
from pathlib import Path

import _common  # noqa: F401

logging.getLogger("seqlib.v1").setLevel(logging.WARNING)

from seqlib.eval.split import read_split  # noqa: E402
from seqlib.io.chart import load_chart  # noqa: E402
from seqlib.io.ground_truth import load_ground_truth  # noqa: E402
from seqlib.settings import cfg  # noqa: E402
from seqlib.v3.model import V3Settings, train  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--solver", choices=("search", "cover"), default=V3Settings.solver)
    ap.add_argument("--harm-floor", type=float, default=V3Settings.harm_floor,
                    help="held-out in-order documents must keep at least this successor accuracy")
    ap.add_argument("--top-k", type=int, default=V3Settings.top_k)
    ap.add_argument("--window", type=int, default=V3Settings.window)
    args = ap.parse_args()

    truths = load_ground_truth(cfg.GROUND_TRUTH_PATH)
    names = [n for n in read_split(cfg.SPLIT_FILE)["train"] if n in truths and not truths[n].problems()]
    if not names:
        raise SystemExit("no training document: set SEQ_CONFIG to a shuffle set built with --role train")
    controls = None
    manifest = Path(cfg.GROUND_TRUTH_PATH).parent / "shuffle_manifest.json"
    if manifest.is_file():
        docs = json.loads(manifest.read_text(encoding="utf-8"))["documents"]
        controls = {d["doc"] for d in docs if d["operations"] == ["unchanged (control)"]}

    def examples():
        for name in names:
            try:
                yield load_chart(name), truths[name]
            except Exception as exc:  # noqa: BLE001
                print(f"  skipped {name}: {exc}")

    settings = V3Settings(solver=args.solver, harm_floor=args.harm_floor, top_k=args.top_k, window=args.window)
    model = train(examples(), settings, controls)
    model.save(cfg.V3_MODEL_PATH)
    print(json.dumps(model.report, indent=2))
    print(f"-> {cfg.V3_MODEL_PATH}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
