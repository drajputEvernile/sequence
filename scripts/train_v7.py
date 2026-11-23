"""Train v7 / v7s / v8, the gated versions (seqlib/v7). Run with the training shuffle set's config active.

    v7   v5's evidence + the gate
    v7s  v6's evidence (seam) + the gate
    v8   everything: v6's evidence + v1's rules as evidence + the gate + both solvers

    $env:SEQ_CONFIG = "<repo>\\workspace\\shuffle_train\\config_shuffled.py"
    python scripts\\train_v7.py --version v7
    python scripts\\train_v7.py --version v7s        # needs the seam model (train_seam.py)
    python scripts\\train_v7.py --version v8         # needs the seam model; the slowest

Cross-fitting: every training document is scored by an evidence model that did not learn from
it, decoded at every margin, and described by the gate's document-level signs. The gate
(P(document out of order)), its threshold and the two margins are tuned on those scores; the
final evidence model then fits on every document. Saved to config.LADDER_MODEL_DIR\\<version>\\;
the report is printed -- scores only, safe to bring back.
"""
import argparse
import json
import logging

import _common  # noqa: F401

logging.getLogger("seqlib.v1").setLevel(logging.WARNING)

from seqlib.eval.split import read_split  # noqa: E402
from seqlib.io.chart import load_chart  # noqa: E402
from seqlib.io.ground_truth import load_ground_truth  # noqa: E402
from seqlib.ladder import load_seam  # noqa: E402
from seqlib.settings import cfg  # noqa: E402
from seqlib.v3.model import V3Settings  # noqa: E402
from seqlib.v7 import VERSIONS, model_path, settings_for  # noqa: E402
from seqlib.v7.model import train_v7  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--version", required=True, choices=sorted(VERSIONS))
    ap.add_argument("--solver", choices=("search", "cover", "best"), default=None,
                    help="default: v7 / v7s search, v8 best (both, the cheaper order)")
    ap.add_argument("--harm-floor", type=float, default=V3Settings.harm_floor)
    ap.add_argument("--folds", type=int, default=5)
    args = ap.parse_args()

    truths = load_ground_truth(cfg.GROUND_TRUTH_PATH)
    names = [n for n in read_split(cfg.SPLIT_FILE)["train"] if n in truths and not truths[n].problems()]
    if not names:
        raise SystemExit("no training document: set SEQ_CONFIG to a shuffle set built with --role train")
    overrides = {"harm_floor": args.harm_floor, "folds": args.folds}
    if args.solver:
        overrides["solver"] = args.solver
    settings = settings_for(args.version, **overrides)
    seam, fit_exclude = None, set()
    if "seam" in settings.groups:
        seam = load_seam(required=True)
        fit_exclude = set(seam.meta.get("trained_on") or [])
        print(f"seam model: {cfg.SEAM_MODEL_PATH} (learned from {len(fit_exclude)} documents; "
              f"{args.version} fits and tunes on the others)")

    def examples():
        for name in names:
            try:
                yield load_chart(name), truths[name]
            except Exception as exc:  # noqa: BLE001
                print(f"  skipped {name}: {exc}")

    model = train_v7(examples(), settings, seam=seam, fit_exclude=fit_exclude, folds=args.folds)
    path = model_path(args.version)
    model.save(path)
    print(json.dumps(model.report, indent=2))
    print(f"-> {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
