"""Train a sequencing version (seqlib/versions.py). Run with the training shuffle set's config active.

    $env:SEQ_CONFIG = "<repo>\\workspace\\shuffle_train\\config_shuffled.py"
    python scripts\\train.py --version v9
    python scripts\\train.py --version v5
    python scripts\\train.py --version v8          # v8 and v9 need the seam model (train_seam.py)

Cross-fitting: every training document is scored by an evidence model that did not learn from
it and decoded at every margin; the margin (v5) or the gate and its two margins (v8, v9) are
tuned on those scores -- v9 on position accuracy, the others on successor accuracy -- then the
final evidence model fits on every document. Saved to config.LADDER_MODEL_DIR\\<version>\\; the
report is printed -- scores only, safe to bring back.
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
from seqlib.ladder import load_seam  # noqa: E402
from seqlib.settings import cfg  # noqa: E402
from seqlib.versions import VERSIONS, is_gated, model_path, settings_for  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--version", required=True, choices=sorted(VERSIONS))
    ap.add_argument("--harm-floor", type=float, default=None, help="default: the version's")
    ap.add_argument("--folds", type=int, default=5)
    args = ap.parse_args()

    truths = load_ground_truth(cfg.GROUND_TRUTH_PATH)
    names = [n for n in read_split(cfg.SPLIT_FILE)["train"] if n in truths and not truths[n].problems()]
    if not names:
        raise SystemExit("no training document: set SEQ_CONFIG to a shuffle set built with --role train")
    overrides = {"folds": args.folds}
    if args.harm_floor is not None:
        overrides["harm_floor"] = args.harm_floor
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

    if is_gated(args.version):
        from seqlib.v7.model import train_v7

        model = train_v7(examples(), settings, seam=seam, fit_exclude=fit_exclude, folds=args.folds)
    else:
        from seqlib.v3.model import train

        controls = None
        manifest = Path(cfg.GROUND_TRUTH_PATH).parent / "shuffle_manifest.json"
        if manifest.is_file():
            docs = json.loads(manifest.read_text(encoding="utf-8"))["documents"]
            controls = {d["doc"] for d in docs if d["operations"] == ["unchanged (control)"]}
        model = train(examples(), settings, controls, seam=seam, fit_exclude=fit_exclude)
    path = model_path(args.version)
    model.save(path)
    print(json.dumps(model.report, indent=2))
    print(f"-> {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
