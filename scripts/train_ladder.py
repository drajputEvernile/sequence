"""Train one version of the experiment ladder (v3, v4, v5, v6). Run with the training shuffle
set's config active.

    $env:SEQ_CONFIG = "<repo>\\workspace\\shuffle_train\\config_shuffled.py"
    python scripts\\train_ladder.py --version v4
    python scripts\\train_ladder.py --version v6          # needs the seam model (train_seam.py) first
    python scripts\\train_ladder.py --version v5 --solver cover

Each version: the evidence model (no scan inputs, plus the version's extra evidence) and the move
margin tuned on 20% held-out training documents (shuffled ones measure repair, the ones left in
order measure harm). v6's evidence model fits only on the documents the seam model did not learn
from. Saved to config.LADDER_MODEL_DIR\\<version>\\model.joblib (v3: config.V3_MODEL_PATH); the
report (margin curve, anchors, candidate recall) is printed -- scores only, safe to bring back.
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
from seqlib.ladder import VERSIONS, load_seam, model_path, settings_for  # noqa: E402
from seqlib.settings import cfg  # noqa: E402
from seqlib.v3.model import V3Settings, train  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--version", required=True, choices=sorted(VERSIONS))
    ap.add_argument("--solver", choices=("search", "cover"), default=V3Settings.solver)
    ap.add_argument("--harm-floor", type=float, default=V3Settings.harm_floor)
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

    settings = settings_for(args.version, solver=args.solver, harm_floor=args.harm_floor,
                            top_k=args.top_k, window=args.window)
    seam, fit_exclude = None, set()
    if "seam" in settings.groups:
        seam = load_seam(required=True)
        fit_exclude = set(seam.meta.get("trained_on") or [])
        print(f"seam model: {cfg.SEAM_MODEL_PATH} (learned from {len(fit_exclude)} documents; "
              f"v6's evidence model fits on the others)")

    def examples():
        for name in names:
            try:
                yield load_chart(name), truths[name]
            except Exception as exc:  # noqa: BLE001
                print(f"  skipped {name}: {exc}")

    model = train(examples(), settings, controls, seam=seam, fit_exclude=fit_exclude)
    path = model_path(args.version)
    model.save(path)
    print(json.dumps(model.report, indent=2))
    print(f"-> {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
