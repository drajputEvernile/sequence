"""Fine-tune the seam cross-encoder (for v8 / v9; CPU only). Run with the training shuffle set's config active.

Needs ``pip install -r requirements-seam.txt`` and a local copy of a small pre-trained model in
config.SEAM_BASE_MODEL_PATH (for example sentence-transformers/all-MiniLM-L6-v2, Apache-2.0:
download its folder once and copy it there; nothing is downloaded by this script).

    $env:SEQ_CONFIG = "<repo>\\workspace\\shuffle_train\\config_shuffled.py"
    python scripts\\train_seam.py                    # 3 epochs, lr 5e-5
    python scripts\\train_seam.py --resume --epochs 2  # 2 more epochs on the saved seam

Learns from half of the training documents (a fixed half by name, all of it); the v8 / v9 evidence
model fits on the other half, and a sample of that half is scored after every epoch (held-out AUC,
also against each kind of wrong page). The best epoch is the one saved to config.SEAM_MODEL_PATH.
If the AUC is still rising at the last epoch, --resume goes on from there.

Wrong pages it learns from: the wrong scan neighbour and two random pages of the same document
(seqlib.ladder.seam.TRAIN_NEGATIVES; not the most similar-looking page).
"""
import argparse
import json
import logging
import random

import _common  # noqa: F401

logging.getLogger("seqlib.v1").setLevel(logging.WARNING)

from seqlib.eval.split import read_split  # noqa: E402
from seqlib.io.chart import load_chart  # noqa: E402
from seqlib.io.ground_truth import load_ground_truth  # noqa: E402
from seqlib.ladder.seam import seam_half, train_seam  # noqa: E402
from seqlib.settings import cfg  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--epochs", type=int, default=3)
    ap.add_argument("--lr", type=float, default=5e-5)
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--max-length", type=int, default=256)
    ap.add_argument("--held-out-docs", type=int, default=50,
                    help="documents of the other half scored after every epoch (0 = none)")
    ap.add_argument("--resume", action="store_true", help="go on from the saved seam for --epochs more epochs")
    args = ap.parse_args()
    base = getattr(cfg, "SEAM_BASE_MODEL_PATH", None)
    if not args.resume and (base is None or not base.is_dir()):
        raise SystemExit(f"set SEAM_BASE_MODEL_PATH in config.py to a local model folder (now {base})")
    if args.resume and not (cfg.SEAM_MODEL_PATH / "config.json").is_file():
        raise SystemExit(f"--resume: no seam model at {cfg.SEAM_MODEL_PATH}")
    truths = load_ground_truth(cfg.GROUND_TRUTH_PATH)
    names = [n for n in read_split(cfg.SPLIT_FILE)["train"] if n in truths and not truths[n].problems()]
    half = sorted(seam_half(names))
    other = sorted(set(names) - set(half))
    held = sorted(random.Random(0).sample(other, min(args.held_out_docs, len(other))))
    print(f"seam: learning from {len(half)} of {len(names)} training documents (CPU); "
          f"scoring {len(held)} of the other {len(other)} after every epoch")
    report = train_seam(base, cfg.SEAM_MODEL_PATH, ((load_chart(n), truths[n]) for n in half),
                        epochs=args.epochs, lr=args.lr, batch=args.batch, max_length=args.max_length,
                        held_out=[(load_chart(n), truths[n]) for n in held], resume=args.resume)
    print(json.dumps(report, indent=2))
    print(f"-> {cfg.SEAM_MODEL_PATH}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
