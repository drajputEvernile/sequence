"""Order one chart and show why: the order, each page's method / confidence, and the truth.

    python scripts/order_chart.py <chart> --method v1
    python scripts/order_chart.py case_01 --root F:\\Data\\sequence\\photo_charts --method v1
    python scripts/order_chart.py <chart> --method v2 --truth      # side by side with ground truth
"""
import argparse

import _common  # noqa: F401

from seqlib.io.chart import load_chart
from seqlib.settings import cfg
from seqlib.v1.runner import run_scan_order, run_v1


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("chart")
    ap.add_argument("--root", default=None, help="chart folders root (default config.CHARTS_ROOT)")
    ap.add_argument("--method", choices=("scan", "v1", "v2"), default="v1")
    ap.add_argument("--truth", action="store_true", help="show the ground truth beside it")
    args = ap.parse_args()

    chart = load_chart(args.chart, args.root)
    if args.method == "v2":
        from seqlib.v2.model import PairScorer
        from seqlib.v2.runner import run_v2

        result = run_v2(chart, PairScorer.load(cfg.V2_MODEL_PATH))
    elif args.method == "v1":
        result = run_v1(chart)
    else:
        result = run_scan_order(chart)

    want = {}
    if args.truth:
        from seqlib.io.ground_truth import load_ground_truth

        truth = load_ground_truth(cfg.GROUND_TRUTH_PATH).get(chart.name)
        want = {p: i for i, p in enumerate(truth.main_order, start=1)} if truth else {}
    pages = chart.by_number()
    print(f"{chart.name}: {len(chart.pages)} pages, missing inputs: {', '.join(chart.missing) or 'none'}")
    print(f"{'seq':>4} {'page':>5} {'truth':>5}  detail / first words")
    for seq, number in enumerate(result.main_order, start=1):
        d = result.detail.get(number, {})
        mark = "" if not want else ("  " if want.get(number) == seq else "X ")
        words = " ".join(pages[number].text.split()[:10])
        print(f"{mark}{seq:>4} {number:>5} {want.get(number, ''):>5}  {d}  | {words}")
    for k, number in enumerate(sorted(result.junk_pages), start=1):
        print(f"  J{k:<3} {number:>5}  ({pages[number].group})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
