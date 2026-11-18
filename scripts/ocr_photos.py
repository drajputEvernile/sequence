"""OCR photographed pages locally (Tesseract) into a chart folder the sequencers can read.

Put each document's photos in its own folder under config.PHOTO_INPUT_ROOT, named so they sort
in the order the pages were scanned (IMG_001.jpg, IMG_002.jpg ... or 1.jpg, 2.jpg ...):

    <PHOTO_INPUT_ROOT>/case_01/IMG_001.jpg ...

    python scripts/ocr_photos.py                 # every folder under PHOTO_INPUT_ROOT
    python scripts/ocr_photos.py case_01         # one folder

Output: <PHOTO_OUTPUT_ROOT>/<folder>/ (pages/, ocr/). Then, to see what a method does with it:

    python scripts/order_chart.py case_01 --root <PHOTO_OUTPUT_ROOT> --method v1
"""
import argparse

import _common  # noqa: F401

from seqlib.local_ocr.tesseract_ocr import ocr_chart, photos_in
from seqlib.settings import cfg


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("folders", nargs="*")
    args = ap.parse_args()
    root = cfg.PHOTO_INPUT_ROOT
    folders = [root / f for f in args.folders] if args.folders else sorted(p for p in root.iterdir() if p.is_dir())
    for folder in folders:
        photos = photos_in(folder)
        if not photos:
            print(f"{folder.name}: no images")
            continue
        out = ocr_chart(photos, folder.name, cfg.PHOTO_OUTPUT_ROOT, cfg.TESSERACT_CMD, cfg.TESSERACT_LANG)
        print(f"{folder.name}: {len(photos)} pages -> {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
