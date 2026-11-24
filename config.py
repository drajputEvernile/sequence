"""Every setting and path this repo uses. Edit this file on each machine.

Nothing here calls Azure, a database, an OCR service or the imaging pipeline. The repo works
from four inputs already on disk:

    1. raw page images            one folder per document, named RecordID_DocumentID
    2. their OCR                  one folder (or file) per document, same names
    3. the full pipeline outputs  one CSV, one row per page (the imaging_final table export)
    4. the ground truth           one CSV, one row per page (the page_ground_truth table export)

Only documents present in all four are used (scripts/inventory.py lists them); the rest are
skipped until their files arrive.

Paths can be any absolute path. ``REPO`` (this repo's folder) is used for the defaults that
belong inside the repo (workspace/, models/, outputs/). To keep one config per machine without
editing this file, copy it and point SEQ_CONFIG at the copy:

    $env:SEQ_CONFIG = "D:\\sequence_cfg\\config_trainbox.py"    (PowerShell)
    export SEQ_CONFIG=/data/sequence/config_trainbox.py          (Linux)

Order of work (README.md, steps.txt):
    check_inputs.py -> inventory.py -> import_ocr.py -> make_split.py -> make_shuffle_test.py
    -> train.py (on the training shuffle set) -> evaluate.py -> link_report.py / shuffle_report.py
"""
from pathlib import Path

REPO = Path(__file__).resolve().parent

# =============================================================================================
# 1. The four inputs (read only)
# =============================================================================================
RAW_DOCS_ROOT = Path(r"F:\Data\sequence\raw_documents")   # <RAW_DOCS_ROOT>/<RecordID_DocumentID>/1.png ...
OCR_DOCS_ROOT = Path(r"F:\Data\sequence\ocr")             # <OCR_DOCS_ROOT>/<RecordID_DocumentID>/ocr/...

# The folder inside each document folder that holds what is wanted ("" = the whole document
# folder). The OCR comes laid out like the pipeline's workspace, one folder per document with
# ocr/, imaging/, staging/ ... inside it: only <document>/ocr/ is read.
RAW_SUBFOLDER = ""
OCR_SUBFOLDER = "ocr"

PIPELINE_OUTPUT_CSV = Path(r"F:\Data\sequence\imaging_output_full.csv")
GROUND_TRUTH_PATH = Path(r"F:\Data\sequence\ground_truth.csv")

# How a document is named: folder / file names (regex with these named groups). The pipeline CSV
# and the truth CSV name it in their chart_name column, the same text (70082444_56511414).
DOC_KEY_REGEX = r"^(?P<RecordID>\d+)_(?P<DocumentID>\d+)"
# Only for a truth sheet without chart_name (the client export with RecordID / DocumentID).
DOC_NAME_PATTERN = "{RecordID}_{DocumentID}"

# The truth column holding the correct order: page_sequence in the page_ground_truth export
# (Main pages 1..N, blank / junk / invoice pages J1, J2 ...). For the client export: "Sequence".
GT_SEQUENCE_COLUMN = "page_sequence"
# Blank / Junk / Invoice pages are numbered in their own series (1, 2, 3 beside the Main pages'
# 1, 2, 3); True moves them to the J series (J1, J2 ...), as the pipeline's loader does. Leave True.
GT_APPLY_JUNK_FLAGS = True

# =============================================================================================
# 2. Workspace (written by scripts/import_ocr.py)
# =============================================================================================
# import_ocr.py converts each usable document's OCR, whatever its format, into
#   <CHARTS_ROOT>/<doc>/ocr/<doc>_final2.json     (the pipeline's final2 shape: text + line boxes)
# and records which raw image each page is. Everything else is read from the pipeline CSV.
CHARTS_ROOT = REPO / "workspace" / "charts"

# File layout inside a chart folder ("{chart}" = folder name). Leave as is. The imaging/ CSVs
# exist only in a chart folder the imaging pipeline wrote itself; with PIPELINE_OUTPUT_CSV set
# they are not needed.
INPUT_FILES = {
    "ocr_prelim": "ocr/{chart}_prelim.txt",
    "ocr_final1": "ocr/{chart}_final1.json",
    "ocr_final2": "ocr/{chart}_final2.json",
    "junk": "imaging/{chart}_junk.csv",
    "classification": "imaging/{chart}_codeable.csv",
    "dos": "imaging/{chart}_dos.csv",
    "page_details": "imaging/{chart}_additional_page_details.csv",
    "signature": "imaging/{chart}_provider_signature.csv",
    "quality": "imaging/{chart}_quality.csv",
    "hw_printed": "imaging/{chart}_hw_printed.csv",
    "continuity": "imaging/{chart}_continuity.csv",
    "sequencing": "imaging/{chart}_sequencing.csv",
    "final": "imaging/{chart}_final.csv",
}
PAGES_SUBDIR = "pages"

# =============================================================================================
# 3. Train / test split
# =============================================================================================
# Written once by scripts/make_split.py over the usable documents; every later run reads it.
SPLIT_FILE = REPO / "data" / "splits" / "split.json"
TEST_FRACTION = 0.20
SPLIT_SEED = 13

# =============================================================================================
# 4. Sequencing models
# =============================================================================================
# The versions (seqlib/versions.py: v5, v8, v9; scripts/train.py): LADDER_MODEL_DIR\<version>\model.joblib.
LADDER_MODEL_DIR = REPO / "models"
# The seam cross-encoder of v8 / v9 (scripts/train_seam.py). SEAM_BASE_MODEL_PATH: a local copy of a small
# pre-trained model to fine-tune: sentence-transformers/all-MiniLM-L6-v2 (Apache-2.0), downloaded once.
SEAM_BASE_MODEL_PATH = REPO / "models" / "base" / "all-MiniLM-L6-v2"   # downloaded once (steps.txt)
SEAM_MODEL_PATH = REPO / "models" / "seam"
# v1's rule files (copied from the pipeline's core-pipeline/resources/canon).
V1_CANON_DIR = REPO / "seqlib" / "v1" / "resources"

# =============================================================================================
# 5. Outputs
# =============================================================================================
OUTPUT_ROOT = REPO / "outputs"
