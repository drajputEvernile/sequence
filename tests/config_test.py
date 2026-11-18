"""Config for the test run: everything relative to the repo, nothing machine-specific."""
import os
import tempfile
from pathlib import Path

_REPO = Path(__file__).resolve().parents[1]
_TMP = Path(os.environ.get("SEQ_TEST_TMP") or tempfile.gettempdir()) / "seq_tests"

CHARTS_ROOT = _TMP / "charts"
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
GROUND_TRUTH_PATH = _TMP / "gt.csv"
SPLIT_FILE = _TMP / "split.json"
TEST_FRACTION = 0.25
SPLIT_SEED = 13
V2_MODEL_PATH = _TMP / "v2.joblib"
EMBEDDING_MODEL_PATH = None
EMBEDDING_DEVICE = "cpu"
V1_CANON_DIR = _REPO / "seqlib" / "v1" / "resources"
TESSERACT_CMD = Path(r"C:\Program Files\Tesseract-OCR\tesseract.exe")
TESSERACT_LANG = "eng"
PHOTO_INPUT_ROOT = _TMP / "photos"
PHOTO_OUTPUT_ROOT = _TMP / "photo_charts"
OUTPUT_ROOT = _TMP / "outputs"
# test_pipeline.py writes a Chart_Name / Id / Sequence sheet; test_flow.py passes its own column
GT_SEQUENCE_COLUMN = "Sequence"
RAW_DOCS_ROOT = _TMP / "flow" / "raw_documents"
OCR_DOCS_ROOT = _TMP / "flow" / "ocr"
DOC_KEY_REGEX = r"^(?P<RecordID>\d+)_(?P<DocumentID>\d+)"
DOC_NAME_PATTERN = "{RecordID}_{DocumentID}"
REPO = _REPO
PIPELINE_OUTPUT_CSV = _TMP / "flow" / "imaging_output_full.csv"
