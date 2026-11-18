import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
os.environ["SEQ_CONFIG"] = str(ROOT / "tests" / "config_test.py")
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tests"))
