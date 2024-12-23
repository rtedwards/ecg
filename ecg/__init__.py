"""ECG library."""

from pathlib import Path

PACKAGE_ROOT = Path(__file__).resolve().parent  # ecg/ecg
PROJECT_ROOT = PACKAGE_ROOT.parent  # ecg
DATA_DIR = PROJECT_ROOT / "data"
