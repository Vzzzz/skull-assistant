"""Tests run from the repo root (pytest.ini); core/ is imported as a package from there."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
