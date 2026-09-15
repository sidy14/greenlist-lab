"""Project-root conftest.py.

Ensures `src/` is importable so that tests can do
    from src.analyzer import Analyzer
while the modules inside `src/` keep using flat imports like
    from surface_detector import SurfaceDetector
(which is how the CLI already imports them).
"""
import pathlib
import sys

_SRC = pathlib.Path(__file__).resolve().parent / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))
