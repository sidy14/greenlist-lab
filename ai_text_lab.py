"""
ai_text_lab — public Python API.

Usage:
    from ai_text_lab import analyze, clean, diff
    report = analyze(open("f.txt").read(), model_id="openai-community/gpt2")
    print(report.to_human())
"""
import sys
from pathlib import Path

_SRC = Path(__file__).resolve().parent / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from analyzer import Analyzer, AnalysisReport
from surface_detector import SurfaceDetector, SurfaceReport, Finding
from normalizer import Normalizer, NormalizeResult
from differ import render_full, diff_lines, diff_chars

__all__ = [
    "Analyzer", "AnalysisReport",
    "SurfaceDetector", "SurfaceReport", "Finding",
    "Normalizer", "NormalizeResult",
    "analyze", "clean", "diff",
    "render_full", "diff_lines", "diff_chars",
]


def analyze(text: str, model_id: str | None = None, **kwargs) -> AnalysisReport:
    """Run the full pipeline and return a report object."""
    return Analyzer(model_id=model_id, **kwargs).analyze(text)


def clean(text: str) -> str:
    """Return only the sanitized text (no report)."""
    return Analyzer(model_id=None).analyze(text).cleaned_text


def diff(original: str, cleaned: str | None = None, *, mode: str = "both") -> str:
    """Return a textual diff between original and cleaned (auto-clean if None)."""
    if cleaned is None:
        cleaned = clean(original)
    if mode == "lines":
        return diff_lines(original, cleaned, color=False)
    if mode == "chars":
        return diff_chars(original, cleaned, color=False)
    return render_full(original, cleaned, color=False)
