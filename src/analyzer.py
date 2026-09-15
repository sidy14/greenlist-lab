"""
analyzer.py — unified facade: surface + Arabic + statistical + normalization.
"""
from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from typing import Optional

from surface_detector import SurfaceDetector, SurfaceReport
from normalizer import Normalizer, NormalizeResult
from arabic_surface import ArabicSurfaceDetector, ArabicReport
from arabic_normalizer import ArabicNormalizer, ArabicNormalizeResult


@dataclass
class AnalysisReport:
    text_length: int
    surface: dict
    arabic: dict
    statistical: dict
    normalization: dict
    cleaned_text: str

    def to_json(self) -> str:
        return json.dumps(asdict(self), ensure_ascii=False, indent=2)

    def to_human(self) -> str:
        lines = []
        lines.append("=" * 66)
        lines.append("AI-TEXT-LAB ANALYSIS REPORT")
        lines.append("=" * 66)
        lines.append(f"input length         : {self.text_length} chars")
        lines.append("")

        lines.append("── Surface artifacts (Latin) ──")
        s = self.surface
        if s["total_findings"] == 0:
            lines.append("  none detected")
        else:
            for cat, label in [("A", "hidden chars"),
                               ("B", "homoglyphs"),
                               ("C", "structural")]:
                n = s["counts"].get(cat, 0)
                if n:
                    lines.append(f"  {label:20s}: {n}")
        lines.append("")

        lines.append("── Arabic-specific artifacts ──")
        a = self.arabic
        if not a.get("is_arabic"):
            lines.append("  text is not Arabic (or contains <15% Arabic chars)")
        elif not a.get("counts"):
            lines.append("  none detected")
        else:
            labels = {
                "AR1": "diacritics", "AR2": "digits",
                "AR3": "punctuation", "AR4": "alef variants",
                "AR5": "ya variants", "AR6": "presentation forms",
                "AR7": "isolated tashkeel", "AR8": "abnormal tashkeel density",
            }
            for cat, n in sorted(a["counts"].items()):
                lines.append(f"  {labels.get(cat, cat):24s}: {n}")
            lines.append(f"  tashkeel density       : {a['tashkeel_density']:.2%}")
        lines.append("")

        lines.append("── Statistical watermark ──")
        st = self.statistical
        if st.get("status") == "not_attempted":
            lines.append("  not attempted (no model specified)")
        elif st.get("status") == "error":
            lines.append(f"  error: {st.get('message')}")
        else:
            for key in ("greenlist", "synthid"):
                info = st.get(key)
                if info:
                    lines.append(f"  {key:20s}: z={info['z']:+.3f}  "
                                 f"pred={info['prediction']}")
        lines.append("")

        lines.append("── Normalization ──")
        n = self.normalization
        lines.append(f"  chars removed      : {n['removed_chars']}")
        lines.append(f"  runs collapsed     : {n['removed_runs']}")
        lines.append(f"  arabic replaced    : {n['arabic_replaced']}")
        lines.append(f"  net delta          : {n['net_delta']}")
        lines.append("")
        lines.append("=" * 66)
        return "\n".join(lines)


class Analyzer:
    def __init__(
        self,
        *,
        model_id: Optional[str] = None,
        run_greenlist: bool = True,
        run_synthid: bool = True,
        ngram_len: int = 5,
        arabic_alef: bool = False,
        arabic_ya: bool = False,
        arabic_strip_tashkeel: bool = False,
    ) -> None:
        self.model_id = model_id
        self.run_greenlist = run_greenlist
        self.run_synthid = run_synthid
        self.ngram_len = ngram_len

        self._surface = SurfaceDetector()
        self._normalizer = Normalizer()
        self._arabic_surface = ArabicSurfaceDetector()
        self._arabic_normalizer = ArabicNormalizer(
            alef=arabic_alef,
            ya=arabic_ya,
            tashkeel_strip=arabic_strip_tashkeel,
        )
        self._green = None
        self._synth = None

    def _lazy_load_statistical(self):
        if self.model_id is None:
            return
        if self._green is None and self.run_greenlist:
            from greenlist_detector import GreenListDetector
            self._green = GreenListDetector(self.model_id)
        if self._synth is None and self.run_synthid:
            from synthid_detector import SynthIDDetector
            self._synth = SynthIDDetector(self.model_id, ngram_len=self.ngram_len)

    def analyze(self, text: str) -> AnalysisReport:
        sr: SurfaceReport = self._surface.scan(text)
        nr: NormalizeResult = self._normalizer.process(text)

        ar_report: ArabicReport = self._arabic_surface.scan(text)
        ar_result: ArabicNormalizeResult = self._arabic_normalizer.process(nr.cleaned)

        st_data = {"status": "not_attempted"}
        if self.model_id is not None:
            try:
                self._lazy_load_statistical()
                st_data = {"status": "ok"}
                if self._green is not None:
                    r = self._green.detect_from_text(text)
                    st_data["greenlist"] = {
                        "z": r.z_score,
                        "prediction": bool(r.prediction),
                        "green_fraction": r.green_fraction,
                        "tokens": r.num_tokens,
                    }
                if self._synth is not None:
                    r = self._synth.detect_from_text(text)
                    st_data["synthid"] = {
                        "z": r.z_score,
                        "prediction": bool(r.prediction),
                        "mean_g": r.mean_g,
                        "tokens": r.num_tokens,
                    }
            except Exception as exc:
                st_data = {"status": "error", "message": repr(exc)}

        return AnalysisReport(
            text_length=len(text),
            surface={
                "total_findings": len(sr.findings),
                "counts": dict(sr.counts),
                "findings": [f.as_dict() for f in sr.findings[:200]],
            },
            arabic={
                "is_arabic": ar_report.is_arabic,
                "arabic_ratio": ar_report.arabic_ratio,
                "tashkeel_density": ar_report.tashkeel_density,
                "counts": dict(ar_report.counts),
                "findings": [f.as_dict() for f in ar_report.findings[:200]],
            },
            statistical=st_data,
            normalization={
                "removed_chars": nr.removed_chars + ar_result.removed_chars,
                "removed_runs": nr.removed_runs + ar_result.runs,
                "arabic_replaced": ar_result.replaced_chars,
                "net_delta": len(text) - len(ar_result.cleaned),
                "changed": text != ar_result.cleaned,
            },
            cleaned_text=ar_result.cleaned,
        )
