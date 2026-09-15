"""
universal_analyzer.py - 4-layer AI-text detection (v2).

Layer 1: Vendor signatures (deterministic)
Layer 2: Surface artifacts (ZX, homoglyphs, etc.)
Layer 3: Stylometric (perplexity, burstiness, TTR)
Layer 4: Arabic-specific markers (when text is Arabic)

Final integrated score = weighted max of available layers.
"""
from __future__ import annotations
import json
from dataclasses import asdict, dataclass, field

from surface_detector import SurfaceDetector
from arabic_surface import ArabicSurfaceDetector
from arabic_detector import ArabicDetector
from vendor_signatures import VendorDetector
from normalizer import Normalizer
from arabic_normalizer import ArabicNormalizer


@dataclass
class UniversalReport:
    text_length: int
    surface: dict
    arabic_surface: dict
    arabic_detector: dict
    vendor: dict
    stylometric: dict
    final_ai_score: float
    final_verdict: str
    cleaned_text: str
    evidence: list[str]

    def to_json(self) -> str:
        return json.dumps(asdict(self), ensure_ascii=False, indent=2)

    def to_human(self) -> str:
        lines = []
        lines.append("=" * 70)
        lines.append("UNIVERSAL AI-TEXT ANALYSIS")
        lines.append("=" * 70)
        lines.append(f"length          : {self.text_length} chars")
        lines.append(f"AI likelihood   : {self.final_ai_score:.2%}")
        lines.append(f"verdict         : {self.final_verdict}")
        lines.append("")
        lines.append("-- Evidence --")
        for e in self.evidence or ["  (no strong signal)"]:
            lines.append(f"  * {e}")
        lines.append("")
        lines.append("-- Layer 1: Vendor signatures --")
        v = self.vendor
        if v["vendors"]:
            for vendor in v["vendors"]:
                lines.append(f"  [DETECTED] {vendor}")
        else:
            lines.append("  none")
        lines.append("")
        lines.append("-- Layer 2: Surface artifacts (Latin) --")
        lines.append(f"  total findings: {self.surface['total_findings']}")
        lines.append("")
        lines.append("-- Layer 3: Stylometric --")
        st = self.stylometric
        lines.append(f"  perplexity      : {st['perplexity']:.2f}")
        lines.append(f"  burstiness      : {st['burstiness']:.3f}")
        lines.append(f"  TTR             : {st['type_token_ratio']:.3f}")
        lines.append(f"  verdict         : {st['verdict']}")
        for note in st.get("notes", []):
            lines.append(f"    - {note}")
        lines.append("")
        lines.append("-- Layer 4: Arabic detector --")
        ad = self.arabic_detector
        if not ad["is_arabic"]:
            lines.append("  not applicable")
        else:
            lines.append(f"  score           : {ad['ai_score']:.2%}")
            lines.append(f"  verdict         : {ad['verdict']}")
            lines.append(f"  tashkeel density: {ad['tashkeel_density']:.2%}")
            lines.append(f"  opener diversity: {ad['connector_diversity']:.2f}")
            if ad["formal_phrases_found"]:
                lines.append(f"  formal phrases  : {len(ad['formal_phrases_found'])}")
                for p in ad["formal_phrases_found"][:5]:
                    lines.append(f"    - {p}")
        lines.append("")
        lines.append("=" * 70)
        return "\n".join(lines)


class UniversalAnalyzer:
    def __init__(self, run_stylometric: bool = True):
        self.surface = SurfaceDetector()
        self.arabic = ArabicSurfaceDetector()
        self.arabic_detector = ArabicDetector()
        self.vendor = VendorDetector()
        self.normalizer = Normalizer()
        self.ar_norm = ArabicNormalizer()
        self._stylometric = None
        self._run_stylometric = run_stylometric

    def _stylo(self):
        if self._stylometric is None:
            from stylometric_detector import StylometricDetector
            self._stylometric = StylometricDetector()
        return self._stylometric

    def analyze(self, text: str) -> UniversalReport:
        surf = self.surface.scan(text)
        ar = self.arabic.scan(text)
        vend = self.vendor.scan(text)
        ar_det = self.arabic_detector.detect(text)

        stylo_data = {
            "perplexity": 0.0, "burstiness": 0.0, "type_token_ratio": 0.0,
            "ai_score": 0.0, "verdict": "skipped", "notes": [],
        }
        if self._run_stylometric and len(text) > 100:
            r = self._stylo().detect(text)
            stylo_data = {
                "perplexity": r.perplexity,
                "burstiness": r.burstiness,
                "type_token_ratio": r.type_token_ratio,
                "ai_score": r.ai_score,
                "verdict": r.verdict,
                "notes": r.notes,
            }

        # -------- integrated score with evidence --------
        evidence: list[str] = []
        candidates: list[float] = []

        # Layer 1: vendor signature
        if vend.detected_vendors:
            candidates.append(1.0)
            for v in sorted(vend.detected_vendors):
                evidence.append(f"[L1] vendor signature detected: {v}")

        # Layer 2: heavy surface contamination
        surface_a = surf.counts.get("A", 0)
        if surface_a > 0:
            candidates.append(min(0.5 + 0.1 * surface_a, 0.9))
            evidence.append(f"[L2] {surface_a} hidden characters detected")

        # Layer 3: stylometric
        if stylo_data["ai_score"] > 0.4:
            candidates.append(stylo_data["ai_score"])
            evidence.append(
                f"[L3] stylometric score {stylo_data['ai_score']:.0%} "
                f"(PPL={stylo_data['perplexity']:.1f}, burst={stylo_data['burstiness']:.2f})"
            )

        # Layer 4: Arabic detector
        if ar_det.is_arabic and ar_det.ai_score > 0.3:
            candidates.append(ar_det.ai_score)
            evidence.append(
                f"[L4] Arabic AI score {ar_det.ai_score:.0%} "
                f"({len(ar_det.formal_phrases_found)} formal phrases)"
            )

        # Final = max of candidates (most confident layer wins)
        score = max(candidates) if candidates else 0.0

        if score >= 0.7:
            verdict = "likely AI-generated"
        elif score <= 0.3:
            verdict = "likely human-written"
        else:
            verdict = "uncertain"

        nr = self.normalizer.process(text)
        arn = self.ar_norm.process(nr.cleaned)

        return UniversalReport(
            text_length=len(text),
            surface={
                "total_findings": len(surf.findings),
                "counts": dict(surf.counts),
            },
            arabic_surface={
                "is_arabic": ar.is_arabic,
                "tashkeel_density": ar.tashkeel_density,
                "counts": dict(ar.counts),
            },
            arabic_detector={
                "is_arabic": ar_det.is_arabic,
                "ai_score": ar_det.ai_score,
                "verdict": ar_det.verdict,
                "tashkeel_density": ar_det.tashkeel_density,
                "connector_diversity": ar_det.connector_diversity,
                "formal_phrases_found": ar_det.formal_phrases_found,
            },
            vendor={
                "vendors": sorted(vend.detected_vendors),
                "findings": [
                    {"vendor": f.vendor, "marker": f.marker, "pos": f.position}
                    for f in vend.findings[:20]
                ],
            },
            stylometric=stylo_data,
            final_ai_score=score,
            final_verdict=verdict,
            cleaned_text=arn.cleaned,
            evidence=evidence,
        )