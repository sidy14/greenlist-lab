"""
vendor_signatures.py - detect vendor-specific surface fingerprints.

OpenAI o3/o4-mini: NNBSP (U+202F) inserted at specific positions.
Other vendors: HTML comments, model tokens, and provenance markers.
"""
from __future__ import annotations
import re
from dataclasses import dataclass, field


VENDOR_PATTERNS = {
    "OpenAI-o3": {
        "markers": [
            ("\u202f", "Narrow No-Break Space (U+202F)"),
            ("\u2009", "Thin Space (U+2009)"),
        ],
        "min_count": 2,
        "description": "OpenAI o3/o4-mini surface watermark",
    },
    "OpenAI-GPT4o": {
        "markers": [
            (re.compile(r"ImageGen|img_v\d+|dalle"), "ImageGen marker"),
        ],
        "min_count": 1,
        "description": "GPT-4o image generation marker",
    },
    "Anthropic-Claude": {
        "markers": [
            (re.compile(r"<antml:thinking>", re.I), "Claude thinking tag"),
            (re.compile(r"\[assistant\]|\[user\]"), "Chat wrapper"),
        ],
        "min_count": 1,
        "description": "Claude system prompt leakage",
    },
    "Google-Gemini": {
        "markers": [
            (re.compile(r"<ctrl\d+>", re.I), "Gemini control token"),
            (re.compile(r"gemini|bard", re.I), "Gemini self-reference"),
        ],
        "min_count": 1,
        "description": "Gemini control tokens",
    },
    "Meta-Llama": {
        "markers": [
            (re.compile(r"<\|start_header_id\|>"), "Llama 3 header"),
            (re.compile(r"<\|eot_id\|>|<\|end_header_id\|>"), "Llama 3 terminator"),
        ],
        "min_count": 1,
        "description": "Llama 3 chat template",
    },
}


@dataclass
class VendorFinding:
    vendor: str
    marker: str
    position: int
    sample: str
    description: str


@dataclass
class VendorReport:
    text_length: int
    findings: list[VendorFinding] = field(default_factory=list)
    detected_vendors: set[str] = field(default_factory=set)

    def summary(self) -> str:
        if not self.detected_vendors:
            return "no vendor signature detected"
        lines = [f"vendors detected: {sorted(self.detected_vendors)}"]
        for f in self.findings[:20]:
            lines.append(f"  [{f.vendor}] pos={f.position}: {f.marker}")
        return "\n".join(lines)


class VendorDetector:
    def scan(self, text: str) -> VendorReport:
        report = VendorReport(text_length=len(text))

        for vendor, spec in VENDOR_PATTERNS.items():
            hits = []
            for marker, label in spec["markers"]:
                if isinstance(marker, str):
                    for i, ch in enumerate(text):
                        if ch == marker:
                            hits.append((i, ch, label))
                else:
                    for m in marker.finditer(text):
                        hits.append((m.start(), m.group(), label))

            if len(hits) >= spec["min_count"]:
                report.detected_vendors.add(vendor)
                for pos, sample, label in hits[:50]:
                    report.findings.append(VendorFinding(
                        vendor=vendor,
                        marker=label,
                        position=pos,
                        sample=sample[:30],
                        description=spec["description"],
                    ))

        return report