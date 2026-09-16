"""gauge.py - SVG circular gauge for the UI."""
from __future__ import annotations


def render_gauge(score: float, label: str = "AI", size: int = 220) -> str:
    """Return HTML+SVG for a colored circular gauge. score in [0, 1]."""
    pct = max(0.0, min(1.0, score)) * 100
    radius = 80
    circumference = 2 * 3.14159265 * radius
    dash = circumference * (1 - pct / 100)
    color = _color(pct)
    return f'''
<div style="display:flex; flex-direction:column; align-items:center; justify-content:center;">
  <svg width="{size}" height="{size}" viewBox="0 0 200 200">
    <circle cx="100" cy="100" r="{radius}" fill="none"
            stroke="#e6e6e6" stroke-width="16"/>
    <circle cx="100" cy="100" r="{radius}" fill="none"
            stroke="{color}" stroke-width="16"
            stroke-dasharray="{circumference:.1f}"
            stroke-dashoffset="{dash:.1f}"
            stroke-linecap="round"
            transform="rotate(-90 100 100)"/>
    <text x="100" y="95" text-anchor="middle"
          font-size="42" font-weight="bold" fill="{color}">{pct:.0f}%</text>
    <text x="100" y="125" text-anchor="middle"
          font-size="16" fill="#666">{label}</text>
  </svg>
</div>
'''


def _color(pct: float) -> str:
    if pct >= 70:
        return "#e74c3c"   # red
    if pct >= 40:
        return "#f39c12"   # orange
    if pct >= 15:
        return "#f1c40f"   # yellow
    return "#27ae60"       # green


def highlight_sentences(sentences, rtl: bool = True) -> str:
    """Render sentences with colored backgrounds. sentences: list of objects
    with .text, .score, .verdict."""
    direction = "rtl" if rtl else "ltr"
    align = "right" if rtl else "left"
    parts = []
    for s in sentences:
        color = _bg(s.score)
        text = _escape(s.text)
        parts.append(
            f'<span style="background:{color}; border-radius:4px;'
            f' padding:2px 4px; margin:1px 0; display:inline;'
            f' direction:{direction}; text-align:{align};">{text}</span> '
        )
    return (
        f'<div style="line-height:2.2; direction:{direction};'
        f' text-align:{align}; font-size:16px;">{"".join(parts)}</div>'
    )


def _bg(score: float) -> str:
    if score >= 0.7:
        return "rgba(231, 76, 60, 0.20)"      # red
    if score >= 0.4:
        return "rgba(243, 156, 18, 0.20)"     # orange
    if score >= 0.15:
        return "rgba(241, 196, 15, 0.20)"     # yellow
    return "rgba(39, 174, 96, 0.15)"          # green


def _escape(s: str) -> str:
    return (s.replace("&", "&amp;").replace("<", "&lt;")
             .replace(">", "&gt;"))