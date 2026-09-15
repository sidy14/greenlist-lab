"""
differ.py — visual diff between original and cleaned text.

Two views:
  * unified line-level diff (like git diff)
  * character-level diff with invisible chars shown as U+XXXX
"""
from __future__ import annotations

import difflib

RED = "\033[31m"
GREEN = "\033[32m"
DIM = "\033[2m"
RESET = "\033[0m"


def _c(code: str, s: str) -> str:
    return f"{code}{s}{RESET}"


def _show_char(ch: str) -> str:
    if ch == " ":
        return "·"
    if ch == "\n":
        return "⏎"
    if ch == "\t":
        return "⇥"
    if ch.isprintable():
        return ch
    return f"‹U+{ord(ch):04X}›"


def _snippet(s: str, max_len: int = 60) -> str:
    out = "".join(_show_char(ch) for ch in s[:max_len])
    if len(s) > max_len:
        out += "…"
    return out or "∅"


def diff_lines(original: str, cleaned: str, color: bool = True) -> str:
    orig_lines = original.splitlines()
    clean_lines = cleaned.splitlines()
    d = difflib.unified_diff(
        orig_lines, clean_lines,
        fromfile="original", tofile="cleaned",
        lineterm="", n=2,
    )
    lines = []
    for line in d:
        if line.startswith("+++") or line.startswith("---"):
            lines.append(_c(DIM, line) if color else line)
        elif line.startswith("@@"):
            lines.append(_c(DIM, line) if color else line)
        elif line.startswith("+"):
            lines.append(_c(GREEN, line) if color else line)
        elif line.startswith("-"):
            lines.append(_c(RED, line) if color else line)
        else:
            lines.append(line)
    return "\n".join(lines) or _c(DIM, "  (no line-level changes)", ) if color \
           else ("  (no line-level changes)")


def diff_chars(original: str, cleaned: str, max_shown: int = 40,
               color: bool = True) -> str:
    sm = difflib.SequenceMatcher(a=original, b=cleaned, autojunk=False)
    lines = []
    shown = 0
    for tag, i1, i2, j1, j2 in sm.get_opcodes():
        if tag == "equal":
            continue
        if shown >= max_shown:
            lines.append(_c(DIM, f"  … (more changes truncated)") if color
                         else "  … (more changes truncated)")
            break
        shown += 1
        pos = i1
        if tag == "delete":
            lines.append(
                f"  @{pos:>5}  {_c(RED, '−')} {_snippet(original[i1:i2])}"
                if color else f"  @{pos}  -  {_snippet(original[i1:i2])}"
            )
        elif tag == "insert":
            lines.append(
                f"  @{pos:>5}  {_c(GREEN, '+')} {_snippet(cleaned[j1:j2])}"
                if color else f"  @{pos}  +  {_snippet(cleaned[j1:j2])}"
            )
        else:
            arrow = _c(GREEN, "→") if color else "->"
            minus = _c(RED, "−") if color else "-"
            lines.append(
                f"  @{pos:>5}  {minus} {_snippet(original[i1:i2])}  "
                f"{arrow}  {_snippet(cleaned[j1:j2])}"
            )
    return "\n".join(lines) or ("  (no character-level changes)")


def render_full(original: str, cleaned: str, color: bool = True) -> str:
    parts = []
    parts.append(_c(DIM, "── Line-level diff ──") if color else "── Line-level diff ──")
    parts.append(diff_lines(original, cleaned, color=color))
    parts.append("")
    parts.append(_c(DIM, "── Character-level changes ──") if color else "── Character-level changes ──")
    parts.append(diff_chars(original, cleaned, color=color))
    return "\n".join(parts)
