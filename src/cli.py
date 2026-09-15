"""
cli.py - command-line interface for ai-text-lab.

Usage:
    ai-text-lab analyze <file> [--model gpt2] [--json]
    ai-text-lab clean   <file> -o <out>
    ai-text-lab diff    <file> [--mode lines|chars|both]
    ai-text-lab detect  <file> [--json] [--fast]
"""
from __future__ import annotations

import argparse
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).parent))

from analyzer import Analyzer
from differ import render_full, diff_lines, diff_chars


def _read(path: str) -> str:
    if path == "-":
        return sys.stdin.read()
    return pathlib.Path(path).read_text(encoding="utf-8")


def _write(path: str, text: str) -> None:
    if path == "-":
        sys.stdout.write(text)
        return
    pathlib.Path(path).write_text(text, encoding="utf-8")


def cmd_analyze(args) -> int:
    text = _read(args.input)
    analyzer = Analyzer(
        model_id=args.model,
        run_greenlist=not args.no_greenlist,
        run_synthid=not args.no_synthid,
        ngram_len=args.ngram,
    )
    report = analyzer.analyze(text)
    out = report.to_json() if args.json else report.to_human()
    if args.output:
        _write(args.output, out)
    else:
        sys.stdout.write(out)
        if not out.endswith("\n"):
            sys.stdout.write("\n")
    return 0


def cmd_clean(args) -> int:
    text = _read(args.input)
    report = Analyzer(model_id=None).analyze(text)
    _write(args.output, report.cleaned_text)
    return 0


def cmd_diff(args) -> int:
    text = _read(args.input)
    report = Analyzer(model_id=None).analyze(text)
    cleaned = report.cleaned_text
    color = sys.stdout.isatty() and not args.no_color

    if args.mode == "lines":
        out = diff_lines(text, cleaned, color=color)
    elif args.mode == "chars":
        out = diff_chars(text, cleaned, color=color)
    else:
        out = render_full(text, cleaned, color=color)

    if args.output:
        _write(args.output, out)
    else:
        sys.stdout.write(out)
        if not out.endswith("\n"):
            sys.stdout.write("\n")
    return 0


def cmd_detect(args) -> int:
    """Four-layer universal AI-text detection."""
    text = _read(args.input)

    from universal_analyzer import UniversalAnalyzer
    analyzer = UniversalAnalyzer(run_stylometric=not args.fast)
    report = analyzer.analyze(text)

    if args.json:
        out = report.to_json()
    else:
        out = report.to_human()

    if args.output:
        _write(args.output, out)
    else:
        sys.stdout.write(out)
        if not out.endswith("\n"):
            sys.stdout.write("\n")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(prog="ai-text-lab")
    sub = ap.add_subparsers(dest="cmd", required=True)

    a = sub.add_parser("analyze", help="produce a full report (surface + optional statistical)")
    a.add_argument("input")
    a.add_argument("-o", "--output", default=None)
    a.add_argument("--json", action="store_true")
    a.add_argument("--model", default=None)
    a.add_argument("--ngram", type=int, default=5)
    a.add_argument("--no-greenlist", action="store_true")
    a.add_argument("--no-synthid", action="store_true")
    a.set_defaults(func=cmd_analyze)

    c = sub.add_parser("clean", help="emit only the cleaned text")
    c.add_argument("input")
    c.add_argument("-o", "--output", default="-")
    c.set_defaults(func=cmd_clean)

    d = sub.add_parser("diff", help="show what would change")
    d.add_argument("input")
    d.add_argument("-o", "--output", default=None)
    d.add_argument("--mode", choices=["lines", "chars", "both"], default="both")
    d.add_argument("--no-color", action="store_true")
    d.set_defaults(func=cmd_diff)

    t = sub.add_parser(
        "detect",
        help="4-layer universal AI-text detection (vendor + surface + stylometric + Arabic)",
    )
    t.add_argument("input")
    t.add_argument("-o", "--output", default=None)
    t.add_argument("--json", action="store_true")
    t.add_argument("--fast", action="store_true",
                   help="skip stylometric layer (no torch needed)")
    t.set_defaults(func=cmd_detect)

    args = ap.parse_args()
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())