"""
cli.py — command-line interface for ai-text-lab.

Usage:
    python3 -m src.cli analyze <file>
    python3 -m src.cli clean <file> -o <out>
    python3 -m src.cli analyze <file> --model openai-community/gpt2 --json
"""
from __future__ import annotations

import argparse
import json
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).parent))

from analyzer import Analyzer


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


def cmd_clean(args) -> int:
    text = _read(args.input)
    analyzer = Analyzer(model_id=None)
    report = analyzer.analyze(text)
    _write(args.output, report.cleaned_text)
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(
        prog="ai-text-lab",
        description="Detect and clean AI-text surface artifacts and "
                    "statistical watermarks.",
    )
    sub = ap.add_subparsers(dest="cmd", required=True)

    a = sub.add_parser("analyze", help="produce a full report")
    a.add_argument("input", help="input file or '-' for stdin")
    a.add_argument("-o", "--output", default=None,
                   help="write report to this file (default stdout)")
    a.add_argument("--json", action="store_true",
                   help="emit JSON instead of human-readable text")
    a.add_argument("--model", default=None,
                   help="tokenizer id for statistical detection, "
                        "e.g. openai-community/gpt2")
    a.add_argument("--ngram", type=int, default=5)
    a.add_argument("--no-greenlist", action="store_true")
    a.add_argument("--no-synthid", action="store_true")
    a.set_defaults(func=cmd_analyze)

    c = sub.add_parser("clean", help="emit only the cleaned text")
    c.add_argument("input", help="input file or '-' for stdin")
    c.add_argument("-o", "--output", default="-",
                   help="output file or '-' for stdout")
    c.set_defaults(func=cmd_clean)

    args = ap.parse_args()
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
