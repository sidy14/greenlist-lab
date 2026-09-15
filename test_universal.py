import sys
sys.path.insert(0, "src")
from universal_analyzer import UniversalAnalyzer

analyzer = UniversalAnalyzer(run_stylometric=False)

tests = {
    "o3-signature": "The amount is\u202f100\u202fdollars and the date is\u202f2026.",
    "arabic-ai": (
        "\u064a\u062c\u062f\u0631 \u0628\u0627\u0644\u0630\u0643\u0631 "
        "\u0623\u0646 \u0647\u0630\u0627 \u0627\u0644\u0645\u0648\u0636\u0648\u0639 "
        "\u0645\u0647\u0645. "
        "\u0645\u0646 \u0627\u0644\u062c\u062f\u064a\u0631 "
        "\u0628\u0627\u0644\u0630\u0643\u0631 \u0623\u0646 "
        "\u0627\u0644\u0641\u0643\u0631\u0629 \u0631\u0627\u0626\u0639\u0629. "
        "\u0641\u064a \u0627\u0644\u062e\u062a\u0627\u0645 \u0646\u0624\u0643\u062f "
        "\u0630\u0644\u0643."
    ),
    "arabic-clean": "\u0627\u0644\u0634\u0645\u0633 \u062a\u0634\u0631\u0642 "
                    "\u0645\u0646 \u0627\u0644\u0634\u0631\u0642.",
}

for name, text in tests.items():
    print(f"\n### {name} ###")
    r = analyzer.analyze(text)
    print(f"  score  : {r.final_ai_score:.0%}")
    print(f"  verdict: {r.final_verdict}")
    if r.evidence:
        for e in r.evidence:
            print(f"    * {e}")