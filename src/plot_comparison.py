"""plot_comparison.py — side-by-side comparison of the two watermarks."""
from __future__ import annotations
import pathlib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

R = pathlib.Path(__file__).resolve().parent.parent / "data" / "reports"


def comparison_surface():
    gl = pd.read_csv(R / "robustness.csv")
    sy = pd.read_csv(R / "synthid_robustness.csv")
    gl_m = gl.groupby("transform")["z"].mean()
    sy_m = sy.groupby("transform")["z"].mean()

    order = ["identity", "whisper_light", "lowercase", "punct_strip",
             "word_drop_10", "word_shuffle_10"]
    gl_v = [gl_m[t] for t in order]
    sy_v = [sy_m[t] for t in order]

    x = np.arange(len(order))
    w = 0.38
    fig, ax = plt.subplots(figsize=(11, 5))
    b1 = ax.bar(x - w/2, gl_v, w, label="Green-List", color="#2b7bba")
    b2 = ax.bar(x + w/2, sy_v, w, label="SynthID-Text", color="#8e44ad")
    ax.axhline(4.0, color="black", linestyle="--", label="threshold z=4")
    ax.set_xticks(x)
    ax.set_xticklabels(order, rotation=15, ha="right")
    ax.set_ylabel("mean z-score")
    ax.set_title("Surface robustness: Green-List vs SynthID-Text")
    ax.legend()
    for bars in (b1, b2):
        for bar in bars:
            h = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2, h + 0.15,
                    f"{h:.1f}", ha="center", fontsize=9)
    fig.tight_layout()
    fig.savefig(R / "fig4_comparison_surface.png", dpi=150)
    print("wrote fig4_comparison_surface.png")


def comparison_attacks():
    gl_p = pd.read_csv(R / "paraphrase_attack.csv")
    gl_b = pd.read_csv(R / "backtranslation_attack.csv")
    sy_a = pd.read_csv(R / "synthid_attacks.csv")

    rows = [
        ("original",
         gl_p["z_original"].mean(),
         sy_a["z_original"].mean()),
        ("T5-PAWS paraphrase",
         gl_p["z_cross_paraphrase"].mean(),
         sy_a["z_t5_paraphrase"].mean()),
        ("EN-FR-EN backtrans.",
         gl_b["z_backtranslation"].mean(),
         sy_a["z_backtranslation"].mean()),
        ("GPT-2 free rewrite",
         gl_p["z_self_paraphrase"].mean(),
         sy_a["z_self_rewrite"].mean()),
    ]
    labels = [r[0] for r in rows]
    gl_v = [r[1] for r in rows]
    sy_v = [r[2] for r in rows]

    x = np.arange(len(labels))
    w = 0.38
    fig, ax = plt.subplots(figsize=(11, 5))
    b1 = ax.bar(x - w/2, gl_v, w, label="Green-List", color="#2b7bba")
    b2 = ax.bar(x + w/2, sy_v, w, label="SynthID-Text", color="#8e44ad")
    ax.axhline(4.0, color="black", linestyle="--", label="threshold z=4")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=15, ha="right")
    ax.set_ylabel("mean z-score")
    ax.set_title("Semantic attacks: Green-List vs SynthID-Text")
    ax.legend()
    for bars in (b1, b2):
        for bar in bars:
            h = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2, h + 0.15,
                    f"{h:.1f}", ha="center", fontsize=9)
    fig.tight_layout()
    fig.savefig(R / "fig5_comparison_attacks.png", dpi=150)
    print("wrote fig5_comparison_attacks.png")


if __name__ == "__main__":
    comparison_surface()
    comparison_attacks()
