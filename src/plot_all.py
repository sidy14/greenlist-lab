"""plot_all.py — unified visualisation."""
from __future__ import annotations
import pathlib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

ROOT = pathlib.Path(__file__).resolve().parent.parent
R = ROOT / "data" / "reports"


def calibration():
    df = pd.read_csv(R / "calibration.csv")
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.hist(df["z_clean"], bins=8, alpha=0.65, label="clean", color="#2b7bba")
    ax.hist(df["z_wm"], bins=8, alpha=0.65, label="watermarked", color="#c0392b")
    ax.axvline(4.0, color="black", linestyle="--", label="threshold z=4")
    ax.set_xlabel("z-score"); ax.set_ylabel("count")
    ax.set_title("Calibration: clean vs watermarked")
    ax.legend(); fig.tight_layout()
    fig.savefig(R / "fig1_calibration.png", dpi=150)
    print("wrote fig1_calibration.png")


def robustness():
    df = pd.read_csv(R / "robustness.csv")
    means = df.groupby("transform")["z"].mean().sort_values()
    fig, ax = plt.subplots(figsize=(8, 4))
    colors = ["#27ae60" if v > 4 else "#c0392b" for v in means.values]
    ax.barh(means.index, means.values, color=colors)
    ax.axvline(4.0, color="black", linestyle="--", label="threshold z=4")
    ax.set_xlabel("mean z-score")
    ax.set_title("Robustness: surface transformations")
    ax.legend(); fig.tight_layout()
    fig.savefig(R / "fig2_robustness.png", dpi=150)
    print("wrote fig2_robustness.png")


def attacks():
    para = pd.read_csv(R / "paraphrase_attack.csv")
    back = pd.read_csv(R / "backtranslation_attack.csv")

    rows = [
        ("original", para["z_original"].mean()),
        ("T5-PAWS paraphrase", para["z_cross_paraphrase"].mean()),
        ("EN-FR-EN backtranslation", back["z_backtranslation"].mean()),
        ("GPT-2 free rewrite", para["z_self_paraphrase"].mean()),
    ]
    labels, vals = zip(*rows)

    fig, ax = plt.subplots(figsize=(8, 4))
    colors = ["#27ae60" if v > 4 else "#c0392b" for v in vals]
    bars = ax.barh(range(len(labels)), vals, color=colors)
    ax.set_yticks(range(len(labels)))
    ax.set_yticklabels(labels)
    ax.axvline(4.0, color="black", linestyle="--", label="threshold z=4")
    ax.set_xlabel("mean z-score")
    ax.set_title("Semantic attacks: what breaks the watermark")
    ax.legend()
    for bar, val in zip(bars, vals):
        ax.text(val + 0.2, bar.get_y() + bar.get_height()/2,
                f"{val:+.1f}", va="center", fontsize=10)
    fig.tight_layout()
    fig.savefig(R / "fig3_attacks.png", dpi=150)
    print("wrote fig3_attacks.png")


if __name__ == "__main__":
    calibration()
    robustness()
    attacks()
