"""plot_cross_model.py — cross-model comparison of two watermarks."""
from __future__ import annotations
import pathlib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

R = pathlib.Path(__file__).resolve().parent.parent / "data" / "reports"


def fig6_calibration_separation():
    """Bar chart: separation (sigma) across 4 configurations."""
    rows = []
    # GPT-2
    gl_gpt2 = pd.read_csv(R / "calibration.csv")
    sy_gpt2 = pd.read_csv(R / "synthid_calibration.csv")
    rows.append(("GPT-2\nGreen-List",
                 gl_gpt2["z_clean"].mean(), gl_gpt2["z_wm"].mean()))
    rows.append(("GPT-2\nSynthID",
                 sy_gpt2["z_clean"].mean(), sy_gpt2["z_wm"].mean()))
    # TinyLlama
    gl_tl = pd.read_csv(R / "tinyllama_greenlist_calibration.csv")
    sy_tl = pd.read_csv(R / "tinyllama_synthid_calibration.csv")
    rows.append(("TinyLlama\nGreen-List",
                 gl_tl["z_clean"].mean(), gl_tl["z_wm"].mean()))
    rows.append(("TinyLlama\nSynthID",
                 sy_tl["z_clean"].mean(), sy_tl["z_wm"].mean()))

    labels = [r[0] for r in rows]
    sep = [r[2] - r[1] for r in rows]
    zc = [r[1] for r in rows]
    zw = [r[2] for r in rows]

    x = np.arange(len(labels))
    w = 0.35
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.bar(x - w/2, zc, w, label="z_clean mean", color="#95a5a6")
    ax.bar(x + w/2, zw, w, label="z_wm mean", color="#2980b9")
    ax.axhline(4.0, color="black", linestyle="--", label="threshold z=4")
    ax.set_xticks(x); ax.set_xticklabels(labels)
    ax.set_ylabel("mean z-score")
    ax.set_title("Calibration across models and watermarks")
    ax.legend()
    for xi, sep_i in zip(x, sep):
        ax.text(xi, max(zw) + 0.5, f"sep = {sep_i:.2f}σ",
                ha="center", fontsize=10, color="#c0392b", fontweight="bold")
    fig.tight_layout()
    fig.savefig(R / "fig6_cross_model_calibration.png", dpi=150)
    print("wrote fig6_cross_model_calibration.png")


def fig7_heatmap():
    """Heatmap: mean z by (model, watermark, transform)."""
    gl_gpt2 = pd.read_csv(R / "robustness.csv")
    sy_gpt2 = pd.read_csv(R / "synthid_robustness.csv")
    gl_tl   = pd.read_csv(R / "tinyllama_greenlist_robustness.csv")
    sy_tl   = pd.read_csv(R / "tinyllama_synthid_robustness.csv")

    order = ["identity", "whisper_light", "lowercase", "punct_strip",
             "word_drop_10", "word_shuffle_10"]
    cols = ["GPT-2\nGreen-List", "GPT-2\nSynthID",
            "TinyLlama\nGreen-List", "TinyLlama\nSynthID"]
    data = np.zeros((len(order), len(cols)))
    for j, df in enumerate([gl_gpt2, sy_gpt2, gl_tl, sy_tl]):
        m = df.groupby("transform")["z"].mean()
        for i, t in enumerate(order):
            data[i, j] = m.get(t, 0.0)

    fig, ax = plt.subplots(figsize=(8, 6))
    im = ax.imshow(data, cmap="RdYlGn", aspect="auto", vmin=0, vmax=14)
    ax.set_xticks(np.arange(len(cols)))
    ax.set_xticklabels(cols, rotation=0)
    ax.set_yticks(np.arange(len(order)))
    ax.set_yticklabels(order)
    for i in range(len(order)):
        for j in range(len(cols)):
            ax.text(j, i, f"{data[i,j]:.1f}", ha="center", va="center",
                    color="black", fontsize=10)
    ax.set_title("Mean z-score by model × watermark × transform")
    fig.colorbar(im, ax=ax, label="mean z-score")
    fig.tight_layout()
    fig.savefig(R / "fig7_heatmap.png", dpi=150)
    print("wrote fig7_heatmap.png")


def fig8_tpr_summary():
    """Bar chart: TPR@z>4 for each configuration (calibration)."""
    configs = []
    for name, path, col in [
        ("GPT-2 | Green",   "calibration.csv",                  "z_wm"),
        ("GPT-2 | SynthID", "synthid_calibration.csv",          "z_wm"),
        ("TL | Green",      "tinyllama_greenlist_calibration.csv", "z_wm"),
        ("TL | SynthID",    "tinyllama_synthid_calibration.csv",   "z_wm"),
    ]:
        df = pd.read_csv(R / path)
        tpr = (df[col] > 4).mean() * 100
        configs.append((name, tpr))

    labels = [c[0] for c in configs]
    tprs = [c[1] for c in configs]
    colors = ["#27ae60" if t >= 99 else "#e67e22" if t >= 60 else "#c0392b"
              for t in tprs]

    fig, ax = plt.subplots(figsize=(9, 5))
    bars = ax.bar(labels, tprs, color=colors)
    ax.set_ylim(0, 105)
    ax.set_ylabel("TPR@z>4 (%)")
    ax.set_title("Detection rate by model and watermark (8 prompts)")
    ax.axhline(100, color="black", linestyle=":", alpha=0.4)
    for bar, v in zip(bars, tprs):
        ax.text(bar.get_x() + bar.get_width()/2, v + 1,
                f"{v:.0f}%", ha="center", fontsize=11, fontweight="bold")
    fig.tight_layout()
    fig.savefig(R / "fig8_tpr_summary.png", dpi=150)
    print("wrote fig8_tpr_summary.png")


if __name__ == "__main__":
    fig6_calibration_separation()
    fig7_heatmap()
    fig8_tpr_summary()
