"""plot_three_models.py — 3-model comparison and anomaly visualization."""
from __future__ import annotations
import pathlib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

R = pathlib.Path(__file__).resolve().parent.parent / "data" / "reports"


def fig9_tpr_vs_model():
    """Bar chart: TPR@z>4 across 3 models × 2 watermarks."""
    configs = [
        ("GPT-2\n124M\nbase",       "calibration.csv",                          "z_wm"),
        ("Qwen 2.5\n500M\nchat",    "qwen_green_calibration.csv",                "z_wm"),
        ("TinyLlama\n1.1B\nchat",   "tinyllama_greenlist_calibration.csv",       "z_wm"),
    ]
    synthid = [
        ("GPT-2",   "synthid_calibration.csv",         "z_wm"),
        ("Qwen",    "qwen_synthid_calibration.csv",    "z_wm"),
        ("TL",      "tinyllama_synthid_calibration.csv","z_wm"),
    ]

    gl_tprs, sy_tprs, labels = [], [], []
    for (label, path, col), (_, spath, scol) in zip(configs, synthid):
        gl = pd.read_csv(R / path)
        sy = pd.read_csv(R / spath)
        gl_tprs.append((gl[col] > 4).mean() * 100)
        sy_tprs.append((sy[scol] > 4).mean() * 100)
        labels.append(label)

    x = np.arange(len(labels))
    w = 0.38
    fig, ax = plt.subplots(figsize=(9, 5))
    b1 = ax.bar(x - w/2, gl_tprs, w, label="Green-List", color="#2b7bba")
    b2 = ax.bar(x + w/2, sy_tprs, w, label="SynthID-Text", color="#8e44ad")
    ax.axhline(100, color="black", linestyle=":", alpha=0.4)
    ax.set_xticks(x); ax.set_xticklabels(labels)
    ax.set_ylim(0, 115)
    ax.set_ylabel("TPR@z>4 (%)")
    ax.set_title("Detection rate across model sizes and types")
    ax.legend()
    for bars in (b1, b2):
        for bar in bars:
            h = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2, h + 1.5,
                    f"{h:.0f}%", ha="center", fontsize=10, fontweight="bold")
    fig.tight_layout()
    fig.savefig(R / "fig9_tpr_vs_model.png", dpi=150)
    print("wrote fig9_tpr_vs_model.png")


def fig10_separation_curve():
    """Line chart: separation vs model (log-scale x-axis)."""
    models = ["GPT-2", "Qwen 2.5", "TinyLlama"]
    sizes = [124, 500, 1100]

    gl_seps = []
    sy_seps = []
    for gl_path, sy_path in [
        ("calibration.csv", "synthid_calibration.csv"),
        ("qwen_green_calibration.csv", "qwen_synthid_calibration.csv"),
        ("tinyllama_greenlist_calibration.csv", "tinyllama_synthid_calibration.csv"),
    ]:
        g = pd.read_csv(R / gl_path)
        s = pd.read_csv(R / sy_path)
        gl_seps.append(g["z_wm"].mean() - g["z_clean"].mean())
        sy_seps.append(s["z_wm"].mean() - s["z_clean"].mean())

    fig, ax = plt.subplots(figsize=(9, 5))
    ax.plot(sizes, gl_seps, "o-", color="#2b7bba", label="Green-List", lw=2, markersize=10)
    ax.plot(sizes, sy_seps, "s-", color="#8e44ad", label="SynthID-Text", lw=2, markersize=10)
    for x, y, name in zip(sizes, gl_seps, models):
        ax.annotate(f"{y:.1f}", (x, y), textcoords="offset points",
                    xytext=(0, 10), ha="center", color="#2b7bba", fontsize=10)
    for x, y in zip(sizes, sy_seps):
        ax.annotate(f"{y:.1f}", (x, y), textcoords="offset points",
                    xytext=(0, -18), ha="center", color="#8e44ad", fontsize=10)
    ax.set_xscale("log")
    ax.set_xticks(sizes)
    ax.set_xticklabels([f"{m}\n{s}M" for m, s in zip(models, sizes)])
    ax.set_xlabel("model size (params, log scale)")
    ax.set_ylabel("separation (σ)")
    ax.set_title("Watermark separation vs model size — not monotonic")
    ax.axhline(0, color="gray", linestyle=":", alpha=0.5)
    ax.legend()
    fig.tight_layout()
    fig.savefig(R / "fig10_separation_curve.png", dpi=150)
    print("wrote fig10_separation_curve.png")


def fig11_heatmap_3models():
    """Heatmap: mean z by (transform × model × watermark)."""
    order = ["identity", "whisper_light", "lowercase", "punct_strip",
             "word_drop_10", "word_shuffle_10"]
    files = [
        ("GPT-2\nGL",       "robustness.csv"),
        ("GPT-2\nSyn",      "synthid_robustness.csv"),
        ("Qwen\nGL",        "qwen_green_robustness.csv"),
        ("Qwen\nSyn",       "qwen_synthid_robustness.csv"),
        ("TL\nGL",          "tinyllama_greenlist_robustness.csv"),
        ("TL\nSyn",         "tinyllama_synthid_robustness.csv"),
    ]
    data = np.zeros((len(order), len(files)))
    for j, (_, fn) in enumerate(files):
        df = pd.read_csv(R / fn)
        m = df.groupby("transform")["z"].mean()
        for i, t in enumerate(order):
            data[i, j] = m.get(t, 0.0)

    fig, ax = plt.subplots(figsize=(10, 6))
    im = ax.imshow(data, cmap="RdYlGn", aspect="auto", vmin=-2, vmax=13)
    ax.set_xticks(np.arange(len(files)))
    ax.set_xticklabels([f[0] for f in files], fontsize=10)
    ax.set_yticks(np.arange(len(order)))
    ax.set_yticklabels(order)
    for i in range(len(order)):
        for j in range(len(files)):
            ax.text(j, i, f"{data[i,j]:.1f}", ha="center", va="center",
                    color="black", fontsize=10)
    ax.set_title("Mean z-score: transform × watermark × model")
    fig.colorbar(im, ax=ax, label="mean z-score")
    fig.tight_layout()
    fig.savefig(R / "fig11_heatmap_3models.png", dpi=150)
    print("wrote fig11_heatmap_3models.png")


if __name__ == "__main__":
    fig9_tpr_vs_model()
    fig10_separation_curve()
    fig11_heatmap_3models()
