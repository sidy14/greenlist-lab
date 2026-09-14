"""plot_results.py"""
from __future__ import annotations
import pathlib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

ROOT = pathlib.Path(__file__).resolve().parent.parent
OUT = ROOT / "data" / "reports"


def plot_calibration():
    df = pd.read_csv(OUT / "calibration.csv")
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.hist(df["z_clean"], bins=8, alpha=0.6, label="clean", color="#2b7bba")
    ax.hist(df["z_wm"], bins=8, alpha=0.6, label="watermarked", color="#c0392b")
    ax.axvline(4.0, color="black", linestyle="--", label="threshold z=4")
    ax.set_xlabel("z-score"); ax.set_ylabel("count")
    ax.set_title("Green-list calibration")
    ax.legend(); fig.tight_layout()
    fig.savefig(OUT / "calibration.png", dpi=150)
    print("wrote calibration.png")


def plot_robustness():
    df = pd.read_csv(OUT / "robustness.csv")
    means = df.groupby("transform")["z"].mean().sort_values()
    fig, ax = plt.subplots(figsize=(8, 4))
    colors = ["#c0392b" if v < 4 else "#27ae60" for v in means.values]
    ax.barh(means.index, means.values, color=colors)
    ax.axvline(4.0, color="black", linestyle="--", label="threshold z=4")
    ax.set_xlabel("mean z-score")
    ax.set_title("Robustness of green-list watermark")
    ax.legend(); fig.tight_layout()
    fig.savefig(OUT / "robustness.png", dpi=150)
    print("wrote robustness.png")


if __name__ == "__main__":
    plot_calibration()
    plot_robustness()
