"""Exploratory Data Analysis for the EuroSAT dataset.

Generates and saves to outputs/eda/:
  - class_distribution.png  — bar chart of samples per class
  - sample_grid.png         — 10 random images per class (10 × 10 grid)
  - pixel_histograms.png    — per-class pixel intensity distributions
  - channel_stats.json      — per-channel mean and standard deviation
  - split_summary.csv       — train/val/test class counts
"""

import json
import random
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yaml
from PIL import Image


def set_style():
    plt.style.use("seaborn-v0_8-darkgrid")
    plt.rcParams.update(
        {
            "figure.dpi": 150,
            "savefig.dpi": 150,
            "savefig.bbox": "tight",
            "font.size": 9,
        }
    )


def load_config(config_path: str = "config/config.yaml") -> dict:
    with open(config_path) as f:
        return yaml.safe_load(f)


# ---------------------------------------------------------------------------
# 1. Class distribution
# ---------------------------------------------------------------------------

def plot_class_distribution(df: pd.DataFrame, out_path: Path) -> None:
    counts = df["className"].value_counts().sort_index()
    colors = plt.cm.tab10(np.linspace(0, 1, len(counts)))
    fig, ax = plt.subplots(figsize=(10, 5))
    bars = ax.bar(counts.index, counts.values, color=colors, edgecolor="white")
    for bar, val in zip(bars, counts.values):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 40,
            str(val),
            ha="center",
            fontsize=8,
            fontweight="bold",
        )
    ax.set_title("EuroSAT — Class Distribution (27,000 images)", fontsize=13, pad=15)
    ax.set_ylabel("Number of Images")
    ax.set_ylim(0, max(counts.values) * 1.15)
    plt.xticks(rotation=30, ha="right")
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)
    print(f"  ✓ {out_path}")


# ---------------------------------------------------------------------------
# 2. Sample image grid (10 per class)
# ---------------------------------------------------------------------------

def plot_sample_grid(df: pd.DataFrame, out_path: Path, seed: int = 42) -> None:
    random.seed(seed)
    classes = sorted(df["className"].unique())
    n_classes = len(classes)
    n_per_class = 10

    fig, axes = plt.subplots(n_classes, n_per_class, figsize=(n_per_class * 1.2, n_classes * 1.2))
    fig.suptitle("EuroSAT — Random Samples (10 per class)", fontsize=13, y=1.01)

    for row, cls in enumerate(classes):
        class_samples = df[df["className"] == cls]["filename"].tolist()
        chosen = random.sample(class_samples, min(n_per_class, len(class_samples)))
        for col, img_path in enumerate(chosen):
            ax = axes[row, col]
            try:
                img = Image.open(img_path)
                ax.imshow(img)
            except Exception:
                ax.text(0.5, 0.5, "ERR", transform=ax.transAxes, ha="center", va="center")
            ax.axis("off")
            if col == 0:
                ax.set_ylabel(cls, fontsize=8, rotation=0, ha="right", va="center")

    fig.tight_layout()
    fig.savefig(out_path, bbox_inches="tight", pad_inches=0.3)
    plt.close(fig)
    print(f"  ✓ {out_path}")


# ---------------------------------------------------------------------------
# 3. Per-channel statistics (sampled)
# ---------------------------------------------------------------------------

def compute_channel_stats(df: pd.DataFrame, out_path: Path, max_samples: int = 5000) -> None:
    files = df["filename"].tolist()
    if len(files) > max_samples:
        files = random.sample(files, max_samples)

    r_vals, g_vals, b_vals = [], [], []
    for f in files:
        img = Image.open(f).convert("RGB")
        arr = np.array(img, dtype=np.float32) / 255.0
        r_vals.append(arr[:, :, 0].ravel())
        g_vals.append(arr[:, :, 1].ravel())
        b_vals.append(arr[:, :, 2].ravel())

    r_all = np.concatenate(r_vals)
    g_all = np.concatenate(g_vals)
    b_all = np.concatenate(b_vals)

    stats = {
        "R": {"mean": float(r_all.mean()), "std": float(r_all.std())},
        "G": {"mean": float(g_all.mean()), "std": float(g_all.std())},
        "B": {"mean": float(b_all.mean()), "std": float(b_all.std())},
        "samples_used": len(files),
    }

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(stats, f, indent=2)
    print(f"  ✓ {out_path}  (computed from {len(files)} images)")

    # Pretty print
    print(f"     R: mean={stats['R']['mean']:.4f}, std={stats['R']['std']:.4f}")
    print(f"     G: mean={stats['G']['mean']:.4f}, std={stats['G']['std']:.4f}")
    print(f"     B: mean={stats['B']['mean']:.4f}, std={stats['B']['std']:.4f}")


# ---------------------------------------------------------------------------
# 4. Pixel intensity histograms per class
# ---------------------------------------------------------------------------

def plot_pixel_histograms(df: pd.DataFrame, out_path: Path) -> None:
    classes = sorted(df["className"].unique())
    n_classes = len(classes)
    cols = 5
    rows = (n_classes + cols - 1) // cols

    fig, axes = plt.subplots(rows, cols, figsize=(cols * 3, rows * 2.5))
    axes = axes.flatten()

    for i, cls in enumerate(classes):
        ax = axes[i]
        class_files = df[df["className"] == cls]["filename"].tolist()
        sampled = random.sample(class_files, min(200, len(class_files)))

        r_all, g_all, b_all = [], [], []
        for f in sampled:
            arr = np.array(Image.open(f).convert("RGB"), dtype=np.float32) / 255.0
            r_all.append(arr[:, :, 0].ravel())
            g_all.append(arr[:, :, 1].ravel())
            b_all.append(arr[:, :, 2].ravel())

        r_data = np.concatenate(r_all)
        g_data = np.concatenate(g_all)
        b_data = np.concatenate(b_all)

        ax.hist(r_data, bins=64, alpha=0.5, color="red", label="R", density=True)
        ax.hist(g_data, bins=64, alpha=0.5, color="green", label="G", density=True)
        ax.hist(b_data, bins=64, alpha=0.5, color="blue", label="B", density=True)
        ax.set_title(cls, fontsize=8)
        ax.set_xlim(0, 1)
        if i == 0:
            ax.legend(fontsize=6, loc="upper right")

    # Hide unused axes
    for j in range(i + 1, len(axes)):
        axes[j].set_visible(False)

    fig.suptitle("EuroSAT — Per-Class Pixel Intensity Distributions", fontsize=13)
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)
    print(f"  ✓ {out_path}")


# ---------------------------------------------------------------------------
# 5. Split summary
# ---------------------------------------------------------------------------

def save_split_summary(config: dict, out_path: Path) -> None:
    splits = ["train", "val", "test"]
    records = []
    for split_name in splits:
        csv_path = Path(config["splits"][split_name])
        df = pd.read_csv(csv_path)
        counts = df["className"].value_counts().sort_index()
        for cls, cnt in counts.items():
            records.append({"split": split_name, "class": cls, "count": cnt})

    summary_df = pd.DataFrame(records)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    summary_df.to_csv(out_path, index=False)
    print(f"  ✓ {out_path}")

    # Pivot table
    pivot = summary_df.pivot(index="class", columns="split", values="count")
    print("\n  Split verification (class × split):")
    print(pivot.to_string())


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    set_style()
    config = load_config()
    eda_dir = Path(config["outputs"]["eda"])
    eda_dir.mkdir(parents=True, exist_ok=True)

    # Load full dataset from train.csv
    train_csv = config["splits"]["train"]
    df = pd.read_csv(train_csv)
    print(f"Loaded {len(df)} samples from {train_csv}\n")

    # 1. Class distribution
    print("1. Plotting class distribution ...")
    plot_class_distribution(df, eda_dir / "class_distribution.png")

    # 2. Sample grid
    print("2. Generating sample grid ...")
    plot_sample_grid(df, eda_dir / "sample_grid.png", seed=config["reproducibility"]["seed"])

    # 3. Channel statistics
    print("3. Computing channel statistics ...")
    compute_channel_stats(df, eda_dir / "channel_stats.json")

    # 4. Pixel histograms
    print("4. Plotting pixel intensity histograms ...")
    plot_pixel_histograms(df, eda_dir / "pixel_histograms.png")

    # 5. Split summary
    print("5. Verifying splits ...")
    save_split_summary(config, eda_dir / "split_summary.csv")

    print("\nEDA complete. All outputs saved to", eda_dir.resolve())


if __name__ == "__main__":
    main()
