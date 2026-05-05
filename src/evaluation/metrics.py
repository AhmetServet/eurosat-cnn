"""Evaluation utilities for EuroSAT CNN models.

Computes accuracy, precision, recall, F1, confusion matrices, ROC/AUC,
and generates plots.
"""

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import (
    accuracy_score,
    auc,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_curve,
)
from tqdm import tqdm


def set_style():
    plt.style.use("seaborn-v0_8-darkgrid")
    plt.rcParams.update({"figure.dpi": 150, "savefig.dpi": 150, "savefig.bbox": "tight"})


def load_best_model(model, checkpoint_path: Path, device: torch.device) -> nn.Module:
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=True)
    # Handle torch.compile _orig_mod prefix
    ckpt = {k.replace("_orig_mod.", ""): v for k, v in ckpt.items()}
    model.load_state_dict(ckpt)
    return model.to(device).eval()


@torch.no_grad()
def predict(model: nn.Module, loader, device: torch.device):
    all_preds = []
    all_labels = []
    all_probs = []

    for images, labels in tqdm(loader, desc="Evaluating", leave=False):
        images = images.to(device, non_blocking=True)
        outputs = model(images)
        probs = torch.softmax(outputs, dim=1)
        _, preds = torch.max(outputs, 1)
        all_preds.append(preds.cpu())
        all_labels.append(labels)
        all_probs.append(probs.cpu())

    return (
        torch.cat(all_preds).numpy(),
        torch.cat(all_labels).numpy(),
        torch.cat(all_probs).numpy(),
    )


def compute_metrics(y_true, y_pred, y_probs, class_names: list, num_classes: int) -> dict:
    acc = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred, average=None, zero_division=0)
    recall = recall_score(y_true, y_pred, average=None, zero_division=0)
    f1 = f1_score(y_true, y_pred, average=None, zero_division=0)

    per_class = {}
    for i, name in enumerate(class_names):
        per_class[name] = {
            "precision": round(float(precision[i]), 4),
            "recall": round(float(recall[i]), 4),
            "f1": round(float(f1[i]), 4),
        }

    # AUC (macro and per-class)
    auc_scores = {}
    for i, name in enumerate(class_names):
        y_true_bin = (y_true == i).astype(int)
        if len(np.unique(y_true_bin)) < 2:
            auc_scores[name] = None
        else:
            fpr, tpr, _ = roc_curve(y_true_bin, y_probs[:, i])
            auc_scores[name] = round(float(auc(fpr, tpr)), 4)

    valid_aucs = [v for v in auc_scores.values() if v is not None]
    macro_auc = round(float(np.mean(valid_aucs)), 4) if valid_aucs else None

    return {
        "accuracy": round(float(acc), 4),
        "precision_macro": round(float(np.mean(precision)), 4),
        "recall_macro": round(float(np.mean(recall)), 4),
        "f1_macro": round(float(np.mean(f1)), 4),
        "auc_macro": macro_auc,
        "per_class": per_class,
        "auc_per_class": auc_scores,
    }


# ---------------------------------------------------------------------------
# Confusion matrix plot
# ---------------------------------------------------------------------------

def plot_confusion_matrix(y_true, y_pred, class_names, out_path: Path, normalize: bool = True):
    cm = confusion_matrix(y_true, y_pred, normalize="true" if normalize else None)
    fig, ax = plt.subplots(figsize=(9, 8))
    im = ax.imshow(cm, cmap="Blues", aspect="auto")
    plt.colorbar(im, ax=ax, fraction=0.046)

    ax.set_xticks(range(len(class_names)))
    ax.set_yticks(range(len(class_names)))
    ax.set_xticklabels(class_names, rotation=45, ha="right", fontsize=7)
    ax.set_yticklabels(class_names, fontsize=7)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title("Confusion Matrix" + (" (normalized)" if normalize else ""))

    # Annotate cells
    threshold = 0.5 if normalize else cm.max() / 2
    for i in range(len(class_names)):
        for j in range(len(class_names)):
            val = cm[i, j]
            fmt = f"{val:.2f}" if normalize else str(int(val))
            ax.text(
                j, i, fmt,
                ha="center", va="center",
                color="white" if val > threshold else "black",
                fontsize=6,
            )

    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)


# ---------------------------------------------------------------------------
# ROC curves plot
# ---------------------------------------------------------------------------

def plot_roc_curves(y_true, y_probs, class_names, out_path: Path):
    fig, ax = plt.subplots(figsize=(8, 7))

    for i, name in enumerate(class_names):
        y_true_bin = (y_true == i).astype(int)
        if len(np.unique(y_true_bin)) < 2:
            continue
        fpr, tpr, _ = roc_curve(y_true_bin, y_probs[:, i])
        roc_auc = auc(fpr, tpr)
        ax.plot(fpr, tpr, label=f"{name} (AUC={roc_auc:.3f})", linewidth=1.2)

    ax.plot([0, 1], [0, 1], "k--", linewidth=0.8, alpha=0.4)
    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.05])
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title("ROC Curves (One-vs-All)")
    ax.legend(loc="lower right", fontsize=6, ncol=2)
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)


# ---------------------------------------------------------------------------
# Training history plot
# ---------------------------------------------------------------------------

def plot_training_history(history: dict, out_path: Path, model_name: str):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))

    epochs = range(1, len(history["train_loss"]) + 1)
    ax1.plot(epochs, history["train_loss"], label="Train Loss", marker=".")
    ax1.plot(epochs, history["val_loss"], label="Val Loss", marker=".")
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Loss")
    ax1.set_title(f"{model_name} — Loss")
    ax1.legend()

    ax2.plot(epochs, history["train_acc"], label="Train Acc", marker=".")
    ax2.plot(epochs, history["val_acc"], label="Val Acc", marker=".")
    ax2.set_xlabel("Epoch")
    ax2.set_ylabel("Accuracy")
    ax2.set_title(f"{model_name} — Accuracy")
    ax2.legend()

    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)


# ---------------------------------------------------------------------------
# Model comparison bar chart
# ---------------------------------------------------------------------------

def plot_comparison(df: pd.DataFrame, out_path: Path):
    fig, ax = plt.subplots(figsize=(10, 5))
    x = np.arange(len(df))
    width = 0.2

    ax.bar(x - 1.5 * width, df["accuracy"], width, label="Accuracy")
    ax.bar(x - 0.5 * width, df["f1_macro"], width, label="F1 (Macro)")
    ax.bar(x + 0.5 * width, df["auc_macro"], width, label="AUC (Macro)")
    ax.bar(x + 1.5 * width, df["precision_macro"], width, label="Precision (Macro)")

    ax.set_xticks(x)
    ax.set_xticklabels(df["model"], rotation=30, ha="right")
    ax.set_ylabel("Score")
    ax.set_title("Model Comparison — EuroSAT Test Set")
    ax.legend(fontsize=7)
    ax.set_ylim(0.8, 1.0)

    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)
