"""Evaluation entry point.

Loads all trained models, evaluates on the test set, and produces:
  - outputs/metrics/{model}.json
  - outputs/plots/{model}_confusion.png
  - outputs/plots/{model}_roc.png
  - outputs/plots/{model}_training.png
  - outputs/report/model_comparison.csv
  - outputs/report/model_comparison.png

Usage:
    python scripts/04_evaluate.py
"""

import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pandas as pd
import torch
import yaml

from src.data.dataset import EurosatDataModule
from src.data.transforms import get_val_transforms
from src.evaluation.metrics import (
    compute_metrics,
    load_best_model,
    plot_comparison,
    plot_confusion_matrix,
    plot_roc_curves,
    plot_training_history,
    predict,
    set_style,
)
from src.models.factory import create_model, list_models


def get_device() -> str:
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def main():
    set_style()

    with open("config/config.yaml") as f:
        cfg = yaml.safe_load(f)

    device_name = get_device()
    device = torch.device(device_name)
    print(f"Device: {device_name}")

    class_names = sorted(
        ["AnnualCrop", "Forest", "HerbaceousVegetation", "Highway",
         "Industrial", "Pasture", "PermanentCrop", "Residential", "River", "SeaLake"]
    )
    num_classes = len(class_names)
    input_size = cfg["model"]["input_size"]

    val_tf = get_val_transforms(input_size)
    datamodule = EurosatDataModule(cfg, val_transforms=val_tf, pin_memory=(device_name == "cuda"))
    test_loader = datamodule.test_dataloader()

    checkpoint_dir = Path(cfg["outputs"]["checkpoints"])
    metrics_dir = Path(cfg["outputs"]["metrics"])
    plots_dir = Path(cfg["outputs"]["plots"])
    report_dir = Path(cfg["outputs"]["report"])
    metrics_dir.mkdir(parents=True, exist_ok=True)
    plots_dir.mkdir(parents=True, exist_ok=True)
    report_dir.mkdir(parents=True, exist_ok=True)

    model_names = list_models()
    summary = []

    for name in model_names:
        best_path = checkpoint_dir / f"{name}_best.pt"
        history_path = checkpoint_dir / f"{name}_history.json"

        if not best_path.exists():
            print(f"\n[SKIP] {name} — no checkpoint found at {best_path}")
            continue

        print(f"\n{'='*60}")
        print(f"Evaluating {name}")
        print(f"{'='*60}")

        model = create_model(
            name, num_classes=num_classes,
            dropout=cfg["training"]["dropout"],
            hidden_dim=cfg["model"]["head_hidden_dim"],
            freeze_backbone=cfg["model"]["freeze_backbone"],
        )

        model = load_best_model(model, best_path, device)

        t0 = time.time()
        y_pred, y_true, y_probs = predict(model, test_loader, device)
        elapsed = time.time() - t0
        n_samples = len(y_true)

        metrics = compute_metrics(y_true, y_pred, y_probs, class_names, num_classes)
        metrics["inference_ms_per_image"] = round((elapsed / n_samples) * 1000, 2)
        metrics["total_params"] = sum(p.numel() for p in model.parameters())
        metrics["trainable_params"] = sum(p.numel() for p in model.parameters() if p.requires_grad)

        # Save metrics JSON
        with open(metrics_dir / f"{name}.json", "w") as f:
            json.dump(metrics, f, indent=2)

        # Confusion matrix
        plot_confusion_matrix(y_true, y_pred, class_names, plots_dir / f"{name}_confusion.png")

        # ROC curves
        plot_roc_curves(y_true, y_probs, class_names, plots_dir / f"{name}_roc.png")

        # Training history
        if history_path.exists():
            history = json.loads(history_path.read_text())
            plot_training_history(history, plots_dir / f"{name}_training.png", name)

        print(f"  Accuracy: {metrics['accuracy']:.4f}  |  "
              f"F1 Macro: {metrics['f1_macro']:.4f}  |  "
              f"AUC Macro: {metrics['auc_macro']:.4f}")
        print(f"  Inference: {metrics['inference_ms_per_image']} ms/img  ({elapsed:.1f}s total)")

        summary.append({
            "model": name,
            "accuracy": metrics["accuracy"],
            "f1_macro": metrics["f1_macro"],
            "auc_macro": metrics["auc_macro"],
            "precision_macro": metrics["precision_macro"],
            "inference_ms": metrics["inference_ms_per_image"],
            "total_params_M": round(metrics["total_params"] / 1e6, 1),
        })

    if not summary:
        print("\nNo models evaluated.")
        return

    # Comparison report
    df = pd.DataFrame(summary)
    df = df.sort_values("accuracy", ascending=False)
    df.to_csv(report_dir / "model_comparison.csv", index=False)
    plot_comparison(df, report_dir / "model_comparison.png")

    print(f"\n{'='*60}")
    print("MODEL COMPARISON (sorted by accuracy)")
    print(f"{'='*60}")
    print(df.to_string(index=False))
    print(f"\nWinner: {df.iloc[0]['model']}  (accuracy={df.iloc[0]['accuracy']:.4f})")

    print(f"\nAll outputs saved to:")
    print(f"  metrics/ → {metrics_dir}")
    print(f"  plots/   → {plots_dir}")
    print(f"  report/  → {report_dir}")


if __name__ == "__main__":
    main()
