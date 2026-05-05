"""Training entry point.

Two-phase training for large GPUs:
  Phase 1 — frozen backbone, train classifier head (10 epochs, lr=1e-3)
  Phase 2 — unfreeze backbone, fine-tune full model  (20 epochs, lr=1e-5)

Usage:
    python scripts/03_train.py --model resnet50
    python scripts/03_train.py --all
    python scripts/03_train.py --all --no-resume   (start fresh)
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import torch
import yaml

from src.data.dataset import EurosatDataModule
from src.data.transforms import get_train_transforms, get_val_transforms
from src.models.factory import create_model, list_models, unfreeze_backbone
from src.training.config import TrainingConfig
from src.training.trainer import train_model


def set_seed(seed: int) -> None:
    import random

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def get_device() -> str:
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def print_system_info(device: str, cfg: dict) -> None:
    bs = cfg["training"]["batch_size"]
    print(f"PyTorch: {torch.__version__} | Device: {device} | Batch size: {bs}")
    if device == "cuda":
        gpu_name = torch.cuda.get_device_name(0)
        vram = torch.cuda.get_device_properties(0).total_memory / 1e9
        print(f"  GPU: {gpu_name} ({vram:.1f} GB VRAM)")


def main():
    parser = argparse.ArgumentParser(description="Train EuroSAT CNN models")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--model", type=str, help="Single model to train")
    group.add_argument("--all", action="store_true", help="Train all 5 models")
    parser.add_argument("--config", type=str, default="config/config.yaml")
    parser.add_argument("--no-resume", action="store_true", help="Ignore checkpoint, start fresh")
    parser.add_argument("--phase1-only", action="store_true", help="Skip fine-tuning phase")
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    device = get_device()
    set_seed(cfg["reproducibility"]["seed"])
    print_system_info(device, cfg)

    num_classes = cfg["model"]["num_classes"]
    input_size = cfg["model"]["input_size"]
    fine_tune = cfg["training"].get("fine_tune", False) and not args.phase1_only

    # Shared data module
    train_tf = get_train_transforms(input_size)
    val_tf = get_val_transforms(input_size)
    datamodule = EurosatDataModule(
        cfg,
        train_transforms=train_tf,
        val_transforms=val_tf,
        pin_memory=(device != "mps"),
    )

    checkpoint_dir = Path(cfg["outputs"]["checkpoints"])

    model_names = list_models() if args.all else [args.model]
    for name in model_names:
        if name not in list_models():
            print(f"Unknown model: {name}. Available: {list_models()}")
            sys.exit(1)

    print(f"Models to train: {model_names}")
    if fine_tune:
        print("Two-phase training: Phase 1 (frozen) → Phase 2 (fine-tune)")
    print()

    # ── Phase 1: Frozen backbone → train classifier head ──
    phase1_cfg = TrainingConfig.from_config_dict(cfg, device, phase="phase1")
    for name in model_names:
        model = create_model(
            name,
            num_classes=num_classes,
            dropout=phase1_cfg.dropout,
            hidden_dim=phase1_cfg.head_hidden_dim,
            freeze_backbone=True,
        )
        train_model(
            model=model,
            model_name=name,
            datamodule=datamodule,
            config=phase1_cfg,
            checkpoint_dir=checkpoint_dir,
            train_csv=cfg["splits"]["train"],
            resume=not args.no_resume,
        )

    if not fine_tune:
        print("\nAll training complete. (Phase 1 only)")
        return

    # ── Phase 2: Unfreeze → fine-tune entire model ──
    phase2_cfg = TrainingConfig.phase2_config(cfg, device)
    if phase2_cfg is None:
        return

    for name in model_names:
        phase1_best = checkpoint_dir / f"{name}_best.pt"
        if not phase1_best.exists():
            print(f"\n[SKIP] {name} fine-tune — no Phase 1 checkpoint at {phase1_best}")
            continue

        print(f"\n{'='*60}")
        print(f"Phase 2 — Fine-tuning {name}")
        print(f"{'='*60}")

        # Create unfrozen model and load Phase 1 weights
        model = create_model(
            name,
            num_classes=num_classes,
            dropout=phase2_cfg.dropout,
            hidden_dim=phase2_cfg.head_hidden_dim,
            freeze_backbone=False,
        )
        ckpt = torch.load(phase1_best, map_location=device, weights_only=True)
        ckpt = {k.replace("_orig_mod.", ""): v for k, v in ckpt.items()}
        model.load_state_dict(ckpt)
        print(f"  Loaded Phase 1 weights from {phase1_best}")

        # Fine-tune with full model
        train_model(
            model=model,
            model_name=f"{name}_ft",
            datamodule=datamodule,
            config=phase2_cfg,
            checkpoint_dir=checkpoint_dir,
            train_csv=cfg["splits"]["train"],
            resume=not args.no_resume,
        )

    print("\nAll training complete. (Phase 1 + Phase 2)")


if __name__ == "__main__":
    main()
