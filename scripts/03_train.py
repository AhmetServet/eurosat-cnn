"""Training entry point.

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
from src.models.factory import create_model, list_models
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


def print_system_info(device: str) -> None:
    print(f"PyTorch: {torch.__version__} | Device: {device}")
    if device == "cuda":
        print(f"  GPU: {torch.cuda.get_device_name(0)}")
        print(f"  VRAM: {torch.cuda.get_device_properties(0).total_mem / 1e9:.1f} GB")
    elif device == "mps":
        print("  GPU: Apple Silicon (MPS)")


def main():
    parser = argparse.ArgumentParser(description="Train EuroSAT CNN models")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--model", type=str, help="Single model to train")
    group.add_argument("--all", action="store_true", help="Train all 5 models")
    parser.add_argument("--config", type=str, default="config/config.yaml")
    parser.add_argument("--no-resume", action="store_true", help="Ignore checkpoint, start fresh")
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    device = get_device()
    set_seed(cfg["reproducibility"]["seed"])
    print_system_info(device)

    train_config = TrainingConfig.from_config_dict(cfg, device)
    input_size = cfg["model"]["input_size"]

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

    print(f"Models to train: {model_names}\n")

    for name in model_names:
        model = create_model(
            name,
            num_classes=cfg["model"]["num_classes"],
            dropout=train_config.dropout,
            hidden_dim=train_config.head_hidden_dim,
            freeze_backbone=train_config.freeze_backbone,
        )
        train_model(
            model=model,
            model_name=name,
            datamodule=datamodule,
            config=train_config,
            checkpoint_dir=checkpoint_dir,
            train_csv=cfg["splits"]["train"],
            resume=not args.no_resume,
        )

    print("\nAll training complete.")


if __name__ == "__main__":
    main()
