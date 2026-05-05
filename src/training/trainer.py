"""Training loop with checkpoint/resume, AMP, early stopping, LR scheduling."""

import json
import time
from pathlib import Path

import pandas as pd
import torch
import torch.nn as nn
from torch.cuda.amp import GradScaler, autocast
from torch.optim import AdamW
from torch.optim.lr_scheduler import ReduceLROnPlateau
from tqdm import tqdm

from src.training.config import TrainingConfig


def compute_class_weights(train_csv: str, num_classes: int) -> torch.Tensor:
    df = pd.read_csv(train_csv)
    counts = df["label"].value_counts().sort_index().values
    weights = 1.0 / counts
    return torch.tensor(weights / weights.sum() * num_classes, dtype=torch.float32)


def _compile_model(model: nn.Module, device: torch.device) -> nn.Module:
    if device.type == "cuda":
        try:
            return torch.compile(model)
        except Exception:
            return model
    elif device.type == "mps":
        try:
            return torch.compile(model, backend="aot_eager")
        except Exception:
            return model
    return model


def _clean_state_dict(state_dict: dict) -> dict:
    """Strip torch.compile's _orig_mod. prefix from state_dict keys."""
    return {k.replace("_orig_mod.", ""): v for k, v in state_dict.items()}


def save_checkpoint(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler,
    epoch: int,
    best_val_loss: float,
    patience_counter: int,
    history: dict,
    path: Path,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model_state_dict": _clean_state_dict(model.state_dict()),
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": scheduler.state_dict(),
            "epoch": epoch,
            "best_val_loss": best_val_loss,
            "patience_counter": patience_counter,
            "history": history,
        },
        path,
    )


def load_checkpoint(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler,
    path: Path,
    device: torch.device,
):
    ckpt = torch.load(path, map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model_state_dict"])
    optimizer.load_state_dict(ckpt["optimizer_state_dict"])
    scheduler.load_state_dict(ckpt["scheduler_state_dict"])
    return (
        ckpt["epoch"],
        ckpt["best_val_loss"],
        ckpt["patience_counter"],
        ckpt["history"],
    )


def train_one_epoch(model, loader, criterion, optimizer, device, use_amp: bool, scaler):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    for images, labels in tqdm(loader, desc="Train", leave=False):
        images, labels = images.to(device, non_blocking=True), labels.to(device, non_blocking=True)
        optimizer.zero_grad()

        if use_amp:
            with autocast():
                outputs = model(images)
                loss = criterion(outputs, labels)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

        running_loss += loss.item() * images.size(0)
        _, preds = torch.max(outputs, 1)
        correct += (preds == labels).sum().item()
        total += labels.size(0)

    return running_loss / total, correct / total


@torch.no_grad()
def validate(model, loader, criterion, device):
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0

    for images, labels in tqdm(loader, desc="Val", leave=False):
        images, labels = images.to(device, non_blocking=True), labels.to(device, non_blocking=True)
        outputs = model(images)
        loss = criterion(outputs, labels)
        running_loss += loss.item() * images.size(0)
        _, preds = torch.max(outputs, 1)
        correct += (preds == labels).sum().item()
        total += labels.size(0)

    return running_loss / total, correct / total


def train_model(
    model: nn.Module,
    model_name: str,
    datamodule,
    config: TrainingConfig,
    checkpoint_dir: Path,
    train_csv: str,
    resume: bool = True,
) -> dict:
    device = torch.device(config.device)
    use_amp = device.type == "cuda"

    model = model.to(device)

    class_weights = compute_class_weights(train_csv, num_classes=10).to(device)
    criterion = nn.CrossEntropyLoss(weight=class_weights)
    optimizer = AdamW(
        model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay
    )
    scheduler = ReduceLROnPlateau(
        optimizer, mode="min", factor=config.scheduler_factor,
        patience=config.scheduler_patience,
    )
    scaler = GradScaler() if use_amp else None

    train_loader = datamodule.train_dataloader()
    val_loader = datamodule.val_dataloader()

    # State
    start_epoch = 1
    best_val_loss = float("inf")
    patience_counter = 0
    best_epoch = 0
    history = {"train_loss": [], "train_acc": [], "val_loss": [], "val_acc": []}

    ckpt_path = checkpoint_dir / f"{model_name}_checkpoint.pt"
    best_path = checkpoint_dir / f"{model_name}_best.pt"

    # Resume from checkpoint if requested and exists (load BEFORE compiling)
    if resume and ckpt_path.exists():
        start_epoch, best_val_loss, patience_counter, history = load_checkpoint(
            model, optimizer, scheduler, ckpt_path, device
        )
        start_epoch += 1
        best_epoch = start_epoch - 1
        print(f"  Resumed from {ckpt_path} → starting epoch {start_epoch}")
    elif resume:
        print("  No checkpoint found — training from scratch")

    # Compile after loading (so saved state_dicts stay clean / device-agnostic)
    model = _compile_model(model, device)

    print(f"\n{'='*60}")
    print(f"Training {model_name} | device={device} | epochs={config.epochs}")
    print(f"  AMP={'on' if use_amp else 'off'} | batch_size={config.batch_size} | compile=True")
    print(f"{'='*60}")

    for epoch in range(start_epoch, config.epochs + 1):
        t0 = time.time()

        train_loss, train_acc = train_one_epoch(
            model, train_loader, criterion, optimizer, device, use_amp, scaler
        )
        val_loss, val_acc = validate(model, val_loader, criterion, device)
        elapsed = time.time() - t0

        scheduler.step(val_loss)

        history["train_loss"].append(round(train_loss, 6))
        history["train_acc"].append(round(train_acc, 6))
        history["val_loss"].append(round(val_loss, 6))
        history["val_acc"].append(round(val_acc, 6))

        print(
            f"Epoch {epoch:2d}/{config.epochs} | "
            f"train_loss={train_loss:.4f} train_acc={train_acc:.4f} | "
            f"val_loss={val_loss:.4f} val_acc={val_acc:.4f} | "
            f"{elapsed:.0f}s"
        )

        # Save checkpoint after every epoch (for resume)
        save_checkpoint(
            model, optimizer, scheduler, epoch, best_val_loss, patience_counter, history, ckpt_path
        )

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_epoch = epoch
            patience_counter = 0
            torch.save(model.state_dict(), best_path)
        else:
            patience_counter += 1
            if patience_counter >= config.early_stopping_patience:
                print(
                    f"Early stopping at epoch {epoch} "
                    f"(no improvement for {config.early_stopping_patience} epochs)"
                )
                break

    # Save final history
    history_path = checkpoint_dir / f"{model_name}_history.json"
    history_path.write_text(
        json.dumps({"best_epoch": best_epoch, "best_val_loss": best_val_loss, **history}, indent=2)
    )

    print(
        f"\nBest: epoch={best_epoch} val_loss={best_val_loss:.4f} | "
        f"saved to {best_path}"
    )
    return history
