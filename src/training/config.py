"""Training configuration dataclass."""

from dataclasses import dataclass
from typing import Optional


@dataclass
class TrainingConfig:
    batch_size: int = 64
    epochs: int = 30
    learning_rate: float = 1e-3
    weight_decay: float = 1e-4
    num_workers: int = 4
    pin_memory: bool = True
    early_stopping_patience: int = 5
    scheduler_factor: float = 0.5
    scheduler_patience: int = 3
    dropout: float = 0.3
    head_hidden_dim: int = 512
    freeze_backbone: bool = True
    compile_model: bool = False
    device: str = "cpu"

    @classmethod
    def from_config_dict(
        cls, cfg: dict, device: str = "cpu", phase: str = "phase1"
    ) -> "TrainingConfig":
        t = cfg["training"]
        m = cfg["model"]
        p = t.get(phase, {})
        return cls(
            batch_size=t.get("batch_size", 64),
            epochs=p.get("epochs", t.get("epochs", 30)),
            learning_rate=p.get("learning_rate", t.get("learning_rate", 1e-3)),
            weight_decay=t.get("weight_decay", 1e-4),
            num_workers=t.get("num_workers", 4),
            pin_memory=t.get("pin_memory", True),
            early_stopping_patience=t.get("early_stopping_patience", 5),
            scheduler_factor=t.get("scheduler_factor", 0.5),
            scheduler_patience=t.get("scheduler_patience", 3),
            dropout=t.get("dropout", 0.3),
            head_hidden_dim=m.get("head_hidden_dim", 512),
            freeze_backbone=(phase == "phase1"),
            compile_model=t.get("compile", False),
            device=device,
        )

    @classmethod
    def phase2_config(
        cls, cfg: dict, device: str = "cpu"
    ) -> Optional["TrainingConfig"]:
        if not cfg.get("training", {}).get("fine_tune", False):
            return None
        return cls.from_config_dict(cfg, device, phase="phase2")
