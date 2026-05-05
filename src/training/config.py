"""Training configuration dataclass."""

from dataclasses import dataclass, field


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
    device: str = "cpu"

    @classmethod
    def from_config_dict(cls, cfg: dict, device: str = "cpu") -> "TrainingConfig":
        t = cfg["training"]
        m = cfg["model"]
        return cls(
            batch_size=t.get("batch_size", 64),
            epochs=t.get("epochs", 30),
            learning_rate=t.get("learning_rate", 1e-3),
            weight_decay=t.get("weight_decay", 1e-4),
            num_workers=t.get("num_workers", 4),
            pin_memory=t.get("pin_memory", True),
            early_stopping_patience=t.get("early_stopping_patience", 5),
            scheduler_factor=t.get("scheduler_factor", 0.5),
            scheduler_patience=t.get("scheduler_patience", 3),
            dropout=m.get("dropout", 0.3),
            head_hidden_dim=m.get("head_hidden_dim", 512),
            freeze_backbone=m.get("freeze_backbone", True),
            device=device,
        )
