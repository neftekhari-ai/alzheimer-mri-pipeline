"""Shared configuration for MRI experiments."""
from dataclasses import dataclass
from pathlib import Path
import torch

@dataclass
class Config:
    data_dir: str = "data/manifest.csv"
    batch_size: int = 32
    epochs: int = 50
    lr: float = 3e-4
    temperature: float = 0.5
    num_classes: int = 3
    image_size: int = 224
    seed: int = 42
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    output_dir: str = "checkpoints"
    num_workers: int = 4

    def ensure_output_dir(self) -> Path:
        path = Path(self.output_dir)
        path.mkdir(parents=True, exist_ok=True)
        return path
