#!/usr/bin/env python3
"""
Metrics for P-EAGLE Training and Inference
"""

from dataclasses import dataclass
from typing import List, Dict
import numpy as np


@dataclass
class SpeculationResult:
    """Result of a speculative decoding step."""
    accepted_tokens: int
    draft_tokens: List[int]
    verified_tokens: List[int]
    acceptance_rate: float
    tree_size: int


@dataclass
class GenerationMetrics:
    """Metrics for generation performance."""
    total_tokens: int
    accepted_tokens: int
    target_forward_passes: int
    drafter_forward_passes: int
    mean_acceptance_length: float
    speedup: float
    wall_time: float

    def to_dict(self) -> Dict:
        return {
            "total_tokens": self.total_tokens,
            "accepted_tokens": self.accepted_tokens,
            "target_forward_passes": self.target_forward_passes,
            "drafter_forward_passes": self.drafter_forward_passes,
            "mean_acceptance_length": self.mean_acceptance_length,
            "speedup": self.speedup,
            "wall_time": self.wall_time,
            "acceptance_rate": self.accepted_tokens / max(self.total_tokens, 1),
            "tokens_per_second": self.total_tokens / max(self.wall_time, 0.001)
        }


class MetricsTracker:
    """Track metrics during training."""

    def __init__(self):
        self.metrics = {
            "train_loss": [],
            "val_loss": [],
            "learning_rate": [],
            "grad_norm": [],
            "acceptance_lengths": []
        }

    def log(self, key: str, value: float):
        if key not in self.metrics:
            self.metrics[key] = []
        self.metrics[key].append(value)

    def get_average(self, key: str, window: int = 100) -> float:
        if key not in self.metrics or not self.metrics[key]:
            return 0.0
        values = self.metrics[key][-window:]
        return np.mean(values)

    def get_summary(self) -> Dict:
        return {
            key: {
                "mean": np.mean(values) if values else 0.0,
                "std": np.std(values) if values else 0.0,
                "min": np.min(values) if values else 0.0,
                "max": np.max(values) if values else 0.0,
                "count": len(values)
            }
            for key, values in self.metrics.items()
        }
