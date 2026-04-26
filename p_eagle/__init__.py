"""P-EAGLE: Production-grade Parallel Speculative Decoding Framework"""

__version__ = "1.0.0"

from .data_preparation import EAGLEDistiller, run_eagle_distillation
from .models import EagleDrafterModel, EagleMTPHead, TreeAttentionMask
from .training import EagleTrainer
from .inference import PEAGLEInference
from .utils import (
    EagleDataset,
    align_segments_to_tokens,
    fuse_tri_layer_features,
    masked_mse_loss,
    GenerationMetrics,
    SpeculationResult
)

__all__ = [
    "EAGLEDistiller",
    "run_eagle_distillation",
    "EagleDrafterModel",
    "EagleMTPHead",
    "TreeAttentionMask",
    "EagleTrainer",
    "PEAGLEInference",
    "EagleDataset",
    "align_segments_to_tokens",
    "fuse_tri_layer_features",
    "masked_mse_loss",
    "GenerationMetrics",
    "SpeculationResult",
]
