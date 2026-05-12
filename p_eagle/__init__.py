"""P-EAGLE: Production-grade Parallel Speculative Decoding Framework"""

__version__ = "1.0.0"

from .models import EagleDrafterModel, EagleMTPHead

__all__ = [
    "EagleDrafterModel",
    "EagleMTPHead",
]
