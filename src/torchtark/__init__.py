"""PyTorch Tail-Averaged Randomized Kaczmarz."""

from .solver import tark
from .batched import tark_batched

__all__ = ["tark", "tark_batched"]
