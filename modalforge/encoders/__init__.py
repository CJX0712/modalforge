"""Encoder package: image and text feature extractors.

Both encoders are pure ``numpy`` so the whole stack runs offline with zero heavy
dependencies. The image encoder captures *shape* (HOG) **and** *color* (per-cell
mean RGB) so colour concepts in the captions are recoverable from pixels.
"""

from .image_encoder import HogImageEncoder
from .text_encoder import TfidfEncoder

__all__ = ["HogImageEncoder", "TfidfEncoder"]
