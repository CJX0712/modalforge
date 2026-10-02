"""Image encoder: HOG (shape) + per-cell mean RGB (colour), pure numpy.

Output dimension = ``hog_cells^2 * (hog_bins + 3)``. Deterministic, no fit state
beyond the configuration, so ``fit`` is a no-op and ``transform`` is stateless.
"""

from __future__ import annotations

import numpy as np

from ..core.config import Config
from ..core.errors import E300EncoderError


def _hog_block(gray: np.ndarray, cells: int, bins: int) -> np.ndarray:
    """Compute a (cells*cells*bins,) HOG descriptor for one grayscale image."""
    h, w = gray.shape
    # gradients via central differences
    gx = np.zeros_like(gray, dtype=np.float64)
    gy = np.zeros_like(gray, dtype=np.float64)
    gx[:, 1:-1] = (gray[:, 2:].astype(np.float64) - gray[:, :-2].astype(np.float64)) / 2.0
    gy[1:-1, :] = (gray[2:, :].astype(np.float64) - gray[:-2, :].astype(np.float64)) / 2.0
    mag = np.sqrt(gx**2 + gy**2)
    ang = (np.arctan2(gy, gx) * 180.0 / np.pi) % 180.0  # 0..180

    ch, cw = h // cells, w // cells
    hist = np.zeros((cells, cells, bins), dtype=np.float64)
    bin_width = 180.0 / bins
    for i in range(cells):
        for j in range(cells):
            m = mag[i * ch : (i + 1) * ch, j * cw : (j + 1) * cw]
            a = ang[i * ch : (i + 1) * ch, j * cw : (j + 1) * cw]
            # soft-assign to two neighbouring orientation bins
            b0 = (a / bin_width).astype(int) % bins
            b1 = (b0 + 1) % bins
            frac = a / bin_width - (a / bin_width).astype(int)
            np.add.at(hist[i, j], b0, m * (1 - frac))
            np.add.at(hist[i, j], b1, m * frac)
    desc = hist.reshape(-1)
    # L2 normalize the whole descriptor (contrast invariance)
    n = np.linalg.norm(desc)
    if n > 1e-8:
        desc = desc / n
    return desc


def _color_block(rgb: np.ndarray, cells: int) -> np.ndarray:
    """Per-cell mean RGB, flattened -> captures colour concepts."""
    h, w, _ = rgb.shape
    ch, cw = h // cells, w // cells
    out = np.zeros((cells, cells, 3), dtype=np.float64)
    for i in range(cells):
        for j in range(cells):
            out[i, j] = (
                rgb[i * ch : (i + 1) * ch, j * cw : (j + 1) * cw].reshape(-1, 3).mean(axis=0)
            )
    return out.reshape(-1) / 255.0


class HogImageEncoder:
    """Shape + colour image encoder (numpy only)."""

    name = "hog+rgb"

    def __init__(self, cfg: Config | None = None) -> None:
        self.cfg = cfg or Config()

    def fit(self, images: np.ndarray) -> "HogImageEncoder":
        self._validate(images)
        return self

    def transform(self, images: np.ndarray) -> np.ndarray:
        self._validate(images)
        cells = self.cfg.hog_cells
        bins = self.cfg.hog_bins
        out = np.zeros((images.shape[0], cells * cells * (bins + 3)), dtype=np.float32)
        for k, img in enumerate(images):
            rgb = img.astype(np.float64)
            if rgb.ndim == 2:  # grayscale input
                gray = rgb
                rgb3 = np.stack([rgb, rgb, rgb], axis=-1)
            else:
                gray = rgb.mean(axis=-1)
                rgb3 = rgb
            hog = _hog_block(gray, cells, bins)
            col = _color_block(rgb3, cells)
            out[k] = np.concatenate([hog, col]).astype(np.float32)
        return out

    def fit_transform(self, images: np.ndarray) -> np.ndarray:
        return self.fit(images).transform(images)

    @staticmethod
    def _validate(images: np.ndarray) -> None:
        if images.ndim not in (3, 4):
            raise E300EncoderError(f"images must be (n,H,W) or (n,H,W,3), got {images.ndim}d")
        if images.ndim == 4 and images.shape[3] not in (1, 3):
            raise E300EncoderError("images channel must be 1 or 3")
