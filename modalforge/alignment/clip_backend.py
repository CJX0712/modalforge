"""Optional CLIP backend (modern SOTA cross-modal alignment).

Gracefully unavailable when ``transformers``/``torch`` are absent (offline / CPU
environments). When present, CLIP's image and text encoders already live in a
shared space, so ``transform`` simply returns their normalised embeddings.
"""

from __future__ import annotations

import numpy as np

from ..core.errors import E400AlignmentError

try:  # pragma: no cover - only when transformers+torch installed
    import torch  # noqa: F401
    from transformers import CLIPModel, CLIPProcessor  # noqa: F401

    _CLIP_OK = True
except Exception:  # pragma: no cover
    _CLIP_OK = False


class ClipAligner:
    """CLIP-based cross-modal aligner (optional, network/weights required)."""

    name = "clip"

    def __init__(self, model_name: str = "openai/clip-vit-base-patch32") -> None:
        self.model_name = model_name
        self._model = None
        self._processor = None

    def available(self) -> bool:
        return _CLIP_OK

    def fit(self, X_img: np.ndarray, X_txt: np.ndarray) -> "ClipAligner":
        if not _CLIP_OK:
            raise E400AlignmentError("transformers/torch not available for CLIP")
        # CLIP is pre-trained; fit only loads the weights (may need network).
        self._processor = CLIPProcessor.from_pretrained(self.model_name)
        self._model = CLIPModel.from_pretrained(self.model_name)
        self._model.eval()
        return self

    def _encode_images(self, images: np.ndarray) -> np.ndarray:
        from PIL import Image as _PILImage

        pil = [_PILImage.fromarray(im.astype(np.uint8)) for im in images]
        inputs = self._processor(images=pil, return_tensors="pt")
        with _no_grad():
            out = self._model.get_image_features(**inputs)
        return _unit(out.detach().cpu().numpy().astype(np.float64))

    def _encode_texts(self, texts: list[str]) -> np.ndarray:
        inputs = self._processor(text=list(texts), return_tensors="pt", padding=True)
        with _no_grad():
            out = self._model.get_text_features(**inputs)
        return _unit(out.detach().cpu().numpy().astype(np.float64))

    def transform(self, X_img: np.ndarray, X_txt: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        if self._model is None:
            raise E400AlignmentError("ClipAligner.transform before fit")
        Zx = self._encode_images(X_img)
        Zy = self._encode_texts([str(t) for t in X_txt])
        return Zx.astype(np.float32), Zy.astype(np.float32)


def _unit(M: np.ndarray) -> np.ndarray:
    nrm = np.linalg.norm(M, axis=1, keepdims=True)
    nrm[nrm < 1e-12] = 1.0
    return M / nrm


def _no_grad():
    import torch

    return torch.no_grad()
