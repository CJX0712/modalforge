"""Persist / load an AlignedCorpus to disk (npz for images, json for captions)."""

from __future__ import annotations

import json
import os

import numpy as np

from ..core.errors import E200DataError
from ..core.types import AlignedCorpus


def save_corpus(corpus: AlignedCorpus, path: str) -> None:
    """Save a corpus to ``<path>`` (a directory)."""
    os.makedirs(path, exist_ok=True)
    np.save(os.path.join(path, "images.npy"), corpus.images)
    with open(os.path.join(path, "captions.json"), "w", encoding="utf-8") as fh:
        json.dump(
            {"item_ids": corpus.item_ids, "captions": corpus.captions},
            fh,
            ensure_ascii=False,
            indent=2,
        )


def load_corpus(path: str) -> AlignedCorpus:
    """Load a corpus previously written by :func:`save_corpus`."""
    img_path = os.path.join(path, "images.npy")
    cap_path = os.path.join(path, "captions.json")
    if not (os.path.exists(img_path) and os.path.exists(cap_path)):
        raise E200DataError(f"corpus path incomplete: {path}")
    images = np.load(img_path)
    with open(cap_path, "r", encoding="utf-8") as fh:
        meta = json.load(fh)
    return AlignedCorpus(
        item_ids=list(meta["item_ids"]),
        images=images,
        captions=list(meta["captions"]),
    )
