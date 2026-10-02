"""ModalForge end-to-end pipeline.

Dependency chain (acyclic, downward-only):
``pipeline -> {data, encoders, alignment, retrieval, fusion} -> core``.

The pipeline:
1. generates (or accepts) a train/test aligned corpus,
2. encodes images (HOG+RGB) and text (TF-IDF),
3. fits every available aligner on *train* and projects *test*,
4. evaluates each aligner with symmetric cross-modal retrieval metrics,
5. fuses the fitted spaces with MMFuse,
6. compares against a random baseline and returns a benchmark table.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np

from ..alignment.base import build_aligners
from ..core.config import Config, config_from_env
from ..core.types import AlignedCorpus, BenchmarkRow
from ..data.synthetic import generate_train_test
from ..encoders.image_encoder import HogImageEncoder
from ..encoders.text_encoder import TfidfEncoder
from ..fusion.mmfuse import MMFuse
from ..retrieval.retriever import cosine_sim, retrieval_metrics


@dataclass
class PipelineResult:
    rows: List[BenchmarkRow]
    cfg: dict
    aligners_used: List[str]
    spaces: Dict[str, Tuple[np.ndarray, np.ndarray]] = field(default_factory=dict)
    test_corpus: Optional[AlignedCorpus] = None

    def to_table(self) -> str:
        header = (
            f"{'aligner':<18}{'avail':<6}{'R@1 i2t':<9}{'R@5 i2t':<9}"
            f"{'mAP i2t':<9}{'MRR i2t':<9}{'R@1 t2i':<9}{'R@5 t2i':<9}{'mAP t2i':<9}"
        )
        lines = [header, "-" * len(header)]
        for r in self.rows:
            d = r.as_dict()
            lines.append(
                f"{r.aligner:<18}{('Y' if r.available else 'N'):<6}"
                f"{d['recall@1_img2txt']:<9}{d['recall@5_img2txt']:<9}"
                f"{d['map_img2txt']:<9}{d['mrr_img2txt']:<9}"
                f"{d['recall@1_txt2img']:<9}{d['recall@5_txt2img']:<9}{d['map_txt2img']:<9}"
            )
        return "\n".join(lines)

    def to_benchmark_json(self, path: Optional[str] = None) -> dict:
        payload = {
            "system": "ModalForge",
            "author": "晨星",
            "config": self.cfg,
            "aligners_used": self.aligners_used,
            "random_floor_note": "random_baseline row is the expected random retrieval ceiling",
            "results": [r.as_dict() for r in self.rows],
        }
        if path:
            with open(path, "w", encoding="utf-8") as fh:
                json.dump(payload, fh, ensure_ascii=False, indent=2)
        return payload


class ModalForgePipeline:
    """Orchestrates the full cross-modal retrieval benchmark."""

    def __init__(self, cfg: Optional[Config] = None) -> None:
        self.cfg = cfg or config_from_env()

    def _metrics(self, Q: np.ndarray, C: np.ndarray, labels, ks) -> dict:
        sims = cosine_sim(Q, C)
        kk = min(max(ks), sims.shape[1])
        order = np.argsort(-sims, axis=1)[:, :kk]
        return retrieval_metrics(order, labels, labels, ks)

    def run(
        self,
        train: Optional[AlignedCorpus] = None,
        test: Optional[AlignedCorpus] = None,
    ) -> PipelineResult:
        cfg = self.cfg
        if train is None or test is None:
            train, test = generate_train_test(cfg)

        img_enc = HogImageEncoder(cfg)
        txt_enc = TfidfEncoder(cfg)
        Xtr_img = img_enc.fit_transform(train.images)
        Xtr_txt = txt_enc.fit_transform(train.captions)
        Xte_img = img_enc.transform(test.images)
        Xte_txt = txt_enc.transform(test.captions)
        # Exact-pair retrieval: every query has exactly ONE ground-truth item (its
        # own paired counterpart in the test split). We use positional indices as
        # labels so correctness == "retrieved the paired item", which is the strict,
        # non-saturated definition for measuring cross-modal alignment recovery.
        labels = list(range(len(test.captions)))
        ks = cfg.recall_ks
        k = max(ks)

        spaces: Dict[str, Tuple[np.ndarray, np.ndarray]] = {}
        rows: List[BenchmarkRow] = []

        for al in build_aligners(cfg):
            if not al.available():
                rows.append(
                    BenchmarkRow(
                        al.name,
                        False,
                        0,
                        0,
                        0,
                        0,
                        0,
                        0,
                        0,
                        0,
                        notes="backend unavailable (offline / missing deps)",
                    )
                )
                continue
            try:
                al.fit(Xtr_img, Xtr_txt)
                _ = al.transform(Xtr_img, Xtr_txt)
                Zx_te, Zy_te = al.transform(Xte_img, Xte_txt)
            except Exception as exc:  # pragma: no cover - defensive
                rows.append(
                    BenchmarkRow(
                        al.name,
                        False,
                        0,
                        0,
                        0,
                        0,
                        0,
                        0,
                        0,
                        0,
                        notes=f"fit/transform failed: {exc!r}",
                    )
                )
                continue
            spaces[al.name] = (Zx_te, Zy_te)
            m_i = self._metrics(Zx_te, Zy_te, labels, ks)
            m_t = self._metrics(Zy_te, Zx_te, labels, ks)
            rows.append(
                BenchmarkRow(
                    al.name,
                    True,
                    m_i["recall@k"][1],
                    m_i["recall@k"][5],
                    m_i["map"],
                    m_i["mrr"],
                    m_t["recall@k"][1],
                    m_t["recall@k"][5],
                    m_t["map"],
                    m_t["mrr"],
                    notes="ok",
                )
            )

        # random baseline
        nq = len(labels)
        nc = Xte_txt.shape[0]
        rng = np.random.default_rng(cfg.seed + 12345)
        rand_order = np.array([rng.permutation(nc)[:k] for _ in range(nq)])
        rm_i = retrieval_metrics(rand_order, labels, labels, ks)
        rm_t = retrieval_metrics(rand_order, labels, labels, ks)
        rows.append(
            BenchmarkRow(
                "random_baseline",
                True,
                rm_i["recall@k"][1],
                rm_i["recall@k"][5],
                rm_i["map"],
                rm_i["mrr"],
                rm_t["recall@k"][1],
                rm_t["recall@k"][5],
                rm_t["map"],
                rm_t["mrr"],
                notes="expected random retrieval ceiling",
            )
        )

        # MMFuse flagship
        if spaces:
            mm = MMFuse(cfg.fusion_min_conf, cfg.fusion_topk)
            res_i = mm.retrieve(spaces, "img2txt", k)
            res_t = mm.retrieve(spaces, "txt2img", k)
            fm_i = retrieval_metrics(res_i.ranks, labels, labels, ks)
            fm_t = retrieval_metrics(res_t.ranks, labels, labels, ks)
            rows.append(
                BenchmarkRow(
                    "mmfuse",
                    True,
                    fm_i["recall@k"][1],
                    fm_i["recall@k"][5],
                    fm_i["map"],
                    fm_i["mrr"],
                    fm_t["recall@k"][1],
                    fm_t["recall@k"][5],
                    fm_t["map"],
                    fm_t["mrr"],
                    notes="confidence-gated fusion of " + ", ".join(spaces.keys()),
                )
            )

        return PipelineResult(
            rows=rows,
            cfg=cfg.as_dict(),
            aligners_used=list(spaces.keys()),
            spaces=spaces,
            test_corpus=test,
        )
