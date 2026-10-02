"""Command-line entry point for ModalForge.

Usage:
    python -m modalforge.cli --seed 20261002 --out benchmark.json
"""

from __future__ import annotations

import argparse
import sys

from .core.config import config_from_env
from .pipeline.pipeline import ModalForgePipeline


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(
        prog="modalforge", description="Multimodal cross-modal retrieval benchmark"
    )
    p.add_argument("--seed", type=int, default=None)
    p.add_argument("--n-train", type=int, default=None)
    p.add_argument("--n-test", type=int, default=None)
    p.add_argument("--out", type=str, default="benchmark.json")
    p.add_argument("--quiet", action="store_true")
    args = p.parse_args(argv)

    cfg = config_from_env()
    if args.seed is not None:
        cfg.seed = args.seed
    if args.n_train is not None:
        cfg.n_train = args.n_train
    if args.n_test is not None:
        cfg.n_test = args.n_test

    pipe = ModalForgePipeline(cfg)
    result = pipe.run()
    if not args.quiet:
        print(result.to_table())
        print()
        best = max(
            (r for r in result.rows if r.available and r.aligner != "random_baseline"),
            key=lambda r: r.recall_at_1_img2txt + r.recall_at_1_txt2img,
            default=None,
        )
        if best:
            print(
                f"best aligner: {best.aligner} (R@1 i2t={best.recall_at_1_img2txt:.3f}, "
                f"R@1 t2i={best.recall_at_1_txt2img:.3f})"
            )

    payload = result.to_benchmark_json(args.out)
    if not args.quiet:
        print(f"benchmark written -> {args.out} ({len(payload['results'])} rows)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
