"""End-to-end demonstration: generate corpus -> train aligners -> benchmark.

Produces ``benchmark.json`` and prints the cross-aligner comparison table. Fully
offline: synthetic images are rendered with PIL, no network or model downloads.
"""

from __future__ import annotations

import os
import sys

# allow running as a script: ensure repo root is importable.
# run_demo.py lives at <repo>/modalforge/examples/run_demo.py, so the repo root
# is three levels up (examples -> modalforge pkg -> repo root).
_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from modalforge.core.config import config_from_env  # noqa: E402
from modalforge.pipeline.pipeline import ModalForgePipeline  # noqa: E402


def main() -> int:
    cfg = config_from_env()
    out_dir = os.path.join(_REPO_ROOT, "examples")
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, "benchmark.json")

    print("=" * 72)
    print("ModalForge · 多模态跨模态检索演示")
    print(f"seed={cfg.seed}  n_train={cfg.n_train}  n_test={cfg.n_test}")
    print("=" * 72)

    pipe = ModalForgePipeline(cfg)
    result = pipe.run()
    print()
    print(result.to_table())

    result.to_benchmark_json(out_path)
    print()
    print(f"✅ benchmark.json -> {out_path}")
    print(f"   对齐器已用: {', '.join(result.aligners_used)}")

    # highlight the headline comparison
    by_name = {r.aligner: r for r in result.rows}
    rb = by_name.get("random_baseline")
    mm = by_name.get("mmfuse")
    if rb and mm:
        lift = (mm.recall_at_1_img2txt - rb.recall_at_1_img2txt) / max(
            rb.recall_at_1_img2txt, 1e-9
        )
        print(f"   MMFuse R@1(i2t) 相对随机基线提升: {lift * 100:.1f}%")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
