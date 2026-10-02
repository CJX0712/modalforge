"""End-to-end pipeline tests: offline fallback, beats random, deterministic."""

from modalforge.core.config import Config
from modalforge.pipeline.pipeline import ModalForgePipeline


def _small_cfg():
    return Config(
        seed=20261002,
        n_train=160,
        n_test=60,
        cca_dims=8,
        contrastive_dim=8,
        contrastive_epochs=8,
        recall_ks=(1, 5, 10),
    )


def test_pipeline_runs_offline():
    cfg = _small_cfg()
    result = ModalForgePipeline(cfg).run()
    names = {r.aligner for r in result.rows}
    # core offline aligners must be present
    assert "numpy_cca" in names
    assert "numpy_contrastive" in names
    assert "mmfuse" in names
    assert "random_baseline" in names


def test_mmfuse_beats_random():
    cfg = _small_cfg()
    result = ModalForgePipeline(cfg).run()
    by = {r.aligner: r for r in result.rows}
    rb = by["random_baseline"]
    mm = by["mmfuse"]
    assert mm.available and rb.available
    # cross-modal retrieval should clearly beat the random ceiling
    assert mm.recall_at_1_img2txt > rb.recall_at_1_img2txt
    assert mm.recall_at_1_txt2img > rb.recall_at_1_txt2img


def test_pipeline_deterministic():
    cfg = _small_cfg()
    r1 = ModalForgePipeline(cfg).run()
    r2 = ModalForgePipeline(cfg).run()
    b1 = {r.aligner: r for r in r1.rows}
    b2 = {r.aligner: r for r in r2.rows}
    for name in ("numpy_cca", "mmfuse", "random_baseline"):
        assert abs(b1[name].recall_at_1_img2txt - b2[name].recall_at_1_img2txt) < 1e-9


def test_benchmark_json_roundtrip(tmp_path):
    cfg = _small_cfg()
    result = ModalForgePipeline(cfg).run()
    out = tmp_path / "benchmark.json"
    payload = result.to_benchmark_json(str(out))
    assert isinstance(payload["results"], list)
    assert any(r["aligner"] == "mmfuse" for r in payload["results"])
    assert out.exists()
