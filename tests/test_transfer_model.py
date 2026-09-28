"""The transfer-learning Python model: bundle contract, extractor choice, embeddings."""

import json
from pathlib import Path

import numpy as np
import pytest

from feature_extraction import ast_embeddings, clap_embeddings, embeddings, ensemble_embeddings
from src.services.pipeline import ModelsUnavailable, _extractor_for_bundle

ROOT = Path(__file__).resolve().parents[1]
BEST = ROOT / "python_models" / "best"
HGB = ROOT / "python_models" / "archive" / "hgb_v1_2026-09-25"
HAVE_CNN14 = embeddings.default_checkpoint_path().exists()


def _served_version() -> str:
    return json.loads((BEST / "feature_config.json").read_text())["feature_version"]


def _served_backbones() -> list[str]:
    version = _served_version()
    if version.startswith(ensemble_embeddings.PREFIX):
        return ensemble_embeddings.backbones_from_version(version)
    return ["ast" if version.startswith("ast") else "cnn14"]


def _have_ast() -> bool:
    try:
        from huggingface_hub import try_to_load_from_cache
    except ImportError:
        return False
    hit = try_to_load_from_cache(ast_embeddings.AST_MODEL, "model.safetensors",
                                 revision=ast_embeddings.AST_REVISION)
    return isinstance(hit, str)


def _have_clap() -> bool:
    try:
        from huggingface_hub import try_to_load_from_cache
    except ImportError:
        return False
    hit = try_to_load_from_cache(clap_embeddings.CLAP_MODEL, "pytorch_model.bin",
                                 revision=clap_embeddings.CLAP_REVISION)
    return isinstance(hit, str)


HAVE_AST = _have_ast()
HAVE_WEIGHTS = {"ast": HAVE_AST, "clap": _have_clap(), "cnn14": HAVE_CNN14}
HAVE_SERVED = all(HAVE_WEIGHTS[b] for b in _served_backbones())


def test_served_bundle_declares_its_embedding_version_and_lineage():
    backbones = _served_backbones()
    features = json.loads((BEST / "feature_config.json").read_text())
    meta = json.loads((BEST / "model_meta.json").read_text())
    labels = json.loads((BEST / "label_encoder.json").read_text())["class_names"]
    classes = [c["name"] for c in json.loads((ROOT / "config/classes.json").read_text())["classes"]]
    assert features["feature_version"] == _served_version()
    assert features["columns"] == ensemble_embeddings.embedding_columns(backbones)
    assert sorted(labels) == sorted(classes)
    # The bundle records its training data and selection.
    assert "0.5*val_macro_f1" in meta["selection_criterion"]
    assert len(meta["train_ids_sha256"]) == 64
    assert meta["metrics"]["split"] == "test" and meta["metrics"]["n_records"] == 450


def test_extractor_follows_the_bundle(tmp_path):
    from feature_extraction.features import FeatureExtractor

    extractor = _extractor_for_bundle(BEST)
    if len(_served_backbones()) > 1:
        assert isinstance(extractor, ensemble_embeddings.EnsembleFeatureExtractor)
        assert extractor.backbones == _served_backbones()
    else:
        expected = (ast_embeddings.AstFeatureExtractor if _served_backbones() == ["ast"]
                    else embeddings.EmbeddingFeatureExtractor)
        assert isinstance(extractor, expected)
    if HGB.exists():
        assert isinstance(_extractor_for_bundle(HGB), FeatureExtractor)
    for stale_version in ("panns-old-0.1", "ast-old-0.1", "ensemble:ast-old-0.1+clap-old-0.1"):
        stale = tmp_path / stale_version
        stale.mkdir()
        (stale / "feature_config.json").write_text(json.dumps({"feature_version": stale_version}))
        with pytest.raises(ModelsUnavailable):
            _extractor_for_bundle(stale)


def test_ast_chunks_cover_the_clip_without_a_tiny_tail():
    sr = ast_embeddings.SAMPLE_RATE
    assert len(ast_embeddings.chunks(np.zeros(sr * 3))) == 1
    pieces = ast_embeddings.chunks(np.zeros(int(sr * 20.8)))   # 10.24 + 10.24 + 0.32 s
    assert len(pieces) == 2 and sum(len(p) for p in pieces) == int(sr * 20.8)


@pytest.mark.skipif(not HAVE_CNN14, reason="PANNs checkpoint not downloaded")
def test_cnn14_embeddings_are_deterministic_and_handle_short_clips():
    rng = np.random.default_rng(0)
    y = rng.normal(0, 0.1, 16000 * 2).astype(np.float32)
    emb = embeddings.shared_embedder()
    a, b = emb.embed(y), emb.embed(y)
    assert a.shape == (embeddings.EMBEDDING_DIM,)
    assert np.allclose(a, b)
    short = emb.embed(y[:3000])            # 0.19 s, padded to one second
    assert short.shape == (embeddings.EMBEDDING_DIM,) and np.isfinite(short).all()


@pytest.mark.skipif(not HAVE_AST, reason="AST weights not in the Hugging Face cache")
def test_ast_embeddings_are_deterministic_and_handle_short_clips():
    rng = np.random.default_rng(0)
    y = rng.normal(0, 0.1, 16000 * 2).astype(np.float32)
    emb = ast_embeddings.shared_embedder()
    a, b = emb.embed(y), emb.embed(y)
    assert a.shape == (ast_embeddings.EMBEDDING_DIM,)
    assert np.allclose(a, b)
    short = emb.embed(y[:3000])
    assert short.shape == (ast_embeddings.EMBEDDING_DIM,) and np.isfinite(short).all()


@pytest.mark.skipif(not HAVE_SERVED, reason="pretrained weights for the served backbone not present")
def test_predictor_returns_plain_strings_and_all_ten_scores():
    from audio_preprocessing.pipeline import AudioPipeline
    from src.inference.predictor import PythonModelPredictor

    predictor = PythonModelPredictor.load(BEST, _extractor_for_bundle(BEST).extract)
    assert all(type(c) is str for c in predictor.class_names)
    t = np.arange(16000 * 2) / 16000
    siren = (0.5 * np.sin(2 * np.pi * (700 + 300 * np.sin(2 * np.pi * 1.5 * t)) * t)).astype(np.float32)
    pre = AudioPipeline().preprocess_samples(siren, 16000, origin="upload")
    result = predictor.predict_from_preprocessed(pre)
    assert type(result.predicted_class) is str
    assert len(result.confidences) == 10
    assert abs(sum(result.confidences.values()) - 1.0) < 1e-6


def test_embed_passes_the_cached_sample_rate(tmp_path, monkeypatch):
    """CLAP defaults to 48 kHz; the cache is 16 kHz, so the rate must be passed on."""
    import python_models.train_transfer as trainer

    seen = []

    class Recorder:
        def embed(self, samples, sample_rate=48000):
            seen.append(sample_rate)
            return np.zeros(4)

    monkeypatch.setattr(trainer, "make_embedder", lambda backbone: Recorder())
    monkeypatch.setattr(trainer, "ROOT", tmp_path)
    monkeypatch.setattr(trainer, "emb_dir", lambda variant, backbone="cnn14": tmp_path / "emb")
    monkeypatch.setattr(trainer, "read_index",
                        lambda variant: {"A1": {"sample_rate": "16000", "rejected": "False"}})
    monkeypatch.setattr(trainer, "load_manifest", lambda: [{"audio_id": "A1", "dataset_split": "train"}])
    monkeypatch.setattr(trainer, "load_cached", lambda audio_id, variant: np.zeros(16000, np.float32))
    trainer.embed("current", ["train"], backbone="clap")
    assert seen == [16000]


def test_ensemble_averages_members_on_their_own_columns():
    from src.inference.ensemble import EnsembleMember, SoftVotingEnsemble

    class Fixed:
        """Stand-in classifier: returns a fixed score list and checks its input width."""

        def __init__(self, classes, scores, width):
            self.classes_ = np.asarray(classes)
            self.scores, self.width = np.asarray(scores), width

        def predict_proba(self, X):
            assert X.shape[1] == self.width
            return np.tile(self.scores, (len(X), 1))

    classes = ["A", "B"]
    ensemble = SoftVotingEnsemble([
        EnsembleMember("first", Fixed(classes, [0.9, 0.1], 3), 0, 3),
        EnsembleMember("second", Fixed(["B", "A"], [0.7, 0.3], 2), 3, 5),  # other class order
    ], classes)
    proba = ensemble.predict_proba(np.zeros((1, 5)))
    assert ensemble.n_features_in_ == 5
    assert np.allclose(proba, [[0.6, 0.4]])
    assert list(ensemble.predict(np.zeros((1, 5)))) == ["A"]
