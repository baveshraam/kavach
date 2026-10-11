"""The three pretrained speaker embedders behind one constructor, and where each one's cohort cache lives."""

from pathlib import Path

import numpy as np
import pytest

from kavach.audio import Audio
from kavach.embedding import ECAPAEmbedder
from kavach.eval.embedders import EMBEDDERS, WavLMSVEmbedder, cohort_dir, make_embedder


def test_registry_names_and_construction_is_lazy():
    assert list(EMBEDDERS) == ["ecapa", "resnet", "wavlm"]
    e = make_embedder("ecapa", device="cpu")
    r = make_embedder("resnet", device="cpu")
    w = make_embedder("wavlm", device="cpu")
    assert isinstance(e, ECAPAEmbedder) and e.model_name == "speechbrain/spkrec-ecapa-voxceleb"
    assert isinstance(r, ECAPAEmbedder) and r.model_name == "speechbrain/spkrec-resnet-voxceleb"
    assert isinstance(w, WavLMSVEmbedder) and w.model_name == "microsoft/wavlm-base-plus-sv"
    assert e._model is None and r._model is None and w._model is None  # nothing was loaded


def test_unknown_embedder_is_refused_with_the_known_names():
    with pytest.raises(ValueError, match="ecapa, resnet, wavlm"):
        make_embedder("nope")


def test_ecapa_keeps_its_existing_cache_directory_and_the_others_get_their_own(tmp_path):
    assert cohort_dir(tmp_path, "ecapa") == tmp_path / "cohort" / "emb"
    assert cohort_dir(tmp_path, "resnet") == tmp_path / "cohort" / "emb_resnet"
    assert cohort_dir(tmp_path, "wavlm") == tmp_path / "cohort" / "emb_wavlm"


def test_wavlm_refuses_a_clip_too_short_and_returns_a_normalised_vector(monkeypatch):
    w = WavLMSVEmbedder(device="cpu")

    class FakeModel:
        def __call__(self, **kw):
            class Out:
                embeddings = __import__("torch").ones(1, 512) * 3.0

            return Out()

    class FakeFE:
        def __call__(self, samples, sampling_rate, return_tensors):
            import torch

            return {"input_values": torch.zeros(1, len(samples))}

    monkeypatch.setattr(w, "_load", lambda: (FakeFE(), FakeModel()))
    rng = np.random.default_rng(0)
    ok = Audio((rng.normal(size=32000) * 0.1).astype(np.float32), 16000, "t")
    v = w.embed(ok)
    assert v.vector.shape == (512,) and abs(np.linalg.norm(v.vector) - 1) < 1e-9
    from kavach.audio import AudioError

    with pytest.raises(AudioError):
        w.embed(Audio((rng.normal(size=4000) * 0.1).astype(np.float32), 16000, "short"))


def test_a_whole_clip_cohort_file_lists_the_audio_to_embed_again(tmp_path):
    from kavach.cohort import enumerate_cached

    f = tmp_path / "k.npz"
    np.savez(f, vec=np.ones((2, 4)), speaker=np.array(["S01", "S02"]), session=np.array(["s1", "s1"]),
             path=np.array(["a.wav", "b.wav"]), dur=np.array([5.0, 6.0]))
    items = enumerate_cached(f)
    assert [(i.speaker, i.path) for i in items] == [("S01", "a.wav"), ("S02", "b.wav")]
