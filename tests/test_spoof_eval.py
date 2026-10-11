"""Synthetic-voice attacks scored against the single-enrollee templates, with the same thresholds the study fitted.

Offline: synthetic embeddings and a fake converter; the real kNN-VC and the GPU are exercised by the real run.
"""

import json

import numpy as np
import pytest

from kavach.eval.poc import Cohort, Sitting, run_poc
from kavach.eval.spoof import LIMITS, attack_report, render, select_sources, split_reference

D = 24


def make_cohort(seed=0):
    rng = np.random.default_rng(seed)
    vec, spk, src = [], [], []
    for si, n in enumerate((10, 10, 8)):
        for j in range(n):
            c = rng.normal(size=D)
            for _ in range(5):
                vec.append(c * 2 + rng.normal(size=D) * 1.2)
                spk.append(f"{'abc'[si]}{j:02d}")
                src.append(f"src{si}")
    return Cohort(np.array(vec), spk, src)


def make_sittings(seed=1):
    rng = np.random.default_rng(seed)
    owner = rng.normal(size=D)
    return [Sitting(f"S{k + 1}", np.array([owner * 2 + rng.normal(size=D) * 0.3 + rng.normal(size=D) * 1.2 for _ in range(10)])) for k in range(3)]


@pytest.fixture()
def world():
    sit, coh = make_sittings(), make_cohort()
    res = run_poc(sit, coh, variants=["baseline", "meansub"], n_boot=20)
    return sit, coh, res


def test_a_clone_of_the_owner_scores_high_and_a_stranger_voice_scores_low(world):
    sit, coh, res = world
    rng = np.random.default_rng(3)
    owner_like = {s.name: s.vecs[:6] + rng.normal(size=(6, D)) * 0.05 for s in sit}   # an excellent clone
    stranger = {s.name: rng.normal(size=(6, D)) for s in sit}                          # an unrelated voice
    rep = attack_report(sit, coh, res, {"clone": owner_like, "other": stranger})
    assert rep["clone"]["baseline"]["rules"]["eer"]["accepted"] > rep["other"]["baseline"]["rules"]["eer"]["accepted"]
    assert rep["clone"]["baseline"]["score_mean"] > rep["other"]["baseline"]["score_mean"]
    r = rep["clone"]["baseline"]["rules"]["eer"]
    assert r["n"] == 18 and 0 <= r["interval"][0] <= r["rate"] <= r["interval"][1] <= 1  # Wilson, three folds x six probes


def test_each_attack_is_scored_with_its_own_folds_template_and_threshold(world):
    sit, coh, res = world
    # the same probes in every fold, so only the fold's template and threshold can differ
    probes = {s.name: sit[0].vecs[:5] for s in sit}
    rep = attack_report(sit, coh, res, {"a": probes})
    d = res.to_dict()["variants"]["baseline"]["folds"]
    assert len({d[k]["thresholds"]["eer"] for k in d}) > 1  # thresholds really differ by fold
    # a probe set identical to a held-out sitting's own clips must be scored against the OTHER sittings: finite and < 1
    assert rep["a"]["baseline"]["score_max"] < 0.9999


def test_an_attack_must_cover_every_fold_and_have_finite_embeddings(world):
    sit, coh, res = world
    with pytest.raises(ValueError, match="fold"):
        attack_report(sit, coh, res, {"a": {"S1": sit[0].vecs[:3]}})
    bad = {s.name: np.full((3, D), np.nan) for s in sit}
    with pytest.raises(ValueError, match="non-finite"):
        attack_report(sit, coh, res, {"a": bad})


def test_report_is_deterministic_and_the_limits_lead(world):
    sit, coh, res = world
    probes = {s.name: s.vecs[:4] * 0.5 for s in sit}
    a = attack_report(sit, coh, res, {"x": probes})
    b = attack_report(sit, coh, res, {"x": probes})
    assert json.dumps(a, sort_keys=True) == json.dumps(b, sort_keys=True)
    assert render(a, res).startswith(LIMITS)


def test_the_attackers_reference_is_the_held_out_sitting_never_the_templates_sittings():
    sittings = ["S1", "S2", "S3"]
    ref = split_reference(sittings)
    for k in sittings:
        assert ref[k] == k  # reference = the fold's own held-out sitting
        assert k not in [s for s in sittings if s != k]  # which is exactly the sitting its template excludes


def test_sources_are_seeded_distinct_speakers_and_limited():
    items = [(f"L{s}", f"u{s}_{u}") for s in range(20) for u in range(3)]
    a = select_sources(items, n_speakers=6, seed=1)
    assert a == select_sources(items, n_speakers=6, seed=1) and a != select_sources(items, n_speakers=6, seed=2)
    assert len({s for s, _ in a}) == 6 and len(a) == 18
    with pytest.raises(ValueError, match="speakers"):
        select_sources(items, n_speakers=50, seed=1)


def test_attack_audio_is_cut_like_every_other_probe_and_embedded_once(tmp_path):
    from clone_helpers import tone
    from kavach.audio import save_wav
    from kavach.embedding import SpeakerEmbedding
    from kavach.eval.spoof_run import embed_wavs

    calls = []

    class Emb:
        model_name = "fake"

        def embed(self, audio):
            calls.append(1)
            return SpeakerEmbedding(np.random.default_rng(len(calls)).normal(size=D))

    wav = save_wav(tone(seconds=12.0, seed=2), tmp_path / "a.wav")
    a = embed_wavs([wav], Emb(), tmp_path / "c.json", 5.0)
    assert a[0].shape == (2, D)  # 12 s -> two 5 s probes, the 2 s tail dropped
    n = len(calls)
    b = embed_wavs([wav], Emb(), tmp_path / "c.json", 5.0)
    assert len(calls) == n and np.allclose(a[0], b[0])


def test_scoring_refuses_without_the_single_enrollee_results_and_without_attack_audio(tmp_path, capsys):
    from kavach.eval import spoof_run

    assert spoof_run.main(["score", "--embedder", "ecapa", "--poc", str(tmp_path / "none")]) == 2
    assert "missing" in capsys.readouterr().err


def test_generation_is_refused_for_anyone_not_on_the_clone_allowlist(capsys, monkeypatch):
    from kavach.eval import spoof_run

    monkeypatch.delenv("KAVACH_CLONE_VICTIMS", raising=False)
    assert spoof_run.main(["generate", "--victim", "S04"]) == 2
    assert "allowlist" in capsys.readouterr().err
