"""The population benchmark: many enrollees, leave-one-session-out, speakers split once into norm / dev / test.

Offline: synthetic embeddings only.
"""

import json

import numpy as np
import pytest

from kavach.eval.population import LIMITS, PopSpeaker, render, run_population, split_population

D = 24


def make_pop(seed=0, n=24, sessions=3, whole=4, probes=6, noise=0.5, shift=0.3, sources=("libri-dev", "libri-test"), single=0):
    rng = np.random.default_rng(seed)
    out = []
    for i in range(n):
        centre = rng.normal(size=D)
        n_sess = 1 if i < single else sessions
        w, p = {}, {}
        for s in range(n_sess):
            off = rng.normal(size=D) * shift
            w[f"c{s}"] = np.array([centre * 2 + off + rng.normal(size=D) * noise * 2 for _ in range(whole)])
            p[f"c{s}"] = np.array([centre * 2 + off + rng.normal(size=D) * noise * 2 for _ in range(probes)])
        out.append(PopSpeaker(f"spk{i:02d}", sources[i % len(sources)], w, p))
    return out


def test_split_is_disjoint_complete_stratified_and_seeded():
    pop = make_pop()
    a = split_population(pop, 7)
    ids = [x for p in a.values() for x in p]
    assert sorted(ids) == sorted(s.id for s in pop) and len(ids) == len(set(ids))
    for part in a.values():
        assert {s.source for s in pop if s.id in part} == {"libri-dev", "libri-test"}
    assert split_population(pop, 7) == a and split_population(pop, 8) != a


def test_every_enrollee_is_scored_against_other_speakers_only_and_never_themselves():
    pop = make_pop(n=24, sessions=3, probes=6)
    r = run_population(pop, variants=["baseline"], n_boot=20)
    d = r.to_dict()
    parts = d["protocol"]["split"]
    by = {s.id: s for s in pop}
    for part in ("dev", "test"):
        n_probes_part = sum(sum(len(v) for v in by[i].probes.values()) for i in parts[part])
        for e in r.scores["baseline"][part]:
            own = sum(len(v) for v in by[e].probes.values())
            folds = len(by[e].probes)
            assert len(r.scores["baseline"][part][e]["impostor"]) == folds * (n_probes_part - own)  # never their own probes
            assert len(r.scores["baseline"][part][e]["genuine"]) == own  # every session is held out once


def test_a_held_out_session_never_shapes_its_own_template():
    pop = make_pop()
    base = run_population(pop, variants=["baseline"], n_boot=20)
    part = "test"
    e = sorted(base.scores["baseline"][part])[0]
    # change the WHOLE-clip embeddings of one session of e: only folds that enrol on it move; the held-out one must not
    pop2 = [PopSpeaker(s.id, s.source, {**s.whole, "c0": s.whole["c0"] + 5.0} if s.id == e else s.whole, s.probes) for s in pop]
    alt = run_population(pop2, variants=["baseline"], n_boot=20)
    n0 = len(pop[[s.id for s in pop].index(e)].probes["c0"])
    g0, g1 = base.scores["baseline"][part][e]["genuine"][:n0], alt.scores["baseline"][part][e]["genuine"][:n0]
    np.testing.assert_allclose(g0, g1, atol=1e-12)  # c0 held out: its own whole clips are not in its template
    assert not np.allclose(base.scores["baseline"][part][e]["genuine"][n0:], alt.scores["baseline"][part][e]["genuine"][n0:])


def test_the_threshold_never_sees_test_speakers():
    pop = make_pop()
    base = run_population(pop, variants=["baseline"], n_boot=20).to_dict()
    test_ids = set(base["protocol"]["split"]["test"])
    rng = np.random.default_rng(5)
    pop2 = [PopSpeaker(s.id, s.source, {k: rng.normal(size=v.shape) for k, v in s.whole.items()},
                       {k: rng.normal(size=v.shape) for k, v in s.probes.items()}) if s.id in test_ids else s for s in pop]
    alt = run_population(pop2, variants=["baseline"], n_boot=20).to_dict()
    assert base["variants"]["baseline"]["thresholds"] == alt["variants"]["baseline"]["thresholds"]


def test_same_seed_same_numbers_and_the_limits_lead():
    a = run_population(make_pop(), seed=3, n_boot=30)
    b = run_population(make_pop(), seed=3, n_boot=30)
    assert json.dumps(a.to_dict(), sort_keys=True) == json.dumps(b.to_dict(), sort_keys=True)
    assert render(a).startswith(LIMITS) and a.to_dict()["limits"] == LIMITS


def test_clusters_are_enrollees_so_the_interval_is_informative_with_many():
    d = run_population(make_pop(n=40), variants=["baseline"], n_boot=50).to_dict()
    assert d["variants"]["baseline"]["test"]["n_enrollees"] >= 5
    assert "informative" in render(run_population(make_pop(n=40), variants=["baseline"], n_boot=50)).lower()


def test_refuses_when_there_are_too_few_multi_session_enrollees():
    with pytest.raises(ValueError, match="enrollee"):
        run_population(make_pop(n=24, single=24), n_boot=20)
    with pytest.raises(ValueError, match="non-finite"):
        bad = make_pop()
        bad[0].probes["c0"][0, 0] = np.nan
        run_population(bad, n_boot=20)
    with pytest.raises(ValueError, match="unknown variant"):
        run_population(make_pop(), variants=["x"], n_boot=20)


def test_separable_gives_near_zero_eer_and_identical_gives_about_half():
    easy = run_population(make_pop(noise=0.1, shift=0.1), variants=["baseline"], n_boot=30).to_dict()
    assert easy["variants"]["baseline"]["test"]["eer"] < 0.02
    rng = np.random.default_rng(1)
    pop = [PopSpeaker(s.id, s.source, {k: rng.normal(size=v.shape) for k, v in s.whole.items()},
                      {k: rng.normal(size=v.shape) for k, v in s.probes.items()}) for s in make_pop()]
    hard = run_population(pop, variants=["baseline"], n_boot=30).to_dict()
    assert 0.3 < hard["variants"]["baseline"]["test"]["eer"] < 0.7


def test_the_probe_length_artefact_and_the_enrolment_curve_are_reported():
    d = run_population(make_pop(), variants=["baseline"], n_boot=20, budgets=(1, 2, None)).to_dict()
    art = d["artefact"]
    assert set(art) == {"matched_probes", "matched_whole", "mismatched"}
    assert [b["clips"] for b in d["budget"]] == [1, 2, "all"]
    assert all(np.isfinite(b["eer"]) for b in d["budget"])
