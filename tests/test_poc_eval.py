"""The proof-of-concept evaluation protocol: leakage, determinism, refusals, and agreement with the calibration.

Offline: synthetic embeddings only, no model, no GPU.
"""

import json

import numpy as np
import pytest

from kavach.calibrate_voice import loso_scores
from kavach.eval.poc import (
    LIMITS,
    VARIANTS,
    Cohort,
    Sitting,
    _eer_fast,
    run_poc,
    split_speakers,
)

D = 24


def _unit(m):
    m = np.asarray(m, float)
    return m / np.linalg.norm(m, axis=-1, keepdims=True)


def make_cohort(seed=0, per_source=(12, 12, 8), clips=6, noise=0.6, with_enrollee=False):
    rng = np.random.default_rng(seed)
    vec, spk, src = [], [], []
    for si, n in enumerate(per_source):
        for j in range(n):
            centre = rng.normal(size=D)
            for _ in range(clips):
                vec.append(centre * 2 + rng.normal(size=D) * noise * 2)
                spk.append(f"{'abc'[si]}{j:02d}")
                src.append(f"src{si}")
    if with_enrollee:
        vec.append(rng.normal(size=D)); spk.append("S04"); src.append("src0")
    return Cohort(np.array(vec), spk, src)


def make_sittings(seed=1, n=3, clips=12, noise=0.6, shift=0.3, same_as_impostor=False):
    rng = np.random.default_rng(seed)
    owner = rng.normal(size=D)
    out = []
    for k in range(n):
        off = rng.normal(size=D) * shift
        rows = []
        for _ in range(clips):
            centre = rng.normal(size=D) if same_as_impostor else owner
            rows.append(centre * 2 + off + rng.normal(size=D) * noise * 2)
        out.append(Sitting(f"S{k + 1}", np.array(rows)))
    return out


def test_split_is_disjoint_stratified_and_seeded():
    c = make_cohort()
    a = split_speakers(c, 7)
    assert not (a["norm"] & a["fit"]) and not (a["norm"] & a["test"]) and not (a["fit"] & a["test"])
    assert a["norm"] | a["fit"] | a["test"] == set(c.speaker)
    for src in ("src0", "src1", "src2"):  # every source reaches every part
        spk = {s for s, x in zip(c.speaker, c.source) if x == src}
        assert all(spk & a[p] for p in ("norm", "fit", "test"))
    assert split_speakers(c, 7) == a
    assert split_speakers(c, 8) != a


def test_enrollee_among_impostors_is_refused():
    with pytest.raises(ValueError, match="enrollee"):
        run_poc(make_sittings(), make_cohort(with_enrollee=True), n_boot=20)


def test_same_inputs_same_seed_give_identical_results():
    a = run_poc(make_sittings(), make_cohort(), seed=3, n_boot=50).to_dict()
    b = run_poc(make_sittings(), make_cohort(), seed=3, n_boot=50).to_dict()
    assert json.dumps(a, sort_keys=True) == json.dumps(b, sort_keys=True)
    c = run_poc(make_sittings(), make_cohort(), seed=4, n_boot=50).to_dict()
    assert c["protocol"]["split"] != a["protocol"]["split"]


def test_baseline_agrees_with_the_calibration_code():
    sit, coh = make_sittings(), make_cohort()
    r = run_poc(sit, coh, variants=["baseline"], n_boot=20)
    loso = loso_scores({s.name: s.vecs for s in sit}, coh.vec)
    test = set(r.to_dict()["protocol"]["split"]["test"])
    cols = np.array([s in test for s in coh.speaker])
    sc = r.scores["baseline"]
    for s in sit:
        np.testing.assert_allclose(sc["genuine"][s.name], loso.by_session[s.name], atol=1e-9)
    got = np.concatenate([sc["impostor"][spk] for spk in sorted(sc["impostor"])])
    expected = []
    for spk in sorted(sc["impostor"]):
        idx = np.array([i for i, x in enumerate(coh.speaker) if x == spk])
        expected.append(loso.impostor_by_fold[:, idx].ravel())
    np.testing.assert_allclose(np.sort(got), np.sort(np.concatenate(expected)), atol=1e-9)
    assert cols.sum() > 0


def test_a_fold_threshold_does_not_see_its_own_test_trials():
    sit, coh = make_sittings(), make_cohort()
    base = run_poc(sit, coh, variants=["baseline"], n_boot=20).to_dict()
    test = set(base["protocol"]["split"]["test"])
    # change every test impostor and the held-out sitting S1: the S1 fold's thresholds must not move
    rng = np.random.default_rng(99)
    vec = coh.vec.copy()
    for i, s in enumerate(coh.speaker):
        if s in test:
            vec[i] = rng.normal(size=D)
    sit2 = [Sitting("S1", rng.normal(size=sit[0].vecs.shape))] + sit[1:]
    alt = run_poc(sit2, Cohort(vec, coh.speaker, coh.source), variants=["baseline"], n_boot=20).to_dict()
    f0 = base["variants"]["baseline"]["folds"]["S1"]["thresholds"]
    f1 = alt["variants"]["baseline"]["folds"]["S1"]["thresholds"]
    assert f0 == f1
    f2 = base["variants"]["baseline"]["folds"]["S2"]["thresholds"]
    assert f2 != alt["variants"]["baseline"]["folds"]["S2"]["thresholds"]  # S2's fold does enrol/fit on S1


def test_refuses_instead_of_flattering():
    coh = make_cohort()
    sit = make_sittings()
    bad = [Sitting("S1", np.where(np.eye(len(sit[0].vecs), D) > 0, np.nan, sit[0].vecs))] + sit[1:]
    with pytest.raises(ValueError, match="non-finite"):
        run_poc(bad, coh, n_boot=20)
    with pytest.raises(ValueError, match="at least three"):
        run_poc(sit[:2], coh, n_boot=20)
    with pytest.raises(ValueError, match="empty"):
        run_poc([Sitting("S1", np.zeros((0, D)))] + sit[1:], coh, n_boot=20)
    with pytest.raises(ValueError, match="speakers"):
        run_poc(sit, make_cohort(per_source=(2,), clips=3), n_boot=20)
    with pytest.raises(ValueError, match="unknown variant"):
        run_poc(sit, coh, variants=["nope"], n_boot=20)


def test_separable_classes_give_a_near_zero_eer_and_identical_classes_about_half():
    easy = run_poc(make_sittings(noise=0.15), make_cohort(noise=0.15), n_boot=50).to_dict()
    assert easy["variants"]["baseline"]["test"]["eer"] < 0.02
    hard = run_poc(make_sittings(same_as_impostor=True), make_cohort(), n_boot=50).to_dict()
    assert 0.3 < hard["variants"]["baseline"]["test"]["eer"] < 0.7


def test_all_variants_run_and_selection_uses_dev_only():
    r = run_poc(make_sittings(), make_cohort(), n_boot=30).to_dict()
    assert set(r["variants"]) == set(VARIANTS)
    dev = {k: v["dev_eer"] for k, v in r["variants"].items()}
    assert r["selected"] == min(dev, key=lambda k: (dev[k], list(VARIANTS).index(k)))
    for v in r["variants"].values():
        assert np.isfinite(v["test"]["eer"]) and v["test"]["n_genuine"] == 36


def test_fast_eer_matches_the_reference_eer():
    from kavach.eval.metrics import compute_eer

    rng = np.random.default_rng(0)
    g, i = rng.normal(1.0, 1, 400), rng.normal(0.0, 1, 4000)
    assert abs(_eer_fast(g, i) - compute_eer(g, i)[0]) < 0.01


def test_limits_lead_the_report_and_the_json():
    from kavach.eval.poc import render

    r = run_poc(make_sittings(), make_cohort(), n_boot=30)
    text = render(r)
    assert text.startswith(LIMITS)
    assert r.to_dict()["limits"] == LIMITS
    assert "not established" in text.lower() or "informative" in text.lower()


# ---------------------------------------------------------------- the runner: embedding, cache, outputs

import shutil  # noqa: E402

from clone_helpers import tone  # noqa: E402
from kavach.embedding import SpeakerEmbedding  # noqa: E402
from kavach.eval import poc_run  # noqa: E402
from kavach.studio.store import StudioStore  # noqa: E402

KW = dict(kind="words", prompt_id="w1", environment="QUIET_ROOM", orig_bytes=b"x", orig_ext=".webm", device="DEMO_LAPTOP_MIC")


class FakeEmbedder:
    model_name = "fake/ecapa"

    def __init__(self):
        self.n = 0
        self.owner = np.random.default_rng(5).normal(size=D)

    def embed(self, audio):
        self.n += 1
        rng = np.random.default_rng(self.n)
        return SpeakerEmbedding(self.owner * 2 + rng.normal(size=D) * 0.8)


@pytest.fixture()
def world(tmp_path):
    studio = StudioStore(tmp_path / "studio" / "S04", "S04")
    for session in ("S1", "S2", "S3"):
        for k in range(8):
            studio.add_clip(session_id=session, audio=tone(seconds=3.0, seed=hash((session, k)) % 1000), **KW)
    coh = make_cohort()
    files = []
    for si in range(3):
        idx = [i for i, s in enumerate(coh.source) if s == f"src{si}"]
        f = tmp_path / f"src{si}__ecapa@5s.npz"
        np.savez(f, vec=coh.vec[idx], speaker=np.array([coh.speaker[i] for i in idx]),
                 session=np.array(["x"] * len(idx)), path=np.array(["p"] * len(idx)), dur=np.full(len(idx), 5.0))
        files.append(str(f))
    return tmp_path, files


def _args(tmp, files, extra=()):
    a = ["--studio", str(tmp / "studio" / "S04"), "--sessions", "S1,S2,S3", "--out", str(tmp / "out"),
         "--cache", str(tmp / "cache"), "--n-boot", "20", "--sweep", "0"]
    for f in files:
        a += ["--cohort", f]
    return a + list(extra)


def test_the_runner_writes_every_output_and_a_manifest(world):
    tmp, files = world
    assert poc_run.main(_args(tmp, files), embedder=FakeEmbedder()) == 0
    out = tmp / "out"
    for name in ("report.md", "results.json", "preregistration.json", "tables/headline.tex", "tables/variants.tex",
                 "tables/per_sitting.tex", "figures/det.pdf", "figures/det.png", "figures/scores.pdf", "figures/scores.png"):
        assert (out / name).stat().st_size > 0, name
    res = json.loads((out / "results.json").read_text(encoding="utf-8"))
    man = res["manifest"]
    for key in ("commit", "dirty", "seed", "embedder", "clips_per_sitting", "cohort_files", "command", "genuine_probe_seconds"):
        assert key in man
    assert man["clips_per_sitting"] == {"S1": 8, "S2": 8, "S3": 8}
    assert man["probes_per_sitting"] == {"S1": 8, "S2": 8, "S3": 8}  # 3 s clips are one probe each
    assert json.loads((out / "preregistration.json").read_text(encoding="utf-8"))["sha256"] == res["protocol"]["sha256"]
    assert (out / "report.md").read_text(encoding="utf-8").startswith(LIMITS)


def test_the_second_run_uses_the_cache_and_repeats_exactly(world):
    tmp, files = world
    emb = FakeEmbedder()
    assert poc_run.main(_args(tmp, files), embedder=emb) == 0
    first = (tmp / "out" / "results.json").read_text(encoding="utf-8")
    n = emb.n
    assert poc_run.main(_args(tmp, files), embedder=emb) == 0
    assert emb.n == n
    assert (tmp / "out" / "results.json").read_text(encoding="utf-8") == first


def test_the_runner_refuses_a_session_without_clips_and_leaves_no_outputs(world, capsys):
    tmp, files = world
    assert poc_run.main(_args(tmp, files)[:3] + ["S1,S2,S9"] + _args(tmp, files)[4:], embedder=FakeEmbedder()) == 2
    assert not (tmp / "out").exists()
    assert "S9" in capsys.readouterr().err


def test_the_enrollee_is_removed_from_the_cohort_files_not_scored_against_themselves(world):
    tmp, files = world
    z = np.load(files[0], allow_pickle=False)
    np.savez(files[0], vec=np.vstack([z["vec"], z["vec"][:1]]), speaker=np.append(z["speaker"], "S04"),
             session=np.append(z["session"], "x"), path=np.append(z["path"], "p"), dur=np.append(z["dur"], 5.0))
    assert poc_run.main(_args(tmp, files), embedder=FakeEmbedder()) == 0
    split = json.loads((tmp / "out" / "results.json").read_text(encoding="utf-8"))["protocol"]["split"]
    assert all("S04" not in v for v in split.values())


def test_owner_clips_are_cut_to_the_probe_length_of_the_cohort(tmp_path):
    studio = StudioStore(tmp_path / "studio" / "S04", "S04")
    studio.add_clip(session_id="S1", audio=tone(seconds=12.0, seed=1), **KW)
    emb = FakeEmbedder()
    whole, probes, _, durs, _ = poc_run.embed_sittings(studio, ["S1"], "words", emb, tmp_path / "c.json", probe_seconds=5.0)
    assert len(whole["S1"]) == 1 and len(probes["S1"]) == 2  # 12 s -> 5 s + 5 s, the 2 s tail is dropped
    assert all(abs(d - 5.0) < 0.5 for d in durs["S1"])
    _, whole_probe, _, _, _ = poc_run.embed_sittings(studio, ["S1"], "words", emb, tmp_path / "c2.json", probe_seconds=0)
    assert len(whole_probe["S1"]) == 1


def test_a_perfect_system_does_not_get_a_zero_width_interval():
    r = run_poc(make_sittings(noise=0.05), make_cohort(noise=0.05), n_boot=50).to_dict()
    t = r["variants"]["baseline"]["test"]
    assert t["eer"] == 0.0 and t["eer_interval"][1] > 0.02


def test_the_fast_eer_threshold_equals_the_reference_one():
    from kavach.eval.metrics import compute_eer
    from kavach.eval.poc import _threshold_eer

    rng = np.random.default_rng(3)
    for shift in (0.5, 1.0, 2.0):
        g, i = rng.normal(shift, 1, 300), rng.normal(0.0, 1, 2000)
        assert abs(_threshold_eer(g, i) - compute_eer(g, i)[1]) < 1e-9


def test_the_sitting_matrix_says_a_new_sitting_costs_score_and_never_scores_a_probe_against_its_own_clip():
    from kavach.eval.poc import sitting_matrix

    sit = make_sittings(shift=0.8)
    m = sitting_matrix(sit)
    cells = m["mean_score"]
    for s in ("S1", "S2", "S3"):
        assert cells[s][s] > max(cells[o][s] for o in ("S1", "S2", "S3") if o != s)  # same sitting is easiest
    # probes cut from clips: a probe is only ever scored against the OTHER clips of its own sitting
    s0 = sit[0]
    leaky = Sitting("S1", s0.vecs, np.repeat(s0.vecs, 2, axis=0), np.repeat(np.arange(len(s0.vecs)), 2))
    assert sitting_matrix([leaky, sit[1], sit[2]])["mean_score"]["S1"]["S1"] < 0.999


def test_budget_curve_and_a_missing_clip_map_are_handled():
    r = run_poc(make_sittings(), make_cohort(), variants=["baseline"], n_boot=20).to_dict()
    assert [b["clips"] for b in r["budget"]] == [1, 2, 5, 10, 20, "all"]
    assert all(np.isfinite(b["eer"]) for b in r["budget"])
    sit = make_sittings()
    bad = [Sitting("S1", sit[0].vecs, sit[0].vecs[:5])] + sit[1:]  # probes without saying which clip they came from
    with pytest.raises(ValueError, match="probe_clip"):
        run_poc(bad, make_cohort(), n_boot=20)


def test_the_seed_sweep_reports_the_spread_over_other_splits(world):
    tmp, files = world
    assert poc_run.main(_args(tmp, files, ["--sweep", "3"]), embedder=FakeEmbedder()) == 0
    sw = json.loads((tmp / "out" / "results.json").read_text(encoding="utf-8"))["seed_sweep"]
    assert [r["seed"] for r in sw["splits"]] == [1, 2, 3]
    lo, med, hi = sw["baseline_eer_min_median_max"]
    assert lo <= med <= hi
    assert "Sensitivity to the random split" in (tmp / "out" / "report.md").read_text(encoding="utf-8")
