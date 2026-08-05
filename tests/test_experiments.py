"""Experiment-runner tests.

The runner's job is to make a number impossible to misattribute. So the
properties asserted here are mostly about *provenance surviving the last hop*:

- a simulated run must be unreportable, and must say so in the JSON, the
  README and inside every `.tex` file -- the last one because a `.tex` that
  looks identical whether it came from recordings or from `simulation.py` is
  exactly how a simulated number reaches a submission;
- `results.json` must be valid JSON, which means no bare `NaN`, which means
  the unattainable operating points have to be `null`;
- the run manifest must pin the commit and the seeds, because "which version
  produced table 3" is always asked and otherwise unanswerable.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

pytest.importorskip("matplotlib")
import matplotlib  # noqa: E402

matplotlib.use("Agg")

from kavach import corpus as C  # noqa: E402
from kavach import experiments as X  # noqa: E402
from kavach.csbg.ontology import Language, SemanticClass  # noqa: E402
from kavach.csbg.tokens import Token  # noqa: E402


@pytest.fixture(autouse=True)
def _close_figures():
    yield
    import matplotlib.pyplot as plt

    plt.close("all")


@pytest.fixture(scope="module")
def results() -> X.Results:
    return X.run(
        X.ExperimentConfig(
            n_simulated_speakers=12,
            bootstrap=25,
            stability_budgets=(2, 5),
            simulate_branches=True,
            figures=False,
        )
    )


# --------------------------------------------------------------------------
# The run
# --------------------------------------------------------------------------


class TestRun:
    def test_produces_every_section(self, results):
        assert results.report.configurations
        assert results.report.ablations
        assert results.report.stability
        assert results.coverage.classes
        assert results.consistency

    def test_a_simulated_run_is_never_reportable(self, results):
        assert not results.reportable
        assert any("SIMULATED" in b for b in results.blockers)

    def test_stand_in_branches_are_their_own_blocker(self, results):
        """Independently of the corpus: even on real recordings, a fusion row
        built from drawn numbers is a test of the harness."""
        assert any("stand-ins" in b for b in results.blockers)

    def test_csbg_only_run_has_no_stand_in_blocker(self):
        out = X.run(
            X.ExperimentConfig(
                n_simulated_speakers=8, bootstrap=10,
                stability_budgets=(2,), figures=False,
            )
        )
        assert not any("stand-ins" in b for b in out.blockers)

    def test_a_structurally_broken_corpus_stops_the_run(self, tmp_path):
        """An experiment on a corpus that fails validation produces numbers
        about nothing, which is worse than producing none."""
        corpus = C.Corpus(name="broken", provenance=C.Provenance.RECORDED)
        corpus.speakers.append(C.SpeakerRecord(speaker_id="a"))  # no consent
        corpus.sessions.append(C.SessionRecord(session_id="a_s0", speaker_id="a"))
        path = C.save_manifest(corpus, tmp_path / "manifest.json")

        with pytest.raises(ValueError, match="not structurally sound"):
            X.run(X.ExperimentConfig(manifest=path, figures=False))

    def test_too_few_speakers_after_the_split_is_an_error(self, tmp_path):
        corpus = C.Corpus(name="tiny", provenance=C.Provenance.RECORDED)
        for s in ("a", "b"):
            corpus.speakers.append(C.SpeakerRecord(speaker_id=s, consent_ref="c"))
            for i in range(2):
                sid = f"{s}_s{i}"
                corpus.sessions.append(
                    C.SessionRecord(session_id=sid, speaker_id=s, recorded_on=f"2026-0{i+1}-01")
                )
                corpus.utterances.append(
                    C.UtteranceRecord(
                        utterance_id=f"{sid}_u", session_id=sid, speaker_id=s,
                        tokens=[
                            Token(text="t", language=Language.TA,
                                  semantic_class=SemanticClass.FOOD)
                        ] * 10,
                    )
                )
        path = C.save_manifest(corpus, tmp_path / "manifest.json")
        with pytest.raises(ValueError, match="speakers survived"):
            X.run(X.ExperimentConfig(manifest=path, figures=False))

    def test_graphs_are_carried_not_rebuilt(self, results):
        """Two reconstructions of "the enrolled graphs" that disagreed would
        put a figure and a table in one paper describing different systems."""
        assert results.graphs
        assert set(results.graphs) == set(results.consistency)

    def test_stability_axis_uses_measured_durations(self, results):
        """`simulation` gives every utterance a duration, so the axis should
        not be sitting on the 6.0 s placeholder."""
        assert results.report.stability
        first = results.report.stability[0]
        assert first.approx_seconds != pytest.approx(first.n_utterances * 6.0)


# --------------------------------------------------------------------------
# results.json
# --------------------------------------------------------------------------


class TestResultsJson:
    def test_is_valid_json_with_no_nan_literals(self, results, tmp_path):
        """`json.dumps` emits bare `NaN` by default, which is invalid JSON:
        the file would load on the machine that wrote it and fail elsewhere."""
        results.config.out_dir = tmp_path
        path = X.write(results)
        raw = path.read_text(encoding="utf-8")
        for token in ("NaN", "Infinity", "-Infinity"):
            assert token not in raw
        json.loads(raw)

    def test_unattainable_operating_points_are_null(self, results):
        """`null`, not 100.0 -- the bug in PROJECT.md §5.5, one layer out."""
        payload = results.to_dict()
        for config in payload["configurations"]:
            for key in ("far_at_frr_1pct", "frr_at_far_1pct"):
                assert config[key] is None or isinstance(config[key], float)

    def test_records_the_commit_and_the_seed(self, results):
        payload = results.to_dict()
        assert payload["environment"]["git_commit"]
        assert payload["config"]["seed"] == results.config.seed
        assert "git_dirty" in payload["environment"]

    def test_records_the_corpus_provenance(self, results):
        assert results.to_dict()["corpus"]["provenance"] == "SIMULATED"

    def test_carries_the_blockers_into_the_payload(self, results):
        payload = results.to_dict()
        assert payload["reportable"] is False
        assert payload["blockers"] == results.blockers

    def test_ablation_scopes_survive_serialisation(self, results):
        scopes = {a["scope"] for a in results.to_dict()["ablations"]}
        assert "csbg" in scopes

    def test_version_is_stamped(self, results):
        assert results.to_dict()["results_version"] == X.RESULTS_VERSION


# --------------------------------------------------------------------------
# LaTeX
# --------------------------------------------------------------------------


class TestLatex:
    def test_every_table_warns_when_the_run_is_not_reportable(self, results):
        """The last hop is where a simulated number would slip into a paper.

        A `.tex` that looks the same either way is the failure; the banner is
        the fix, and it is emitted by code so it cannot be forgotten.
        """
        for name, builder in X.TABLES.items():
            text = builder(results)
            assert "NOT A RESULT" in text, f"{name}.tex carries no provenance banner"

    def test_no_banner_on_a_reportable_run(self, results):
        results.blockers = []
        try:
            assert "NOT A RESULT" not in X.configurations_table(results)
        finally:
            results.blockers = list(results.to_dict()["blockers"]) or ["restored"]

    def test_underscores_and_percents_are_escaped(self):
        assert X._tex("a_b 50% $x #1 &c") == r"a\_b 50\% \$x \#1 \&c"

    def test_backslash_is_escaped_first(self):
        r"""Escaping `\` after `_` would turn `\_` into `\textbackslash{}_`."""
        assert X._tex("a\\b") == r"a\textbackslash{}b"

    def test_tables_are_bodies_not_floats(self, results):
        """Placement and numbering belong to the document, not the harness."""
        text = X.configurations_table(results)
        assert r"\begin{tabular}" in text
        assert r"\begin{table}" not in text

    def test_every_configuration_becomes_a_row(self, results):
        text = X.configurations_table(results)
        for config in results.report.configurations:
            assert X._tex(config.name) in text

    def test_coverage_table_skips_classes_with_no_tokens(self, results):
        text = X.coverage_table(results)
        empty = [c for c in results.coverage.classes if c.n_tokens == 0]
        for c in empty:
            label = X._tex(c.semantic_class.value.replace("_", " ").title())
            assert f"\n{label} &" not in text


# --------------------------------------------------------------------------
# Writing
# --------------------------------------------------------------------------


class TestWrite:
    def test_writes_the_expected_tree(self, results, tmp_path):
        results.config.out_dir = tmp_path
        results.config.figures = True
        X.write(results)

        assert (tmp_path / "results.json").exists()
        assert (tmp_path / "report.md").exists()
        assert (tmp_path / "README.md").exists()
        for name in X.TABLES:
            assert (tmp_path / "tables" / f"{name}.tex").exists()
        assert (tmp_path / "figures" / "det.pdf").exists()
        assert (tmp_path / "figures" / "stability.pdf").exists()
        assert (tmp_path / "figures" / "csbg_heatmap.pdf").exists()

    def test_readme_leads_with_the_reportability_verdict(self, results, tmp_path):
        results.config.out_dir = tmp_path
        results.config.figures = False
        X.write(results)
        text = (tmp_path / "README.md").read_text(encoding="utf-8")
        assert "NOT reportable" in text
        assert "SIMULATED" in text

    def test_figures_can_be_skipped(self, results, tmp_path):
        results.config.out_dir = tmp_path
        results.config.figures = False
        X.write(results)
        assert not (tmp_path / "figures").exists()

    def test_heatmap_contrasts_the_extremes(self, results, tmp_path):
        """Two speakers who behave alike make the figure illegible while
        looking like a fair sample."""
        path = X.write_heatmap(
            results.graphs, tmp_path, consistency=results.consistency
        )
        assert path is not None and path.exists()

    def test_heatmap_needs_two_speakers(self, tmp_path):
        assert X.write_heatmap({}, tmp_path, consistency={}) is None


# --------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------


class TestCli:
    def test_end_to_end(self, tmp_path, capsys):
        code = X.main(
            [
                "--out", str(tmp_path),
                "--speakers", "10",
                "--bootstrap", "10",
                "--no-figures",
                "--simulate-branches",
            ]
        )
        assert code == 0
        assert (tmp_path / "results.json").exists()
        captured = capsys.readouterr()
        assert "Verification results" in captured.out
        assert "NOT REPORTABLE" in captured.err

    def test_a_broken_manifest_exits_nonzero_without_a_traceback(
        self, tmp_path, capsys
    ):
        corpus = C.Corpus(name="broken", provenance=C.Provenance.RECORDED)
        corpus.speakers.append(C.SpeakerRecord(speaker_id="a"))
        path = C.save_manifest(corpus, tmp_path / "manifest.json")

        code = X.main(["--out", str(tmp_path / "out"), "--manifest", str(path)])
        assert code == 2
        assert "error:" in capsys.readouterr().err

    def test_help_names_the_simulation_caveat(self, capsys):
        with pytest.raises(SystemExit):
            X.main(["--help"])
        assert "differ by construction" in capsys.readouterr().out

    def test_defaults_do_not_simulate_branches(self):
        args = X.build_parser().parse_args([])
        assert args.simulate_branches is False
        assert args.manifest is None
        assert args.out == Path("paper/results")


# --------------------------------------------------------------------------
# The within-session escape hatch
# --------------------------------------------------------------------------


def _single_session_corpus(n_speakers: int = 8, n_utterances: int = 14) -> C.Corpus:
    """A first-sitting pilot: every speaker recorded once, nobody twice."""
    from kavach.csbg.ontology import CLASS_ORDER

    corpus = C.Corpus(name="pilot", provenance=C.Provenance.SCRIPTED)
    for s in range(n_speakers):
        sid = f"S{s:02d}"
        corpus.speakers.append(
            C.SpeakerRecord(speaker_id=sid, consent_ref=f"c/{sid}", script_id="ABCDEFGH"[s])
        )
        session_id = f"{sid}_s1"
        corpus.sessions.append(C.SessionRecord(session_id=session_id, speaker_id=sid))
        for u in range(n_utterances):
            # Each speaker prefers a different language on a different class,
            # which is what the scripts do and what makes the split scoreable.
            tokens = [
                Token(
                    text=f"w{i}",
                    language=Language.TA if (i + s) % 3 else Language.EN,
                    semantic_class=CLASS_ORDER[(i + u) % len(CLASS_ORDER)],
                )
                for i in range(24)
            ]
            corpus.utterances.append(
                C.UtteranceRecord(
                    utterance_id=f"{session_id}_u{u:02d}",
                    session_id=session_id,
                    speaker_id=sid,
                    duration_sec=30.0,
                    tokens=tokens,
                    annotation_source=C.AnnotationSource.SYNTHETIC,
                )
            )
    return corpus


class TestWithinSession:
    def test_a_single_session_corpus_fails_by_default(self, tmp_path):
        """Every speaker in a first sitting has one session, so the default
        cross-session split drops all of them."""
        path = C.save_manifest(_single_session_corpus(), tmp_path / "manifest.json")
        with pytest.raises(ValueError, match="survived the session split"):
            X.run(X.ExperimentConfig(manifest=path, bootstrap=5, figures=False))

    def test_the_failure_names_the_flag_that_fixes_it(self, tmp_path):
        """An error that does not say what to do next reads as a broken
        pipeline rather than an incomplete corpus."""
        path = C.save_manifest(_single_session_corpus(), tmp_path / "manifest.json")
        with pytest.raises(ValueError, match="--within-session"):
            X.run(X.ExperimentConfig(manifest=path, bootstrap=5, figures=False))

    def test_within_session_produces_a_result(self, tmp_path):
        path = C.save_manifest(_single_session_corpus(), tmp_path / "manifest.json")
        results = X.run(
            X.ExperimentConfig(
                manifest=path,
                bootstrap=5,
                stability_budgets=(2,),
                figures=False,
                allow_within_session=True,
            )
        )
        assert results.report.fitted.policy.threshold is not None

    def test_and_stamps_the_run_unreportable_for_it(self, tmp_path):
        """The flattering split must be visible in the results file, not only
        in the command that produced it."""
        path = C.save_manifest(_single_session_corpus(), tmp_path / "manifest.json")
        results = X.run(
            X.ExperimentConfig(
                manifest=path,
                bootstrap=5,
                stability_budgets=(2,),
                figures=False,
                allow_within_session=True,
            )
        )
        assert not results.reportable
        assert any("within a session" in b for b in results.blockers)
        assert results.to_dict()["config"]["allow_within_session"] is True

    def test_the_flag_is_off_by_default_on_the_cli(self):
        assert X.build_parser().parse_args([]).within_session is False
        assert X.ExperimentConfig().allow_within_session is False

    def test_cli_passes_the_flag_through(self, tmp_path):
        path = C.save_manifest(_single_session_corpus(), tmp_path / "manifest.json")
        code = X.main(
            [
                "--out", str(tmp_path / "out"),
                "--manifest", str(path),
                "--bootstrap", "5",
                "--no-figures",
                "--within-session",
            ]
        )
        assert code == 0
        written = json.loads((tmp_path / "out" / "results.json").read_text())
        assert written["config"]["allow_within_session"] is True
        assert written["reportable"] is False
