"""Figure tests.

A figure cannot be asserted pixel by pixel without becoming a test of
matplotlib's renderer, so these check the properties that would actually make
a figure wrong in the paper:

- **it survives greyscale**, because reviewers print, and a series told apart
  by hue alone becomes one smear on paper;
- **identity is never carried by colour alone**, which the point above forces
  anyway and accessibility requires independently;
- **nothing is drawn that the data does not support** -- a cell at the backoff
  prior is hatched, a vetoed trial is counted rather than clamped onto the
  axis, a curve is not extended to an operating point the system cannot reach;
- **it renders at all**, for every function, since a figure module that
  crashes on the real objects is discovered the night before a deadline.
"""

from __future__ import annotations

import math
import random

import numpy as np
import pytest

matplotlib = pytest.importorskip("matplotlib")
matplotlib.use("Agg")

from kavach.csbg.graph import CSBG  # noqa: E402
from kavach.csbg.ontology import Language, SemanticClass  # noqa: E402
from kavach.csbg.tokens import Token, UtteranceTokens  # noqa: E402
from kavach.eval import figures as F  # noqa: E402
from kavach.eval.ablation import (  # noqa: E402
    AblationRow,
    build_trials,
    run_ablation,
    split_by_speaker,
)
from kavach.simulation import make_corpus  # noqa: E402


@pytest.fixture(scope="module")
def corpus():
    return make_corpus(
        n_speakers=12, seed=4, separation=0.65,
        enrolment_utterances=20, trial_utterances=5,
    )


@pytest.fixture(scope="module")
def graphs(corpus):
    return {sid: CSBG.build(sid, utts) for sid, utts in corpus.enrolment.items()}


@pytest.fixture(scope="module")
def report(graphs, corpus):
    def acoustic(probe, claimed, _u):
        rng = random.Random(f"a:{probe}:{claimed}")
        return max(0.0, min(1.0, rng.gauss(0.78 if probe == claimed else 0.55, 0.12)))

    return run_ablation(
        graphs, corpus.trials, speaker_score_fn=acoustic, bootstrap=30,
        enrolment=corpus.enrolment, stability_budgets=(2, 5, 20),
    )


@pytest.fixture(autouse=True)
def _close_figures():
    yield
    import matplotlib.pyplot as plt

    plt.close("all")


def _is_grey(hex_colour: str) -> bool:
    h = hex_colour.lstrip("#")
    r, g, b = (int(h[i : i + 2], 16) for i in (0, 2, 4))
    return r == g == b


# --------------------------------------------------------------------------
# House style
# --------------------------------------------------------------------------


class TestStyle:
    def test_the_series_palette_is_achromatic(self):
        """The user's standing constraint, enforced rather than remembered.

        Greyscale-safe is strictly stronger than colourblind-safe, and it is
        the one that matters for a printed proceedings page.
        """
        for colour in (*F.INK, F.GRID, F.FILL):
            assert _is_grey(colour), f"{colour} is not a neutral grey"

    def test_identity_encodings_are_as_deep_as_the_palette(self):
        """Line style and marker are the primary encodings, so there must be
        at least as many of them as there are ink shades to pair with."""
        assert len(F.LINE_STYLES) >= len(F.INK)
        assert len(F.MARKERS) >= len(F.INK)

    def test_style_sets_serif_and_drops_chartjunk(self):
        F.apply_style()
        assert matplotlib.rcParams["font.family"] == ["serif"]
        assert matplotlib.rcParams["axes.spines.top"] is False
        assert matplotlib.rcParams["axes.spines.right"] is False
        assert matplotlib.rcParams["legend.frameon"] is False

    def test_gridlines_are_solid(self):
        """A dashed grid reads as a threshold or a projection."""
        F.apply_style()
        assert matplotlib.rcParams["grid.linestyle"] == "-"

    def test_pdf_text_is_embedded_as_truetype(self):
        """Type-3 text is not selectable and some venues reject it."""
        F.apply_style()
        assert matplotlib.rcParams["pdf.fonttype"] == 42


# --------------------------------------------------------------------------
# DET
# --------------------------------------------------------------------------


class TestDetCurve:
    def test_one_line_per_configuration_plus_the_diagonal(self, report):
        fig = F.det_curve(report.configurations)
        ax = fig.axes[0]
        plotted = [line for line in ax.lines if line.get_label() != "_nolegend_"]
        named = [line for line in plotted if not line.get_label().startswith("_")]
        assert len(named) == len(report.configurations)

    def test_every_line_has_a_distinct_style(self, report):
        fig = F.det_curve(report.configurations)
        ax = fig.axes[0]
        named = [line for line in ax.lines if not line.get_label().startswith("_")]
        styles = [line.get_linestyle() for line in named]
        assert len(set(styles)) == len(styles), "two configurations share a line style"

    def test_axes_are_logarithmic(self, report):
        ax = F.det_curve(report.configurations).axes[0]
        assert ax.get_xscale() == "log"
        assert ax.get_yscale() == "log"

    def test_zero_error_points_do_not_vanish_or_crash(self):
        """A perfect operating point is 0, which a log axis cannot draw.

        Clipping to half the smallest resolvable rate keeps the curve's shape
        without claiming an error rate the trial count cannot support.
        """
        class FakeMetrics:
            eer = 0.0
            n_genuine = 40
            n_impostor = 40
            det_curve = tuple(
                type("P", (), {"far": f, "frr": 1.0 - f})()
                for f in (0.0, 0.25, 0.5, 1.0)
            )

        class FakeConfig:
            name = "perfect"
            metrics = FakeMetrics()

        fig = F.det_curve([FakeConfig()])
        assert fig.axes[0].lines

    def test_legend_appears_only_with_more_than_one_series(self, report):
        """One series needs no legend box -- the title names it."""
        single = F.det_curve(report.configurations[:1])
        assert single.axes[0].get_legend() is None
        multi = F.det_curve(report.configurations)
        assert multi.axes[0].get_legend() is not None


# --------------------------------------------------------------------------
# Ablation bars
# --------------------------------------------------------------------------


class TestAblationBars:
    def rows(self):
        return [
            AblationRow(name="a", eer=0.3, delta=0.02, scope="csbg"),
            AblationRow(name="b", eer=0.25, delta=-0.03, scope="csbg"),
            AblationRow(name="c", eer=0.28, delta=0.01, scope="full system"),
        ]

    def test_scope_filter_keeps_rows_comparable(self):
        """Rows in different scopes are measured on different systems.

        Stacking them in one chart invites exactly the comparison the report's
        prose warns against.
        """
        fig = F.ablation_bars(self.rows(), scope="csbg")
        assert len(fig.axes[0].patches) == 2

    def test_every_bar_is_the_same_ink(self):
        """One series, one colour. Shading by magnitude would encode bar
        length twice and spend the only free channel saying nothing new."""
        fig = F.ablation_bars(self.rows(), scope="csbg")
        colours = {p.get_facecolor() for p in fig.axes[0].patches}
        assert len(colours) == 1

    def test_bars_are_sorted_by_delta(self):
        fig = F.ablation_bars(self.rows(), scope="csbg")
        labels = [t.get_text() for t in fig.axes[0].get_yticklabels()]
        assert labels == ["b", "a"]

    def test_zero_line_is_drawn(self):
        """A signed quantity needs its origin visible or the sign is guesswork."""
        fig = F.ablation_bars(self.rows(), scope="csbg")
        assert any(
            line.get_xdata()[0] == 0.0 for line in fig.axes[0].lines
        ) or fig.axes[0].get_xlim()[0] < 0 < fig.axes[0].get_xlim()[1]

    def test_empty_selection_raises_rather_than_drawing_nothing(self):
        with pytest.raises(ValueError, match="no ablation rows"):
            F.ablation_bars(self.rows(), scope="nonexistent")


# --------------------------------------------------------------------------
# CSBG heatmap
# --------------------------------------------------------------------------


class TestCsbgHeatmap:
    def test_sequential_single_hue_survives_greyscale(self, graphs):
        """A diverging map collapses on paper: both poles print dark and the
        midpoint prints light, which is unreadable exactly where it matters."""
        ids = sorted(graphs)
        fig = F.csbg_heatmap([graphs[ids[0]], graphs[ids[1]]])
        image = fig.axes[0].images[0]
        ramp = image.get_cmap()
        samples = [ramp(v) for v in (0.0, 0.25, 0.5, 0.75, 1.0)]
        for r, g, b, _ in samples:
            assert r == pytest.approx(g, abs=0.02)
            assert g == pytest.approx(b, abs=0.02)
        lightness = [s[0] for s in samples]
        assert lightness == sorted(lightness, reverse=True), "ramp is not monotone"

    def test_low_evidence_cells_are_hatched(self):
        """A class at the backoff prior shows the population's habit, not the
        speaker's. Printing it like a measured cell asserts a measurement."""
        tokens = [
            Token(text="t", language=Language.TA, semantic_class=SemanticClass.FOOD)
            for _ in range(40)
        ]
        graph = CSBG.build("thin", [UtteranceTokens(utterance_id="u", tokens=tokens)])
        fig = F.csbg_heatmap([graph])
        hatched = [p for p in fig.axes[0].patches if p.get_hatch()]
        assert hatched, "no cell hatched despite only one class having evidence"

    def test_colourbar_marks_the_no_preference_landmark(self, graphs):
        """The one thing a sequential ramp loses versus a diverging one."""
        ids = sorted(graphs)
        fig = F.csbg_heatmap([graphs[ids[0]]])
        bar_axis = fig.axes[-1]
        labels = [t.get_text() for t in bar_axis.get_yticklabels()]
        assert "no preference" in labels
        assert "Tamil" in labels and "English" in labels

    def test_low_signal_classes_are_excluded_by_default(self, graphs):
        ids = sorted(graphs)
        fig = F.csbg_heatmap([graphs[ids[0]]])
        labels = [t.get_text() for t in fig.axes[0].get_yticklabels()]
        assert not any("Function Word" == label for label in labels)

    def test_no_graphs_raises(self):
        with pytest.raises(ValueError, match="no graphs"):
            F.csbg_heatmap([])


# --------------------------------------------------------------------------
# Stability
# --------------------------------------------------------------------------


class TestStability:
    def test_curve_and_interval_band_are_drawn(self, report):
        fig = F.stability(report.stability)
        ax = fig.axes[0]
        assert ax.lines
        assert ax.collections, "no confidence band"

    def test_seconds_axis_is_the_same_measure_not_a_second_series(self, report):
        """A secondary x in another unit is not a dual-axis chart, which would
        invent a correlation from an arbitrary scale alignment."""
        fig = F.stability(report.stability, seconds_axis=True)
        assert len(fig.axes[0].lines) == 1

    def test_no_points_raises(self):
        with pytest.raises(ValueError, match="no stability points"):
            F.stability([])


# --------------------------------------------------------------------------
# The A5 scatter
# --------------------------------------------------------------------------


class TestIapmrScatter:
    def data(self, n=12):
        rng = random.Random(2)
        consistency = {f"s{i}": 0.3 + 0.05 * i for i in range(n)}
        iapmr = {
            s: max(0.0, min(1.0, c + rng.gauss(0, 0.05)))
            for s, c in consistency.items()
        }
        return consistency, iapmr

    def test_one_marker_per_speaker(self):
        """The mean is what hides the finding: a system stopping every attack
        on 24 speakers and none on the 25th reads 96%."""
        consistency, iapmr = self.data()
        fig = F.iapmr_scatter(consistency, iapmr)
        points = fig.axes[0].lines[0]
        assert len(points.get_xdata()) == len(consistency)

    def test_only_speakers_present_in_both_inputs_are_plotted(self):
        consistency, iapmr = self.data()
        iapmr.pop("s0")
        fig = F.iapmr_scatter(consistency, iapmr)
        assert len(fig.axes[0].lines[0].get_xdata()) == len(consistency) - 1

    def test_too_few_speakers_raises(self):
        with pytest.raises(ValueError, match="at least 3 speakers"):
            F.iapmr_scatter({"a": 0.5}, {"a": 0.5})

    def test_correlation_is_reported_beside_the_trend_line(self):
        """A fitted line without its correlation reads as stronger than it is."""
        consistency, iapmr = self.data()
        fig = F.iapmr_scatter(consistency, iapmr)
        texts = [t.get_text() for t in fig.axes[0].texts]
        assert any("r =" in t for t in texts)

    def test_extreme_labels_stay_inside_the_axes(self):
        """The most-stolen speaker sits at the right edge by construction, so
        a rightward label overflows the axes every time."""
        consistency, iapmr = self.data()
        fig = F.iapmr_scatter(consistency, iapmr)
        fig.canvas.draw()
        ax = fig.axes[0]
        bounds = ax.get_window_extent()
        for annotation in ax.texts:
            if "r =" in annotation.get_text():
                continue
            box = annotation.get_window_extent()
            assert box.x0 >= bounds.x0 - 1 and box.x1 <= bounds.x1 + 1

    def test_a_flat_x_axis_does_not_crash_the_fit(self):
        consistency = {f"s{i}": 0.5 for i in range(6)}
        iapmr = {f"s{i}": 0.1 * i for i in range(6)}
        assert F.iapmr_scatter(consistency, iapmr) is not None

    def test_a_flat_axis_reports_no_correlation_rather_than_nan(self):
        """0/0 is not a weak correlation, it is no measurement.

        Printing "r = nan" beside a fitted line states a relationship that was
        never computed.
        """
        consistency = {f"s{i}": 0.3 + 0.05 * i for i in range(6)}
        iapmr = {f"s{i}": 0.5 for i in range(6)}
        fig = F.iapmr_scatter(consistency, iapmr)
        for text in fig.axes[0].texts:
            assert "nan" not in text.get_text()


# --------------------------------------------------------------------------
# Score distributions
# --------------------------------------------------------------------------


class TestScoreDistributions:
    def test_thresholds_are_drawn_where_they_were_set(self):
        """The figure that would have caught both miscalibrated floors on sight."""
        rng = np.random.default_rng(0)
        fig = F.score_distributions(
            rng.normal(0.6, 0.05, 60), rng.normal(0.45, 0.05, 300),
            threshold=0.55, veto_floor=0.35,
        )
        positions = {
            round(float(line.get_xdata()[0]), 6)
            for line in fig.axes[0].lines
            if len(set(line.get_ydata())) > 1 or line.get_linestyle() in ("--", ":")
        }
        assert 0.55 in positions and 0.35 in positions

    def test_vetoed_trials_are_counted_not_clamped(self):
        """-inf clamped onto the axis would draw a spike at the lowest bin and
        read as a cluster of very poor scores rather than as no score at all."""
        genuine = [0.6, 0.7, -math.inf, 0.65]
        impostor = [0.4, 0.45, 0.5, -math.inf]
        fig = F.score_distributions(genuine, impostor)
        labels = [t.get_text() for t in fig.axes[0].get_legend().get_texts()]
        assert any("2 vetoed" in label for label in labels)
        assert any("n=3" in label for label in labels)

    def test_all_infinite_on_one_side_raises(self):
        with pytest.raises(ValueError, match="finite scores on both sides"):
            F.score_distributions([-math.inf, -math.inf], [0.4, 0.5])


# --------------------------------------------------------------------------
# Consistency and saving
# --------------------------------------------------------------------------


class TestSpeakerConsistency:
    def test_a_deterministic_speaker_scores_near_one(self):
        tokens = [
            Token(text="t", language=Language.TA, semantic_class=cls_)
            for cls_ in (SemanticClass.FOOD, SemanticClass.KINSHIP)
            for _ in range(30)
        ]
        graph = CSBG.build("det", [UtteranceTokens(utterance_id="u", tokens=tokens)])
        assert F.speaker_consistency(graph) > 0.8

    def test_a_coin_flipping_speaker_scores_near_zero(self):
        rng = random.Random(0)
        tokens = [
            Token(
                text="t",
                language=rng.choice([Language.TA, Language.EN]),
                semantic_class=cls_,
            )
            for cls_ in (SemanticClass.FOOD, SemanticClass.KINSHIP)
            for _ in range(200)
        ]
        graph = CSBG.build("coin", [UtteranceTokens(utterance_id="u", tokens=tokens)])
        assert F.speaker_consistency(graph) < 0.25

    def test_an_empty_graph_is_zero_not_an_error(self):
        assert F.speaker_consistency(CSBG.build("empty", [])) == 0.0

    def test_classes_at_the_prior_are_skipped(self):
        """Counting them would measure the population and attribute it to the
        speaker, flattening every speaker onto the same value."""
        tokens = [
            Token(text="t", language=Language.TA, semantic_class=SemanticClass.FOOD)
            for _ in range(30)
        ]
        graph = CSBG.build("one", [UtteranceTokens(utterance_id="u", tokens=tokens)])
        # Only FOOD has evidence; the other 17 scoring classes sit at the prior.
        assert F.speaker_consistency(graph) > 0.5

    def test_is_bounded(self, graphs):
        for graph in graphs.values():
            assert 0.0 <= F.speaker_consistency(graph) <= 1.0


class TestSaving:
    def test_writes_vector_and_raster(self, report, tmp_path):
        fig = F.det_curve(report.configurations)
        path = F.save(fig, tmp_path / "sub" / "det")
        assert path.suffix == ".pdf"
        assert path.exists()
        assert path.with_suffix(".png").exists()

    def test_pdf_only_when_asked(self, report, tmp_path):
        fig = F.stability(report.stability)
        F.save(fig, tmp_path / "s", also_png=False)
        assert (tmp_path / "s.pdf").exists()
        assert not (tmp_path / "s.png").exists()


class TestEndToEnd:
    def test_every_figure_renders_from_real_objects(self, report, graphs, corpus):
        """A figure module that crashes on the real objects is discovered the
        night before a deadline, which is the point of this test."""
        ids = sorted(graphs)
        assert F.det_curve(report.configurations) is not None
        assert F.ablation_bars(report.ablations, scope="csbg") is not None
        assert F.csbg_heatmap([graphs[ids[0]], graphs[ids[1]]]) is not None
        assert F.stability(report.stability) is not None

        consistency = {s: F.speaker_consistency(g) for s, g in graphs.items()}
        iapmr = {s: 0.5 for s in graphs}
        assert F.iapmr_scatter(consistency, iapmr) is not None

        split = split_by_speaker(build_trials(graphs, corpus.trials), seed=0)
        genuine = [t.csbg_score for t in split.test if t.is_genuine]
        impostor = [t.csbg_score for t in split.test if not t.is_genuine]
        assert F.score_distributions(genuine, impostor) is not None
