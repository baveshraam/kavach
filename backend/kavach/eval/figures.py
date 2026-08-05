"""Publication figures. Every plot the paper prints is produced here.

HOUSE STYLE, AND WHY
--------------------
Serif labels, hairline axes, no top or right spine, no fill gradients, no
colour that carries meaning on its own. Two constraints drive all of it:

**It must survive greyscale.** Reviewers print. A figure whose series are told
apart by hue becomes one grey smear on paper, and the reader cannot recover
which line was which. So identity is carried by **line style and marker
shape**, with ink shade as reinforcement only -- which also satisfies the
accessibility rule that colour is never the sole encoding, for free.

**It must not look generated.** No neon, no gradient fills, no drop shadows,
no colour ramp across nominal categories. A bar chart of five ablations is one
series, so it is one ink; darkening the taller bars would double-encode length
as shade and spend the only free channel on information the bar already shows.

WHAT IS DELIBERATELY NOT HERE
-----------------------------
A dual-axis plot, anywhere. Two measures on two y-scales invent a correlation
by the arbitrary alignment of their scales. `stability` puts seconds on a
secondary *x* axis, which is the same measure in another unit and is not the
same thing.

Every function takes the objects `eval.ablation` and `attacks.suite` already
produce and returns a `Figure` without writing anything. `save` is separate so
the experiment runner decides the format and the path, and so a notebook can
show a figure without littering the filesystem.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from ..csbg.graph import CSBG, LANG_INDEX
from ..csbg.ontology import CLASS_ORDER, Language, scoring_classes

# --------------------------------------------------------------------------
# Style
# --------------------------------------------------------------------------

#: Ink shades, darkest first. Reinforcement for line style, never the primary
#: encoding -- see the module docstring.
INK = ("#1a1a1a", "#4a4a4a", "#6e6e6e", "#333333", "#5c5c5c")

#: Line styles in fixed assignment order. Fixed, because a configuration must
#: keep its style when another is added or dropped: a reader who learned
#: "solid is full fusion" is misled by a figure that reassigns it.
LINE_STYLES = ("-", "--", "-.", ":", (0, (5, 1, 1, 1, 1, 1)))

#: Marker shapes, same fixed order.
MARKERS = ("o", "s", "^", "D", "v")

#: Hairline grey for grid and spines. Solid, never dashed -- a dashed grid
#: reads as a threshold or a projection when it is only a grid.
GRID = "#d4d4d4"

#: Fill for single-series bars and heatmap-free areas.
FILL = "#3d3d3d"

_SERIF = ["DejaVu Serif", "Times New Roman", "Nimbus Roman", "serif"]


@dataclass(frozen=True, slots=True)
class FigureStyle:
    """Sizes in inches, for a two-column proceedings page.

    `single` fits one column of a typical ACL/ISCA template; `wide` spans
    both. Sizing here rather than at each call site keeps every figure in the
    paper at one type size, which is the thing that makes a figure set look
    deliberate.
    """

    single: tuple[float, float] = (3.4, 2.6)
    wide: tuple[float, float] = (7.0, 3.0)
    tall: tuple[float, float] = (3.4, 4.2)
    dpi: int = 300
    base_font: float = 8.0


STYLE = FigureStyle()


def apply_style() -> None:
    """Set the global rcParams. Idempotent; call before building figures."""
    import matplotlib

    matplotlib.rcParams.update(
        {
            "font.family": "serif",
            "font.serif": _SERIF,
            "mathtext.fontset": "dejavuserif",
            "font.size": STYLE.base_font,
            "axes.titlesize": STYLE.base_font + 1,
            "axes.labelsize": STYLE.base_font,
            "xtick.labelsize": STYLE.base_font - 1,
            "ytick.labelsize": STYLE.base_font - 1,
            "legend.fontsize": STYLE.base_font - 1,
            "axes.linewidth": 0.6,
            "axes.edgecolor": "#8a8a8a",
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.grid": True,
            "axes.axisbelow": True,
            "grid.color": GRID,
            "grid.linewidth": 0.5,
            "grid.linestyle": "-",
            "lines.linewidth": 1.2,
            "lines.markersize": 3.5,
            "legend.frameon": False,
            "figure.dpi": STYLE.dpi,
            "savefig.dpi": STYLE.dpi,
            "savefig.bbox": "tight",
            "savefig.pad_inches": 0.02,
            "pdf.fonttype": 42,  # embed as TrueType; editable text in the PDF
            "ps.fonttype": 42,
        }
    )


def _new(size: tuple[float, float]) -> tuple[Any, Any]:
    import matplotlib.pyplot as plt

    apply_style()
    fig, ax = plt.subplots(figsize=size)
    return fig, ax


def save(fig: Any, path: str | Path, *, also_png: bool = True) -> Path:
    """Write a figure as vector PDF, optionally with a PNG beside it.

    PDF because a paper wants vector text; PNG because a slide and a README do
    not read PDFs. `bbox_inches='tight'` is set globally so a long tick label
    cannot be clipped by the figure box -- the failure that shows up only
    after the figure is in the document.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path.with_suffix(".pdf"))
    if also_png:
        fig.savefig(path.with_suffix(".png"))
    return path.with_suffix(".pdf")


def close(fig: Any) -> None:
    import matplotlib.pyplot as plt

    plt.close(fig)


# --------------------------------------------------------------------------
# 1. DET curve
# --------------------------------------------------------------------------


def det_curve(
    configurations: Sequence[Any],
    *,
    title: str = "",
    mark_eer: bool = True,
) -> Any:
    """Detection error trade-off, one line per configuration.

    Log axes, because the interesting region is the low-error corner and a
    linear DET spends most of its area on operating points nobody deploys.

    **A vetoed system's curve is truncated, and that is the honest shape.** A
    veto rejects some genuine trials at every threshold, so FRR has a floor
    the curve cannot go below. Extending the line to the axis would draw an
    operating point the system cannot occupy -- the plotted version of the
    `FAR@1%FRR = 100.00` bug in PROJECT.md §5.5.

    Args:
        configurations: `ablation.ConfigurationResult` objects, or anything
            with `.name` and `.metrics.det_curve`.
        mark_eer: Draw the EER point on each curve. The one direct label worth
            having; a number on every point would be unreadable.
    """
    fig, ax = _new(STYLE.single)

    for i, config in enumerate(configurations):
        points = config.metrics.det_curve
        if not points:
            continue
        far = np.array([p.far for p in points], dtype=float)
        frr = np.array([p.frr for p in points], dtype=float)

        # Zeros cannot be drawn on a log axis. Clipping to half the smallest
        # resolvable rate keeps the line's shape without inventing a point at
        # an error rate the trial count cannot support.
        floor = 0.5 / max(config.metrics.n_genuine, config.metrics.n_impostor, 1)
        keep = (far > 0) & (frr > 0)
        far, frr = np.maximum(far[keep], floor), np.maximum(frr[keep], floor)
        if len(far) == 0:
            continue

        ax.plot(
            far * 100,
            frr * 100,
            linestyle=LINE_STYLES[i % len(LINE_STYLES)],
            color=INK[i % len(INK)],
            label=config.name,
            zorder=3,
        )
        if mark_eer and math.isfinite(config.metrics.eer):
            e = max(config.metrics.eer, floor) * 100
            ax.plot(
                e, e,
                marker=MARKERS[i % len(MARKERS)],
                color=INK[i % len(INK)],
                markeredgecolor="white",
                markeredgewidth=0.6,
                linestyle="none",
                zorder=4,
            )

    lo = 0.3
    ax.plot([lo, 100], [lo, 100], color=GRID, linewidth=0.6, zorder=1)
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlim(lo, 100)
    ax.set_ylim(lo, 100)
    ax.set_xlabel("False acceptance rate (%)")
    ax.set_ylabel("False rejection rate (%)")
    if title:
        ax.set_title(title, loc="left")
    if len(configurations) > 1:
        # `best`, not a fixed corner: which corner is empty depends on how good
        # the systems are. A hardcoded "upper right" lands on top of the curves
        # for a weak configuration set, which is where the legend was found
        # sitting the first time this was rendered.
        ax.legend(loc="best", handlelength=2.4, framealpha=0.9,
                  facecolor="white", edgecolor="none", frameon=True)
    fig.tight_layout()
    return fig


# --------------------------------------------------------------------------
# 2. Ablation bars
# --------------------------------------------------------------------------


def ablation_bars(
    rows: Sequence[Any],
    *,
    scope: str | None = None,
    title: str = "",
) -> Any:
    """Horizontal bars of Δ EER per ablation, zero-anchored.

    Horizontal because ablation names are phrases and rotating them under a
    vertical axis costs a reader more than the space it saves.

    One series, therefore **one ink for every bar**. Shading the larger deltas
    darker would encode bar length twice and spend the only free channel on
    what the length already says.

    Args:
        rows: `ablation.AblationRow` objects.
        scope: Keep only rows of this scope. Rows in different scopes are
            measured on different systems, and a chart that stacked them would
            invite exactly the comparison `AblationReport.to_markdown` warns
            against in prose.
    """
    rows = [r for r in rows if scope is None or r.scope == scope]
    if not rows:
        raise ValueError("no ablation rows to plot")

    order = sorted(rows, key=lambda r: r.delta)
    labels = [r.name for r in order]
    deltas = [r.delta * 100 for r in order]
    y = np.arange(len(order))

    height = max(1.4, 0.32 * len(order) + 0.9)
    fig, ax = _new((STYLE.single[0], height))

    ax.barh(y, deltas, height=0.62, color=FILL, zorder=3)
    ax.axvline(0.0, color="#8a8a8a", linewidth=0.8, zorder=4)
    ax.set_yticks(y, labels)
    ax.set_xlabel("Δ EER vs. un-ablated (percentage points)")
    ax.grid(axis="y", visible=False)

    span = max(abs(min(deltas)), abs(max(deltas)), 0.5)
    ax.set_xlim(-span * 1.35, span * 1.35)

    # Direct-label every bar: there are few, the axis is coarse, and the sign
    # is the entire point of the figure. Labels sit outside the bar end so a
    # short bar cannot clip its own number.
    for yi, d in zip(y, deltas):
        offset = span * 0.05
        ax.text(
            d + (offset if d >= 0 else -offset),
            yi,
            f"{d:+.2f}",
            va="center",
            ha="left" if d >= 0 else "right",
            color="#4a4a4a",
            fontsize=STYLE.base_font - 1.5,
        )

    if title:
        ax.set_title(title, loc="left")
    fig.tight_layout()
    return fig


# --------------------------------------------------------------------------
# 3. CSBG heatmap
# --------------------------------------------------------------------------


def csbg_heatmap(
    graphs: Sequence[CSBG],
    *,
    labels: Sequence[str] | None = None,
    include_low_signal: bool = False,
    title: str = "",
) -> Any:
    """P(Tamil | class) per speaker: the picture of what a CSBG *is*.

    Two contrasting speakers side by side is the figure that makes the
    hypothesis legible in one glance -- this is the thing an embedding cannot
    show, and it is worth a figure for that reason alone.

    **Sequential, one hue, not diverging.** P(Tamil) has a natural midpoint at
    0.5, which argues for a diverging map; but a diverging map collapses in
    greyscale, where both poles print dark and the midpoint prints light, so a
    printed figure becomes unreadable in exactly the place it matters. A
    single light-to-dark ramp is monotone in greyscale. The 0.5 landmark is
    restored as an explicit tick on the colour bar.

    **Cells with too little evidence are hatched, not coloured flat.** A class
    a speaker produced almost no tokens for sits at the backoff prior, so its
    colour is the population's habit rather than this speaker's. Printing that
    identically to a measured cell would show a confident value where there is
    no measurement -- the same error as scoring an unavailable branch zero.
    """
    import matplotlib.pyplot as plt
    from matplotlib.colors import LinearSegmentedColormap

    if not graphs:
        raise ValueError("no graphs to plot")

    classes = list(
        CLASS_ORDER if include_low_signal else scoring_classes(include_low_signal=False)
    )
    labels = list(labels or [g.speaker_id for g in graphs])
    ta = LANG_INDEX[Language.TA]

    matrix = np.zeros((len(classes), len(graphs)))
    counts = np.zeros_like(matrix)
    for j, graph in enumerate(graphs):
        for i, cls_ in enumerate(classes):
            idx = CLASS_ORDER.index(cls_)
            matrix[i, j] = graph.lexical_probs[idx, ta]
            counts[i, j] = graph.lexical_counts[idx].sum()

    apply_style()
    width = max(STYLE.single[0], 1.05 * len(graphs) + 1.6)
    fig, ax = plt.subplots(figsize=(width, 0.19 * len(classes) + 1.1))

    ramp = LinearSegmentedColormap.from_list("kavach_ta", ["#f2f2f2", "#1a1a1a"])
    image = ax.imshow(matrix, cmap=ramp, vmin=0.0, vmax=1.0, aspect="auto")

    floor = graphs[0].smoothing.class_alpha
    for i in range(len(classes)):
        for j in range(len(graphs)):
            if counts[i, j] < floor:
                ax.add_patch(
                    plt.Rectangle(
                        (j - 0.5, i - 0.5), 1, 1,
                        fill=False, hatch="////", linewidth=0.0,
                        edgecolor="#9a9a9a", zorder=3,
                    )
                )

    ax.set_xticks(np.arange(len(graphs)), labels)
    ax.set_yticks(
        np.arange(len(classes)),
        [c.value.replace("_", " ").title() for c in classes],
    )
    ax.set_xticks(np.arange(len(graphs) + 1) - 0.5, minor=True)
    ax.set_yticks(np.arange(len(classes) + 1) - 0.5, minor=True)
    ax.grid(which="minor", color="white", linewidth=1.0)
    ax.grid(which="major", visible=False)
    ax.tick_params(which="minor", length=0)

    bar = fig.colorbar(image, ax=ax, fraction=0.045, pad=0.03)
    bar.set_ticks([0.0, 0.5, 1.0])
    bar.set_ticklabels(["English", "no preference", "Tamil"])
    bar.outline.set_linewidth(0.4)
    bar.ax.tick_params(length=2)

    if title:
        ax.set_title(title, loc="left")
    fig.tight_layout()
    return fig


# --------------------------------------------------------------------------
# 4. Enrolment stability
# --------------------------------------------------------------------------


def stability(
    points: Sequence[Any],
    *,
    title: str = "",
    seconds_axis: bool = True,
) -> Any:
    """EER against enrolment budget, with the bootstrap interval as a band.

    **Read this figure twice.** Left to right it is how much speech a defender
    needs before a CSBG is usable. Read as an eavesdropping budget it is how
    much overheard speech an attacker needs to steal one (§5.1.2) -- one
    measurement, because the estimate an attacker forms and the graph a
    defender enrols converge at the same rate.

    The seconds axis on top is the same measure in another unit, not a second
    series on a second scale. It is labelled approximate because
    `stability_curve` converts with a nominal seconds-per-utterance until real
    durations exist.
    """
    if not points:
        raise ValueError("no stability points to plot")

    fig, ax = _new(STYLE.single)
    x = np.array([p.n_utterances for p in points], dtype=float)
    eer = np.array([p.eer for p in points]) * 100
    lo = np.array([p.ci_low for p in points]) * 100
    hi = np.array([p.ci_high for p in points]) * 100

    ax.fill_between(x, lo, hi, color="#cfcfcf", alpha=0.55, linewidth=0, zorder=2)
    ax.plot(x, eer, color=INK[0], marker=MARKERS[0], markeredgecolor="white",
            markeredgewidth=0.6, zorder=3)

    ax.set_xlabel("Enrolment utterances")
    ax.set_ylabel("CSBG EER (%)")
    ax.set_ylim(bottom=0)

    if seconds_axis:
        seconds = [p.approx_seconds for p in points]
        top = ax.secondary_xaxis(
            "top",
            functions=(
                lambda v: v * (seconds[0] / x[0] if x[0] else 1.0),
                lambda v: v / (seconds[0] / x[0] if x[0] else 1.0),
            ),
        )
        top.set_xlabel("Approximate speech duration (s)")
        top.tick_params(length=2)

    if title:
        ax.set_title(title, loc="left")
    fig.tight_layout()
    return fig


# --------------------------------------------------------------------------
# 5. The A5 tension
# --------------------------------------------------------------------------


def iapmr_scatter(
    consistency: dict[str, float],
    iapmr: dict[str, float],
    *,
    title: str = "",
    annotate_extremes: bool = True,
) -> Any:
    """Per-speaker stealability against per-speaker consistency. **§5.4.**

    The most interesting figure in the paper, and the one that says something
    uncomfortable:

        the more reliably a speaker code-switches, the better the biometric
        works for them and the more cheaply it is stolen.

    Both axes are driven by the same per-speaker quantity, so the prediction is
    a positive correlation. If it holds, the defence has a characterisable weak
    population -- the speakers it protects best -- and naming that population is
    worth more than a better mean.

    One dot per speaker, because the mean is what hides this: a system that
    stops every attack on 24 speakers and none on the 25th reads 96%.

    Args:
        consistency: speaker_id -> within-speaker consistency, in [0, 1].
        iapmr: speaker_id -> impostor attack presentation match rate, in [0, 1].
        annotate_extremes: Label the most and least protected speaker only.
            Labelling all 30 would be unreadable and is the "number on every
            point" failure.
    """
    shared = sorted(set(consistency) & set(iapmr))
    if len(shared) < 3:
        raise ValueError(
            f"need at least 3 speakers present in both inputs, got {len(shared)}"
        )

    x = np.array([consistency[s] for s in shared])
    y = np.array([iapmr[s] * 100 for s in shared])

    fig, ax = _new(STYLE.single)
    ax.plot(
        x, y, linestyle="none", marker=MARKERS[0], color=INK[0],
        markersize=4.0, markeredgecolor="white", markeredgewidth=0.6, zorder=3,
    )

    # A trend line only where a trend is estimable. Both axes must actually
    # vary: a constant y makes the correlation 0/0, and printing the resulting
    # "r = nan" beside a fitted line states a relationship that was not
    # measured. Constant input is not a weak correlation, it is no measurement.
    if len(shared) >= 5 and np.ptp(x) > 1e-9 and np.ptp(y) > 1e-9:
        slope, intercept = np.polyfit(x, y, 1)
        xs = np.array([x.min(), x.max()])
        ax.plot(xs, slope * xs + intercept, color="#8a8a8a", linewidth=0.9,
                linestyle="--", zorder=2)
        r = float(np.corrcoef(x, y)[0, 1])
        ax.text(
            0.03, 0.95, f"r = {r:+.2f}  (n = {len(shared)})",
            transform=ax.transAxes, va="top", ha="left",
            color="#4a4a4a", fontsize=STYLE.base_font - 1,
        )

    if annotate_extremes and len(shared) >= 3:
        midpoint = (x.min() + x.max()) / 2.0
        for idx in (int(np.argmax(y)), int(np.argmin(y))):
            # Label towards the inside of the plot. A point at the right edge
            # labelled rightwards pushes its text outside the axes, which is
            # the label-overflow failure -- and the most-stolen speaker is on
            # the right edge by construction, so it happens every time.
            right = x[idx] > midpoint
            ax.annotate(
                shared[idx],
                (x[idx], y[idx]),
                textcoords="offset points",
                xytext=(-6 if right else 6, 3),
                fontsize=STYLE.base_font - 2,
                color="#6e6e6e",
                ha="right" if right else "left",
            )

    ax.set_xlabel("Within-speaker consistency")
    ax.set_ylabel("A5 attack success rate (IAPMR, %)")
    ax.set_ylim(bottom=0)
    if title:
        ax.set_title(title, loc="left")
    fig.tight_layout()
    return fig


# --------------------------------------------------------------------------
# 6. Score distributions
# --------------------------------------------------------------------------


def score_distributions(
    genuine: Sequence[float],
    impostor: Sequence[float],
    *,
    threshold: float | None = None,
    veto_floor: float | None = None,
    title: str = "",
    xlabel: str = "Branch score",
) -> Any:
    """Genuine and impostor score histograms, with the thresholds drawn on.

    This is the figure that would have caught both miscalibrated floors in
    PROJECT.md §5.1 on sight. The CSBG veto sat at 0.15 while the impostor mode
    was at 0.29 -- one look at the two distributions with the floor drawn
    across them shows a line standing in empty space below everything it was
    meant to catch. Draw it for every branch before quoting any threshold.

    Non-finite scores (vetoed trials, carried as -inf) are counted and reported
    in the legend rather than dropped silently or clamped onto the axis.
    """
    g = np.asarray(genuine, dtype=float)
    i = np.asarray(impostor, dtype=float)
    n_vetoed = int(np.sum(~np.isfinite(g))) + int(np.sum(~np.isfinite(i)))
    g, i = g[np.isfinite(g)], i[np.isfinite(i)]
    if len(g) == 0 or len(i) == 0:
        raise ValueError("need finite scores on both sides")

    fig, ax = _new(STYLE.single)
    bins = np.linspace(
        min(g.min(), i.min()), max(g.max(), i.max()), 32
    )
    ax.hist(i, bins=bins, color="#c4c4c4", edgecolor="white", linewidth=0.4,
            label=f"impostor (n={len(i)})", zorder=2)
    ax.hist(g, bins=bins, histtype="step", color=INK[0], linewidth=1.3,
            label=f"genuine (n={len(g)})", zorder=3)

    # Headroom before the marker labels go in, so they sit above the bars
    # rather than across them, and so `loc="best"` has an empty band to find.
    top = ax.get_ylim()[1] * 1.28
    ax.set_ylim(top=top)

    for value, text, colour, style in (
        (threshold, "threshold", "#4a4a4a", "--"),
        (veto_floor, "veto floor", "#8a4b2f", ":"),
    ):
        if value is None:
            continue
        ax.axvline(value, color=colour, linewidth=0.9, linestyle=style, zorder=4)
        ax.text(
            value, top * 0.99, f" {text}", rotation=90, va="top", ha="left",
            fontsize=STYLE.base_font - 2, color=colour, zorder=5,
        )

    ax.set_xlabel(xlabel)
    ax.set_ylabel("Trials")
    handles, labels_ = ax.get_legend_handles_labels()
    if n_vetoed:
        labels_ = [*labels_, f"{n_vetoed} vetoed (off scale)"]
        handles = [*handles, ax.plot([], [], linestyle="none")[0]]
    ax.legend(handles, labels_, loc="best", framealpha=0.9,
              facecolor="white", edgecolor="none", frameon=True)
    if title:
        ax.set_title(title, loc="left")
    fig.tight_layout()
    return fig


def speaker_consistency(graph: CSBG) -> float:
    """How reliably one speaker follows their own language habits, in [0, 1].

    Mean distance of P(lang | class) from a coin flip, over the classes the
    speaker has own-evidence for, rescaled so 0 is "always 50/50" and 1 is
    "fully deterministic in every class".

    Classes below the smoothing pseudo-count are skipped rather than counted:
    they sit at the backoff prior, so including them would measure the
    *population's* consistency and attribute it to this speaker -- which would
    put every speaker at nearly the same value and flatten the §5.4 scatter
    into noise.
    """
    ta = LANG_INDEX[Language.TA]
    floor = graph.smoothing.class_alpha
    deviations = [
        abs(float(graph.lexical_probs[CLASS_ORDER.index(cls_), ta]) - 0.5)
        for cls_ in scoring_classes(include_low_signal=False)
        if graph.lexical_counts[CLASS_ORDER.index(cls_)].sum() >= floor
    ]
    if not deviations:
        return 0.0
    return float(np.mean(deviations) * 2.0)


__all__ = [
    "FILL",
    "GRID",
    "INK",
    "LINE_STYLES",
    "MARKERS",
    "STYLE",
    "FigureStyle",
    "ablation_bars",
    "apply_style",
    "close",
    "csbg_heatmap",
    "det_curve",
    "iapmr_scatter",
    "save",
    "score_distributions",
    "speaker_consistency",
    "stability",
]
