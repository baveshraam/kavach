"""One command that produces every number and figure in the paper.

    python -m kavach.experiments --out paper/results/

Emits `results.json`, `figures/*.pdf` and `tables/*.tex`. **The paper reads
from `results.json`**, so no number is ever hand-typed into LaTeX -- which is
the only way a table stays consistent with the code after the fourth time the
evaluation is re-run.

WHY A RUN MANIFEST
------------------
`results.json` records the git commit, every seed, the package versions, the
corpus name, its provenance and its reportability reasons. Six months after
submission the question is always "which version of the code produced table
3", and the honest answer is otherwise "nobody knows". The manifest also makes
the run reproducible by a reviewer, which is the point of putting it in the
artefact.

PROVENANCE TRAVELS WITH THE NUMBERS
-----------------------------------
`corpus.Corpus.reportability()` returns the reasons a corpus may not be
reported, and this module carries them into the emitted LaTeX as a visible
caption note, not just a JSON field. A `.tex` file generated from simulated
speakers that looks exactly like one generated from recordings is how a
simulated number ends up in a submission -- the whole repository is arranged
against that, and the last hop into the document is where it would happen.

The tables are `tabular` bodies with a caption note, not floats. Wrapping in
`\\begin{table}` would decide placement and numbering, which belongs to the
document, not to the harness.
"""

from __future__ import annotations

import argparse
import json
import math
import platform
import random
import subprocess
import sys
from collections.abc import Sequence
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from . import corpus as corpus_mod
from .csbg.graph import CSBG
from .eval.ablation import AblationReport, ScoreFn, run_ablation
from .eval.metrics import format_rate
from .fusion import Branch
from .simulation import make_corpus

#: Bumped when the shape of `results.json` changes, so a paper build that
#: reads an older file fails loudly instead of silently finding no key.
RESULTS_VERSION = "1.0"


# --------------------------------------------------------------------------
# Configuration
# --------------------------------------------------------------------------


@dataclass(slots=True)
class ExperimentConfig:
    """Everything that changes a number, in one object that gets serialised.

    A parameter that affects a result and is not on here is a parameter the
    run manifest cannot record, which makes the run irreproducible in exactly
    the way the manifest exists to prevent.
    """

    out_dir: Path = Path("paper/results")
    manifest: Path | None = None
    """Corpus manifest. None falls back to `simulation.make_corpus`, and every
    output is then stamped unreportable."""

    seed: int = 0
    dev_fraction: float = 0.4
    bootstrap: int = 500
    max_veto_frr_cost: float = 0.02
    cohort_norm: bool = True
    stability_budgets: tuple[int, ...] = (2, 5, 10, 20, 30)

    allow_within_session: bool = False
    """Split enrolment from probes inside one sitting for speakers who only
    have one.

    Off by default, and a run with it on is stamped unreportable. It exists
    because the alternative during collection is worse: with a single session
    per speaker `split_sessions` drops everybody, `run` raises, and a pilot
    returns no number at all -- which reads as a broken pipeline rather than as
    an incomplete corpus, and gives nobody anything to check the plumbing with.

    What it buys is a smoke test. What it costs is the claim: enrolment and
    probe speech from one sitting share a microphone, a room and a mood, so the
    resulting EER measures how well the estimator memorises a recording session.
    It is the flattering direction, so a number from this path may not be
    compared against a cross-session number from any other."""

    simulate_branches: bool = False
    """Substitute documented stand-ins for the acoustic and knowledge branches.

    Off by default. These are **not models** -- they draw from a fixed
    distribution so the fusion and ablation machinery has something to weight
    against the CSBG. A run with this on is marked unreportable for that
    reason alone, independently of the corpus."""

    figures: bool = True
    n_simulated_speakers: int = 24
    """Only used when `manifest` is None."""

    def to_dict(self) -> dict[str, Any]:
        return {
            "out_dir": str(self.out_dir),
            "manifest": str(self.manifest) if self.manifest else None,
            "seed": self.seed,
            "dev_fraction": self.dev_fraction,
            "bootstrap": self.bootstrap,
            "max_veto_frr_cost": self.max_veto_frr_cost,
            "cohort_norm": self.cohort_norm,
            "allow_within_session": self.allow_within_session,
            "stability_budgets": list(self.stability_budgets),
            "simulate_branches": self.simulate_branches,
            "figures": self.figures,
            "n_simulated_speakers": self.n_simulated_speakers,
        }


# --------------------------------------------------------------------------
# Stand-in branches
# --------------------------------------------------------------------------


def _stand_in_acoustic(seed: int) -> ScoreFn:
    """A separable stand-in for ECAPA. **Not a model of anything.**

    Genuine trials draw N(0.78, 0.12), impostors N(0.55, 0.12). The overlap is
    deliberate: two perfectly separable branches would put every fusion
    configuration at 0% EER and the table would say nothing at all. Any number
    produced with this in play is a test of the harness.
    """

    def fn(probe: str, claimed: str, _utterances: Any) -> float:
        rng = random.Random(f"acoustic:{seed}:{probe}:{claimed}")
        base = 0.78 if probe == claimed else 0.55
        return max(0.0, min(1.0, rng.gauss(base, 0.12)))

    return fn


def _stand_in_knowledge(seed: int) -> ScoreFn:
    """A near-binary stand-in for the answer matcher, as the real one is."""

    def fn(probe: str, claimed: str, _utterances: Any) -> float:
        rng = random.Random(f"knowledge:{seed}:{probe}:{claimed}")
        if probe == claimed:
            return 1.0 if rng.random() < 0.95 else 0.3
        return 1.0 if rng.random() < 0.05 else 0.05

    return fn


# --------------------------------------------------------------------------
# Results
# --------------------------------------------------------------------------


@dataclass(slots=True)
class Results:
    """Everything one run produced, ready to serialise."""

    config: ExperimentConfig
    environment: dict[str, Any]
    corpus: dict[str, Any]
    report: AblationReport
    coverage: corpus_mod.CoverageReport
    consistency: dict[str, float] = field(default_factory=dict)
    blockers: list[str] = field(default_factory=list)
    """Why these numbers may not be reported. Empty means they may be."""

    graphs: dict[str, CSBG] = field(default_factory=dict)
    """The enrolled CSBGs, carried so the heatmap does not have to rebuild
    them. Rebuilding would be slower and, worse, could diverge: two
    reconstructions of "the enrolled graphs" that disagree would put a figure
    and a table in the same paper describing different systems. Not
    serialised -- a fitted graph belongs in the corpus, not the results file."""

    @property
    def reportable(self) -> bool:
        return not self.blockers

    def to_dict(self) -> dict[str, Any]:
        return {
            "results_version": RESULTS_VERSION,
            "reportable": self.reportable,
            "blockers": list(self.blockers),
            "config": self.config.to_dict(),
            "environment": self.environment,
            "corpus": self.corpus,
            "split": {
                "dev_speakers": self.report.split.dev_speakers,
                "test_speakers": self.report.split.test_speakers,
                "n_dev_trials": len(self.report.split.dev),
                "n_test_trials": len(self.report.split.test),
                "n_cohort_trials": len(self.report.split.cohort),
            },
            "fitted": {
                "weights": {
                    b.value: w for b, w in self.report.fitted.policy.weights.items()
                },
                "threshold": self.report.fitted.policy.threshold,
                "veto_floor": self.report.fitted.veto_floor,
                "veto_frr_cost": self.report.fitted.veto_frr_cost,
                "weights_source": self.report.fitted.weights_source,
                "threshold_source": self.report.fitted.threshold_source,
            },
            "measured_branches": [b.value for b in self.report.measured_branches],
            "configurations": [
                {
                    "name": c.name,
                    "branches": [b.value for b in c.branches],
                    "eer": c.metrics.eer,
                    "eer_ci": list(c.eer_ci),
                    "min_dcf": c.metrics.min_dcf,
                    "min_dcf_params": list(c.metrics.min_dcf_params),
                    "far_at_frr_1pct": _jsonable(c.metrics.far_at_frr_1pct),
                    "frr_at_far_1pct": _jsonable(c.metrics.frr_at_far_1pct),
                    "auc": c.metrics.auc,
                    "n_genuine": c.metrics.n_genuine,
                    "n_impostor": c.metrics.n_impostor,
                    "n_vetoed": c.n_vetoed,
                    "is_reliable": c.metrics.is_reliable,
                }
                for c in self.report.configurations
            ],
            "ablations": [
                {
                    "name": a.name,
                    "scope": a.scope,
                    "eer": a.eer,
                    "delta": a.delta,
                    "note": a.note,
                }
                for a in self.report.ablations
            ],
            "stability": [
                {
                    "n_utterances": p.n_utterances,
                    "approx_seconds": p.approx_seconds,
                    "eer": p.eer,
                    "ci_low": p.ci_low,
                    "ci_high": p.ci_high,
                }
                for p in self.report.stability
            ],
            "fairness": [
                {
                    "condition": f.condition,
                    "group": f.group,
                    "eer": f.eer,
                    "n_genuine": f.n_genuine,
                    "n_impostor": f.n_impostor,
                }
                for f in self.report.fairness
            ],
            "coverage": {
                "total_tokens": self.coverage.total_tokens,
                "total_choice_tokens": self.coverage.total_choice_tokens,
                "n_speakers": self.coverage.n_speakers,
                "min_tokens_for_own_evidence": self.coverage.min_tokens_for_own_evidence,
                "classes": [
                    {
                        "class": c.semantic_class.value,
                        "n_tokens": c.n_tokens,
                        "n_speakers_with_own_evidence": c.n_speakers_with_own_evidence,
                    }
                    for c in self.coverage.classes
                ],
                "prompts": [
                    {
                        "prompt_id": p.prompt_id,
                        "n_utterances": p.n_utterances,
                        "mean_tokens": p.mean_tokens,
                        "hit_rate": p.hit_rate,
                    }
                    for p in self.coverage.prompts
                ],
                "starved_classes": [
                    c.semantic_class.value for c in self.coverage.starved_classes()
                ],
                "wrapperless_prompts": [
                    p.prompt_id for p in self.coverage.wrapperless_prompts()
                ],
            },
            "speaker_consistency": dict(self.consistency),
            "caveats": list(self.report.caveats),
        }


def _jsonable(value: float) -> float | None:
    """NaN and infinity are not JSON. `None` reads as 'unattainable'.

    `json.dumps` emits bare `NaN`, which is invalid JSON and which several
    parsers accept silently and others reject -- so a results file would load
    on the machine that wrote it and fail on a reviewer's. `None` round-trips
    everywhere and keeps the meaning `format_rate` already gives it: not a
    rate, an operating point the system cannot occupy.
    """
    return None if not math.isfinite(value) else value


# --------------------------------------------------------------------------
# The run
# --------------------------------------------------------------------------


def _environment() -> dict[str, Any]:
    """Git commit, interpreter and package versions."""

    def git(*args: str) -> str:
        try:
            return subprocess.run(
                ["git", *args],
                capture_output=True, text=True, timeout=10,
                cwd=Path(__file__).resolve().parent,
            ).stdout.strip()
        except (OSError, subprocess.SubprocessError):
            return ""

    versions: dict[str, str] = {}
    for name in ("numpy", "scipy", "sklearn", "matplotlib"):
        try:
            versions[name] = __import__(name).__version__
        except Exception:  # noqa: BLE001 - a missing package is data, not an error
            versions[name] = "not installed"

    commit = git("rev-parse", "HEAD")
    return {
        "generated_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "git_commit": commit or "unknown",
        "git_dirty": bool(git("status", "--porcelain")),
        "python": platform.python_version(),
        "platform": platform.platform(),
        "packages": versions,
    }


def load_corpus(config: ExperimentConfig) -> corpus_mod.Corpus:
    """The corpus named by the config, or a simulated stand-in.

    Both paths return a `Corpus`, so everything downstream is identical and
    the only difference that survives is `provenance` -- which is precisely
    the difference that must survive.
    """
    if config.manifest is not None:
        return corpus_mod.load_manifest(config.manifest)
    sim = make_corpus(
        n_speakers=config.n_simulated_speakers,
        seed=config.seed,
        separation=0.65,
        consistency=0.85,
        enrolment_utterances=25,
        trial_utterances=6,
    )
    return corpus_mod.from_simulation(sim, sessions_per_speaker=2)


def run(config: ExperimentConfig) -> Results:
    """Score, split, fit, evaluate, ablate -- and record why it is or is not
    reportable."""
    corpus = load_corpus(config)

    problems = corpus.validate()
    if problems:
        raise ValueError(
            "corpus manifest is not structurally sound; fix these before "
            "running an experiment:\n  - " + "\n  - ".join(problems)
        )

    split = corpus_mod.split_sessions(
        corpus, allow_within_session=config.allow_within_session
    )
    if len(split.enrolment) < 4:
        hint = (
            ""
            if config.allow_within_session
            else " Every speaker in a first-sitting pilot has one session; "
            "--within-session scores it anyway and stamps the run unreportable."
        )
        raise ValueError(
            f"only {len(split.enrolment)} speakers survived the session split "
            f"(dropped {len(split.dropped_speakers)} for having one session). "
            f"The evaluation needs at least 4.{hint}"
        )

    graphs = {
        sid: CSBG.build(sid, utterances, total_duration_sec=0.0)
        for sid, utterances in split.enrolment.items()
    }

    speaker_fn = _stand_in_acoustic(config.seed) if config.simulate_branches else None
    knowledge_fn = _stand_in_knowledge(config.seed) if config.simulate_branches else None

    report = run_ablation(
        graphs,
        split.probes,
        speaker_score_fn=speaker_fn,
        knowledge_score_fn=knowledge_fn,
        groups=corpus.groups(),
        dev_fraction=config.dev_fraction,
        seed=config.seed,
        cohort_norm=config.cohort_norm,
        max_veto_frr_cost=config.max_veto_frr_cost,
        bootstrap=config.bootstrap,
        enrolment=split.enrolment,
        stability_budgets=config.stability_budgets,
        seconds_per_utterance=_mean_duration(corpus),
    )

    blockers = list(corpus.reportability())
    if config.simulate_branches:
        blockers.append(
            "the acoustic and knowledge branches are documented stand-ins drawing "
            "from a fixed distribution, not models: any fusion row is a test of "
            "the harness, not a measurement"
        )
    if not split.cross_session:
        blockers.append(
            "enrolment and probe speech were split within a session, which "
            "measures how well the estimator memorises one recording sitting"
        )

    consistency: dict[str, float] = {}
    try:
        from .eval import figures as figures_mod

        consistency = {sid: figures_mod.speaker_consistency(g) for sid, g in graphs.items()}
    except ImportError:
        pass

    return Results(
        config=config,
        environment=_environment(),
        corpus={
            "name": corpus.name,
            "provenance": corpus.provenance.value,
            "protocol_version": corpus.protocol_version,
            "ontology_version": corpus.ontology_version,
            "n_speakers": len(corpus.speakers),
            "n_sessions": len(corpus.sessions),
            "n_utterances": len(corpus.utterances),
            "n_annotated": len(corpus.annotated()),
            "session_split": split.summary(),
            "dropped_speakers": split.dropped_speakers,
            "cross_session": split.cross_session,
            "notes": corpus.notes,
        },
        report=report,
        coverage=corpus.coverage(),
        consistency=consistency,
        blockers=blockers,
        graphs=graphs,
    )


def _mean_duration(corpus: corpus_mod.Corpus) -> float:
    """Mean utterance duration, for the stability axis.

    Falls back to 6.0 s only when nothing is recorded, and that fallback is
    why the axis is labelled *approximate*.
    """
    durations = [u.duration_sec for u in corpus.utterances if u.duration_sec > 0]
    return sum(durations) / len(durations) if durations else 6.0


# --------------------------------------------------------------------------
# LaTeX
# --------------------------------------------------------------------------


#: Character -> LaTeX escape. Applied in **one pass**, deliberately.
#:
#: Sequential `str.replace` calls cannot do this correctly, because several
#: replacements contain characters that later rules also escape: `\` becomes
#: `\textbackslash{}`, and a subsequent `{` rule then escapes the braces of
#: that replacement, yielding `\textbackslash\{\}` -- which typesets as a
#: literal "\{}" rather than a backslash. Reordering the rules only moves the
#: collision, since `~` and `^` expand to braces too.
_TEX_ESCAPES = {
    "\\": r"\textbackslash{}",
    "&": r"\&",
    "%": r"\%",
    "$": r"\$",
    "#": r"\#",
    "_": r"\_",
    "{": r"\{",
    "}": r"\}",
    "~": r"\textasciitilde{}",
    "^": r"\textasciicircum{}",
}


def _tex(text: str) -> str:
    """Escape the characters that make LaTeX fail or silently misrender."""
    return "".join(_TEX_ESCAPES.get(char, char) for char in text)


def _provenance_note(results: Results) -> list[str]:
    """The banner that stops a simulated table looking like a measured one."""
    if results.reportable:
        return []
    lines = [
        "% ---------------------------------------------------------------",
        "% NOT A RESULT. This table was generated from data that cannot be",
        "% reported. Reasons:",
    ]
    lines += [f"%   - {b}" for b in results.blockers]
    lines += [
        "% Delete this banner only when kavach.experiments emits the table",
        "% without it -- do not edit it out by hand.",
        "% ---------------------------------------------------------------",
    ]
    return lines


def configurations_table(results: Results) -> str:
    """The money table: EER, minDCF and operating points per configuration."""
    lines = _provenance_note(results)
    lines += [
        r"\begin{tabular}{lrrrrr}",
        r"\toprule",
        r"Configuration & EER \% [95\% CI] & minDCF & FAR@1\%FRR & FRR@1\%FAR & $n$ gen/imp \\",
        r"\midrule",
    ]
    for c in results.report.configurations:
        lo, hi = c.eer_ci
        veto = f" ({c.n_vetoed} vetoed)" if c.n_vetoed else ""
        lines.append(
            f"{_tex(c.name)} & {c.metrics.eer * 100:.2f} "
            f"[{lo * 100:.2f}--{hi * 100:.2f}] & {c.metrics.min_dcf:.4f} & "
            f"{_tex(format_rate(c.metrics.far_at_frr_1pct))} & "
            f"{_tex(format_rate(c.metrics.frr_at_far_1pct))} & "
            f"{c.metrics.n_genuine}/{c.metrics.n_impostor}{_tex(veto)} \\\\"
        )
    lines += [r"\bottomrule", r"\end{tabular}"]

    if results.report.configurations:
        p, c_miss, c_fa = results.report.configurations[0].metrics.min_dcf_params
        lines += [
            "",
            f"% minDCF parameters: p_target={p}, c_miss={c_miss}, c_fa={c_fa}",
            "% 'n/a' is an operating point the system cannot occupy, not a rate "
            "of 100%. A veto puts a hard floor under FRR.",
        ]
    return "\n".join(lines) + "\n"


def ablations_table(results: Results) -> str:
    """Ablations, with scope, because rows in different scopes are different
    systems and a shared column would invite comparing them."""
    lines = _provenance_note(results)
    lines += [
        r"\begin{tabular}{llrr}",
        r"\toprule",
        r"Removed & Scope & EER \% & $\Delta$ EER \\",
        r"\midrule",
    ]
    for a in results.report.ablations:
        lines.append(
            f"{_tex(a.name)} & {_tex(a.scope)} & {a.eer * 100:.2f} & "
            f"{a.delta * 100:+.2f} \\\\"
        )
    lines += [
        r"\bottomrule",
        r"\end{tabular}",
        "",
        "% Each delta is against the un-ablated baseline OF ITS OWN SCOPE.",
        "% Rows in different scopes are not comparable with one another.",
    ]
    return "\n".join(lines) + "\n"


def stability_table(results: Results) -> str:
    lines = _provenance_note(results)
    lines += [
        r"\begin{tabular}{rrr}",
        r"\toprule",
        r"Enrolment utterances & Approx.\ seconds & EER \% [95\% CI] \\",
        r"\midrule",
    ]
    for p in results.report.stability:
        lines.append(
            f"{p.n_utterances} & {p.approx_seconds:.0f} & {p.eer * 100:.2f} "
            f"[{p.ci_low * 100:.2f}--{p.ci_high * 100:.2f}] \\\\"
        )
    lines += [
        r"\bottomrule",
        r"\end{tabular}",
        "",
        "% Read left to right: speech a defender needs. Read as an eavesdropping",
        "% budget: speech an attacker needs to steal the graph. Same measurement.",
    ]
    return "\n".join(lines) + "\n"


def fairness_table(results: Results) -> str:
    lines = _provenance_note(results)
    lines += [
        r"\begin{tabular}{llrr}",
        r"\toprule",
        r"Condition & Group & EER \% & $n$ gen/imp \\",
        r"\midrule",
    ]
    for f in results.report.fairness:
        lines.append(
            f"{_tex(f.condition)} & {_tex(f.group)} & {f.eer * 100:.2f} & "
            f"{f.n_genuine}/{f.n_impostor} \\\\"
        )
    lines += [r"\bottomrule", r"\end{tabular}"]
    return "\n".join(lines) + "\n"


def coverage_table(results: Results) -> str:
    """Per-class token counts -- the §5.4 resource-contribution table."""
    lines = _provenance_note(results)
    lines += [
        r"\begin{tabular}{lrr}",
        r"\toprule",
        r"Semantic class & Choice tokens & Speakers with own evidence \\",
        r"\midrule",
    ]
    for c in results.coverage.classes:
        if c.n_tokens == 0:
            continue
        lines.append(
            f"{_tex(c.semantic_class.value.replace('_', ' ').title())} & "
            f"{c.n_tokens} & "
            f"{c.n_speakers_with_own_evidence}/{results.coverage.n_speakers} \\\\"
        )
    lines += [
        r"\bottomrule",
        r"\end{tabular}",
        "",
        "% 'Own evidence' means the speaker produced at least "
        f"{results.coverage.min_tokens_for_own_evidence:.0f} tokens of the class,",
        "% the point at which their observations outweigh the backoff prior.",
    ]
    return "\n".join(lines) + "\n"


TABLES = {
    "configurations": configurations_table,
    "ablations": ablations_table,
    "stability": stability_table,
    "fairness": fairness_table,
    "coverage": coverage_table,
}


# --------------------------------------------------------------------------
# Writing
# --------------------------------------------------------------------------


def write_figures(results: Results, out_dir: Path) -> list[Path]:
    """Every figure the paper prints. Missing inputs skip, never crash."""
    from .eval import figures as F

    written: list[Path] = []
    figures_dir = out_dir / "figures"

    if results.report.configurations:
        fig = F.det_curve(results.report.configurations)
        written.append(F.save(fig, figures_dir / "det"))
        F.close(fig)

    scoped = [a for a in results.report.ablations if a.scope == Branch.CSBG.value]
    if scoped:
        fig = F.ablation_bars(scoped, scope=Branch.CSBG.value)
        written.append(F.save(fig, figures_dir / "ablation"))
        F.close(fig)

    if results.report.stability:
        fig = F.stability(results.report.stability)
        written.append(F.save(fig, figures_dir / "stability"))
        F.close(fig)

    return written


def write_heatmap(
    graphs: dict[str, CSBG], out_dir: Path, *, consistency: dict[str, float]
) -> Path | None:
    """The two most contrasting speakers, side by side.

    Most and least consistent rather than the first two in the corpus: the
    figure exists to make the hypothesis legible, and two speakers who happen
    to behave alike make it illegible while looking like a fair sample.
    """
    from .eval import figures as F

    if len(graphs) < 2 or len(consistency) < 2:
        return None
    ordered = sorted(consistency, key=lambda s: consistency[s])
    low, high = ordered[0], ordered[-1]
    fig = F.csbg_heatmap([graphs[high], graphs[low]], labels=[high, low])
    path = F.save(fig, out_dir / "figures" / "csbg_heatmap")
    F.close(fig)
    return path


def write(results: Results) -> Path:
    """Write `results.json`, `tables/*.tex`, `figures/*` and a README."""
    out_dir = results.config.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    results_path = out_dir / "results.json"
    results_path.write_text(
        json.dumps(results.to_dict(), indent=2, ensure_ascii=False, allow_nan=False),
        encoding="utf-8",
    )

    tables_dir = out_dir / "tables"
    tables_dir.mkdir(parents=True, exist_ok=True)
    for name, builder in TABLES.items():
        (tables_dir / f"{name}.tex").write_text(builder(results), encoding="utf-8")

    (out_dir / "report.md").write_text(
        results.report.to_markdown()
        + "\n\n"
        + results.coverage.to_markdown()
        + "\n",
        encoding="utf-8",
    )

    if results.config.figures:
        write_figures(results, out_dir)
        if results.graphs:
            write_heatmap(results.graphs, out_dir, consistency=results.consistency)

    (out_dir / "README.md").write_text(_readme(results), encoding="utf-8")
    return results_path


def _readme(results: Results) -> str:
    env = results.environment
    status = (
        "These numbers are reportable."
        if results.reportable
        else "**These numbers are NOT reportable.**"
    )
    lines = [
        f"# Experiment run {env['generated_utc']}",
        "",
        status,
        "",
        f"- commit `{env['git_commit'][:12]}`"
        + (" **(working tree dirty)**" if env["git_dirty"] else ""),
        f"- python {env['python']} on {env['platform']}",
        f"- corpus `{results.corpus['name']}` ({results.corpus['provenance']}), "
        f"{results.corpus['n_speakers']} speakers, "
        f"{results.corpus['n_utterances']} utterances",
        f"- seed {results.config.seed}, dev fraction {results.config.dev_fraction}, "
        f"{results.config.bootstrap} bootstrap resamples",
        "",
    ]
    if results.blockers:
        lines += ["## Why these numbers may not be reported", ""]
        lines += [f"{i}. {b}" for i, b in enumerate(results.blockers, 1)]
        lines += [""]
    lines += [
        "## Files",
        "",
        "| Path | What it is |",
        "|---|---|",
        "| `results.json` | Every number. **The paper reads from here.** |",
        "| `report.md` | The same run as prose tables |",
        "| `tables/*.tex` | `tabular` bodies for `\\input{}` |",
        "| `figures/*.pdf` | Vector figures; `.png` beside each |",
        "",
        "Nothing here is hand-edited. Re-run "
        "`python -m kavach.experiments` to regenerate.",
    ]
    return "\n".join(lines) + "\n"


# --------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m kavach.experiments",
        description="Produce every number and figure in the KAVACH paper.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Without --manifest the run uses simulation.py, and every output is "
            "stamped unreportable. That is not a limitation to work around: "
            "simulated speakers differ by construction, so separating them "
            "measures the implementation, not the hypothesis."
        ),
    )
    parser.add_argument("--out", type=Path, default=Path("paper/results"),
                        help="output directory (default: paper/results)")
    parser.add_argument("--manifest", type=Path, default=None,
                        help="corpus manifest.json; omit to use simulation.py")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--dev-fraction", type=float, default=0.4)
    parser.add_argument("--bootstrap", type=int, default=500,
                        help="bootstrap resamples for the EER interval")
    parser.add_argument("--speakers", type=int, default=24,
                        help="simulated speakers, when no manifest is given")
    parser.add_argument("--no-cohort-norm", action="store_true")
    parser.add_argument("--no-figures", action="store_true")
    parser.add_argument(
        "--within-session", action="store_true",
        help="split enrolment from probes inside one sitting for speakers with "
             "only one; marks the run unreportable. For a first-sitting pilot, "
             "where the alternative is no number at all",
    )
    parser.add_argument(
        "--simulate-branches", action="store_true",
        help="stand-ins for the acoustic and knowledge branches; marks the run "
             "unreportable, and exists to exercise fusion without models",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    config = ExperimentConfig(
        out_dir=args.out,
        manifest=args.manifest,
        seed=args.seed,
        dev_fraction=args.dev_fraction,
        bootstrap=args.bootstrap,
        cohort_norm=not args.no_cohort_norm,
        allow_within_session=args.within_session,
        figures=not args.no_figures,
        simulate_branches=args.simulate_branches,
        n_simulated_speakers=args.speakers,
    )

    try:
        results = run(config)
    except ValueError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2

    path = write(results)

    print(results.report.to_markdown())
    print()
    print(f"wrote {path.parent}/")
    if results.blockers:
        print()
        print("NOT REPORTABLE:", file=sys.stderr)
        for blocker in results.blockers:
            print(f"  - {blocker}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "RESULTS_VERSION",
    "TABLES",
    "ExperimentConfig",
    "Results",
    "ablations_table",
    "build_parser",
    "configurations_table",
    "coverage_table",
    "fairness_table",
    "load_corpus",
    "main",
    "run",
    "stability_table",
    "write",
    "write_figures",
    "write_heatmap",
]
