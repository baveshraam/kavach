"""The A1-A5 evaluation harness -- the table the paper is built around.

WHAT IT PRODUCES
----------------
For each attack and each system configuration, the fraction of attacks the
system accepted. In ISO/IEC 30107-3 terms that is the **IAPMR** (Impostor
Attack Presentation Match Rate); in plain English, how often the attacker got
in. Three configurations, chosen so each column adds exactly one branch::

    ECAPA only      the standard baseline: acoustic speaker verification
    + knowledge     add the SKG challenge-response
    + CSBG (full)   add the code-switching profile   <- the contribution

Reading down the A4 row of that table is the paper's central claim. Reading
the A5 row is the paper's honesty.

WHAT IT DOES NOT DO
-------------------
It does not run ECAPA, ASR, or the CSBG scorer. Trials arrive with their
branch scores already computed, and the suite's only job is to fuse them
three ways and count. That split keeps the harness testable without any model
loaded, and keeps the expensive pipeline from being re-run once per column --
the three configurations must see *identical* branch scores, or the columns
are not comparable and the whole table is meaningless.

GUARDS AGAINST ACCIDENTALLY FAKING THE RESULT
---------------------------------------------
Four ways this table could look good without being true, each checked by
`AttackTable.paper_ready()`:

1. **Simulated attacks.** A signal-processed pseudo-replay is not a replay.
   Any trial with `simulated=True` disqualifies its row.
2. **Inadmissible clones.** A clone the acoustic branch rejects on its own
   never tested the CSBG. Those trials are excluded from the rate and
   surface separately as attack yield.
3. **Template-written A5.** A5 is a claim about a capable adversary; a
   fixed-template answer is not one.
4. **Too few trials.** With 10 trials a cell, 0% and 20% are the same
   measurement. Every rate carries a Wilson interval and cells below
   `MIN_TRIALS_PER_CELL` are flagged.

None of these block the run. They annotate the output, so a table can be
inspected mid-study and still refuse to claim more than it has earned.
"""

from __future__ import annotations

import math
from collections import defaultdict
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Iterable

from ..fusion import Branch, BranchScore, Decision, FusionPolicy, FusionResult, fuse
from . import AttackType, StyleSource

#: Below this, a per-cell rate is too noisy to state. 30 gives a Wilson
#: half-width of roughly +/-10 points at the extremes, which is about the
#: coarsest an attack claim can be and still mean something.
MIN_TRIALS_PER_CELL = 30


class SystemConfig(str, Enum):
    """The columns of the attack table.

    `ECAPA_ONLY -> ECAPA_KNOWLEDGE -> FULL` is the cumulative progression the
    headline table reads down. `ECAPA_CSBG` is off that path: it pairs the
    contribution with the baseline and *omits* the knowledge branch, which is
    the only way to see what the CSBG contributes on its own rather than on
    top of a challenge-response protocol that is already stopping most
    attacks. It answers the reviewer question "how much of this is just the
    knowledge check?"
    """

    ECAPA_ONLY = "ecapa_only"
    ECAPA_KNOWLEDGE = "ecapa_knowledge"
    ECAPA_CSBG = "ecapa_csbg"
    FULL = "full"

    @property
    def label(self) -> str:
        return {
            SystemConfig.ECAPA_ONLY: "ECAPA alone",
            SystemConfig.ECAPA_KNOWLEDGE: "+ Knowledge",
            SystemConfig.ECAPA_CSBG: "+ CSBG only",
            SystemConfig.FULL: "Full fusion",
        }[self]

    @property
    def branches(self) -> tuple[Branch, ...]:
        return {
            SystemConfig.ECAPA_ONLY: (Branch.SPEAKER,),
            SystemConfig.ECAPA_KNOWLEDGE: (Branch.SPEAKER, Branch.KNOWLEDGE),
            SystemConfig.ECAPA_CSBG: (Branch.SPEAKER, Branch.CSBG),
            SystemConfig.FULL: (Branch.SPEAKER, Branch.KNOWLEDGE, Branch.CSBG),
        }[self]

    @property
    def uses_liveness(self) -> bool:
        """Liveness rides with the knowledge branch, and only with it.

        Both come from the challenge-response protocol: a configuration
        without a challenge has no freshness binding either. Giving the
        ECAPA-only baseline a liveness gate would credit it with machinery no
        deployed baseline has; giving it to `ECAPA_CSBG` would contaminate the
        one column meant to isolate the CSBG, since a replay stopped by
        freshness would be scored as a CSBG success.
        """
        return Branch.KNOWLEDGE in self.branches


def policy_for(config: SystemConfig, base: FusionPolicy | None = None) -> FusionPolicy:
    """Build a fusion policy restricted to one configuration's branches.

    Weights are taken from `base` and renormalised over the active branches,
    so the columns differ only in which evidence is available -- not in how
    the shared evidence is weighted. The threshold is carried across
    unchanged.

    Note the threshold really should be re-tuned per configuration on a dev
    split (fewer branches means a differently-distributed fused score). Doing
    that is `eval.ablation`'s job; passing a `base` whose threshold was fitted
    for that configuration is the correct usage here.
    """
    base = base or FusionPolicy()
    active = {b: base.weights.get(b, 0.0) for b in config.branches if b in base.weights}
    total = sum(active.values())
    if total <= 0:
        # Configuration's branches carry no weight in the base policy; fall
        # back to uniform rather than dividing by zero.
        active = {b: 1.0 / len(config.branches) for b in config.branches}
    else:
        active = {b: w / total for b, w in active.items()}

    return FusionPolicy(
        weights=active,
        threshold=base.threshold,
        borderline_margin=base.borderline_margin,
        liveness_is_gate=base.liveness_is_gate,
        require_knowledge=base.require_knowledge,
        # Vetoes are restricted to branches this configuration actually has,
        # so the baseline columns cannot be credited with a mechanism they do
        # not include. Attributing the CSBG veto to "ECAPA alone" would make
        # the baseline look stronger than any deployed baseline is.
        veto_thresholds={
            b: t for b, t in base.veto_thresholds.items() if b in config.branches
        },
    )


@dataclass(slots=True)
class AttackTrial:
    """One attack attempt, with every branch already scored.

    Scores are on each branch's own scale; the thresholds travel with them so
    a trial is self-contained and a stored trial log can be re-fused later
    without re-running the pipeline.
    """

    trial_id: str
    attack: AttackType
    speaker_id: str

    speaker_score: float
    speaker_threshold: float
    csbg_score: float
    csbg_threshold: float
    knowledge_score: float
    knowledge_threshold: float
    liveness_ok: bool = True

    admissible: bool = True
    """False when the clone never fooled the acoustic branch. Excluded from
    rates, counted in yield. See clone.screen_clone."""

    simulated: bool = False
    """True for signal-processed pseudo-attacks. Disqualifies the row from
    being paper-ready."""

    style_source: StyleSource = StyleSource.NONE
    text_generator: str = ""
    """'llm' or 'template'. A5 requires 'llm'."""

    csbg_reliable: bool = True
    """False when the probe was too short for a trustworthy CSBG score. The
    branch is then marked unavailable rather than scored, so a terse answer
    cannot be mistaken for an atypical one."""

    integrity_score: float | None = None
    """Edit-artefact evidence, when real audio for this attack exists.

    None means the check could not run -- which is the honest state for the
    clone attacks, because there is no cloner here to produce a waveform to
    inspect. A1 and A2 *can* be built from the victim's stored recordings, so
    for those this is a measurement rather than a model, and it is the one
    column in this lab that does not need the `simulated` caveat."""

    integrity_threshold: float = 0.25
    """Default mirrors `integrity.INTEGRITY_FLOOR`. Not imported from there,
    so that a stored trial log carries the floor it was actually scored
    against rather than whatever the constant says when it is re-read."""

    provenance: dict[str, Any] = field(default_factory=dict)

    def branch_scores(
        self, config: SystemConfig, policy: FusionPolicy | None = None
    ) -> list[BranchScore]:
        """Branch scores as `fusion.fuse` wants them, for one configuration.

        `fuse` reads weights from the policy, not from these objects, but the
        weights are filled in anyway so a serialised branch report shows what
        actually drove the decision rather than a row of zeros.
        """
        active = config.branches
        weights = policy.weights if policy else {}
        scores: list[BranchScore] = []

        if Branch.SPEAKER in active:
            scores.append(
                BranchScore(
                    branch=Branch.SPEAKER,
                    score=self.speaker_score,
                    threshold=self.speaker_threshold,
                    weight=weights.get(Branch.SPEAKER, 0.0),
                    detail="ECAPA-TDNN cosine similarity to the enrolled template.",
                )
            )
        if Branch.KNOWLEDGE in active:
            scores.append(
                BranchScore(
                    branch=Branch.KNOWLEDGE,
                    score=self.knowledge_score,
                    threshold=self.knowledge_threshold,
                    weight=weights.get(Branch.KNOWLEDGE, 0.0),
                    detail="Answer match against the speaker knowledge graph.",
                )
            )
        if Branch.CSBG in active:
            scores.append(
                BranchScore(
                    branch=Branch.CSBG,
                    score=self.csbg_score,
                    threshold=self.csbg_threshold,
                    weight=weights.get(Branch.CSBG, 0.0),
                    available=self.csbg_reliable,
                    detail=(
                        "Code-switch behaviour log-likelihood ratio."
                        if self.csbg_reliable
                        else "Probe too short for a reliable CSBG score; branch excluded."
                    ),
                )
            )
        if config.uses_liveness:
            scores.append(
                BranchScore(
                    branch=Branch.LIVENESS,
                    score=1.0 if self.liveness_ok else 0.0,
                    threshold=0.5,
                    weight=0.0,  # gate, not a weighted term
                    detail="Challenge freshness, single-use and within TTL.",
                )
            )
        # Integrity, unlike liveness, is emitted for EVERY configuration --
        # including the ECAPA-only baseline. It is not part of the
        # challenge-response protocol and does not ride with it: edit-artefact
        # tests run on a waveform and need no challenge, no knowledge graph and
        # no CSBG. Any of these four systems could deploy them.
        #
        # The consequence is that the A1 and A2 rows go flat across all four
        # columns, and that flatness is the finding: those attacks are stopped
        # by signal evidence, identically well with or without the research
        # contribution. Folding integrity into `uses_liveness` would have made
        # the contribution columns look better at A2 for a reason that has
        # nothing to do with code-switching.
        scores.append(
            BranchScore(
                branch=Branch.INTEGRITY,
                score=self.integrity_score if self.integrity_score is not None else 0.0,
                threshold=self.integrity_threshold,
                weight=0.0,  # gate, not a weighted term
                available=self.integrity_score is not None,
                detail=(
                    "Edit- and duplicate-artefact tests on the submitted audio."
                    if self.integrity_score is not None
                    else "No audio for this attack; integrity could not be checked."
                ),
            )
        )
        return scores


@dataclass(slots=True)
class TrialOutcome:
    """One trial evaluated under one configuration."""

    trial_id: str
    attack: AttackType
    config: SystemConfig
    result: FusionResult

    @property
    def attacker_succeeded(self) -> bool:
        """Only a clean ACCEPT counts as a breach.

        BORDERLINE escalates to another factor rather than letting anyone in,
        so counting it as success would overstate the attacker. It is
        reported separately -- a configuration that turns breaches into
        borderlines has genuinely improved, and burying that in the accept
        rate would hide it.
        """
        return self.result.decision is Decision.ACCEPT


@dataclass(slots=True)
class CellResult:
    """One (attack, configuration) cell of the table."""

    attack: AttackType
    config: SystemConfig
    n_trials: int = 0
    n_accepted: int = 0
    n_borderline: int = 0
    n_excluded_inadmissible: int = 0
    n_simulated: int = 0

    @property
    def iapmr(self) -> float:
        """Impostor Attack Presentation Match Rate: the fraction accepted.

        ISO/IEC 30107-3. Computed over admissible trials only.
        """
        return self.n_accepted / self.n_trials if self.n_trials else 0.0

    @property
    def borderline_rate(self) -> float:
        return self.n_borderline / self.n_trials if self.n_trials else 0.0

    @property
    def confidence_interval(self) -> tuple[float, float]:
        """95% Wilson interval on the IAPMR.

        Wilson rather than normal-approximation because these rates sit near
        0 and 1, where the normal interval runs outside [0, 1] and is simply
        wrong. It also stays defined at 0/n, which is the cell the paper most
        wants to talk about.
        """
        return wilson_interval(self.n_accepted, self.n_trials)

    @property
    def underpowered(self) -> bool:
        return self.n_trials < MIN_TRIALS_PER_CELL

    def format_rate(self) -> str:
        if not self.n_trials:
            return "--"
        lo, hi = self.confidence_interval
        return f"{self.iapmr:.0%} [{lo:.0%}-{hi:.0%}]"

    def to_dict(self) -> dict[str, Any]:
        lo, hi = self.confidence_interval
        return {
            "attack": self.attack.value,
            "config": self.config.value,
            "n_trials": self.n_trials,
            "n_accepted": self.n_accepted,
            "n_borderline": self.n_borderline,
            "n_excluded_inadmissible": self.n_excluded_inadmissible,
            "n_simulated": self.n_simulated,
            "iapmr": round(self.iapmr, 4),
            "ci_low": round(lo, 4),
            "ci_high": round(hi, 4),
            "underpowered": self.underpowered,
        }


def wilson_interval(successes: int, n: int, z: float = 1.96) -> tuple[float, float]:
    """95% Wilson score interval for a binomial proportion."""
    if n == 0:
        return 0.0, 1.0
    p = successes / n
    denom = 1.0 + z * z / n
    centre = (p + z * z / (2 * n)) / denom
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denom
    return max(0.0, centre - half), min(1.0, centre + half)


@dataclass(slots=True)
class AttackTable:
    """The full attack x configuration table."""

    cells: dict[tuple[AttackType, SystemConfig], CellResult] = field(default_factory=dict)
    yields: dict[AttackType, Any] = field(default_factory=dict)
    """AttackType -> clone.CloneBatchStats, for clone-based attacks."""

    notes: list[str] = field(default_factory=list)

    def cell(self, attack: AttackType, config: SystemConfig) -> CellResult:
        """Get or create the cell for one (attack, configuration) pair."""
        return self.cells.setdefault(
            (attack, config), CellResult(attack=attack, config=config)
        )

    def attacks(self) -> list[AttackType]:
        seen = {a for a, _ in self.cells}
        return [a for a in AttackType if a in seen]

    # ------------------------------------------------------------- validation

    def paper_ready(self) -> tuple[bool, list[str]]:
        """Whether this table can be reported as an experimental result.

        Returns:
            (ready, problems). `problems` is empty only when every check in
            the module docstring passes.
        """
        problems: list[str] = []

        for attack in self.attacks():
            row = [self.cells[(attack, c)] for c in SystemConfig if (attack, c) in self.cells]
            if not row:
                continue

            if any(c.n_simulated for c in row):
                n = max(c.n_simulated for c in row)
                problems.append(
                    f"{attack.label}: {n} simulated trial(s). Signal-processed attacks are "
                    "for detector development; record the attack through real hardware."
                )
            if any(c.underpowered for c in row):
                n = min(c.n_trials for c in row)
                problems.append(
                    f"{attack.label}: only {n} admissible trial(s), below "
                    f"{MIN_TRIALS_PER_CELL}. The interval is too wide to claim a rate."
                )
            if attack in self.yields:
                stats = self.yields[attack]
                if getattr(stats, "yield_rate", 1.0) < 0.5:
                    problems.append(
                        f"{attack.label}: attack yield only "
                        f"{stats.yield_rate:.0%} -- most clones never fooled the acoustic "
                        "branch. Report this as a TTS-quality finding, and say so plainly "
                        "rather than letting it read as a CSBG result."
                    )

        for note in self.notes:
            if note.startswith("A5:template"):
                problems.append(
                    "A5 used template-generated text. A5 is a claim about a capable "
                    "adversary; generate its answers with the LLM path."
                )

        return not problems, problems

    # ------------------------------------------------------------- rendering

    def to_markdown(self) -> str:
        """Render the money table."""
        configs = [c for c in SystemConfig]
        header = "| | Attack | " + " | ".join(c.label for c in configs) + " | Trials |"
        sep = "|---|---|" + "---|" * (len(configs) + 1)
        lines = [header, sep]

        for attack in self.attacks():
            cells = [self.cells.get((attack, c)) for c in configs]
            n = max((c.n_trials for c in cells if c), default=0)
            code = attack.value.split("_")[0]
            rates = " | ".join(c.format_rate() if c else "--" for c in cells)
            lines.append(f"| **{code}** | {attack.label} | {rates} | {n} |")

        lines.append("")
        lines.append("Cells are IAPMR (fraction of attacks accepted) with 95% Wilson intervals.")

        if self.yields:
            lines.append("")
            lines.append("| Attack | Clones attempted | Fooled ECAPA | Yield | Mean similarity |")
            lines.append("|---|---|---|---|---|")
            for attack, stats in self.yields.items():
                lines.append(
                    f"| {attack.label} | {stats.attempted} | {stats.admissible} | "
                    f"{stats.yield_rate:.0%} | {stats.mean_similarity:.3f} |"
                )
            lines.append("")
            lines.append(
                "Yield is the fraction of clones that fooled the acoustic branch. Only those "
                "count as trials above: a clone ECAPA rejects unaided never tested the CSBG."
            )

        ready, problems = self.paper_ready()
        if not ready:
            lines.append("")
            lines.append("**Not yet reportable:**")
            lines.extend(f"- {p}" for p in problems)

        return "\n".join(lines)

    def to_dict(self) -> dict[str, Any]:
        ready, problems = self.paper_ready()
        return {
            "cells": [c.to_dict() for c in self.cells.values()],
            "yields": {a.value: s.to_dict() for a, s in self.yields.items()},
            "paper_ready": ready,
            "problems": problems,
            "notes": self.notes,
        }


class AttackSuite:
    """Runs trials through every configuration and accumulates the table."""

    def __init__(self, policy: FusionPolicy | None = None) -> None:
        self.policy = policy or FusionPolicy()
        self._policies = {c: policy_for(c, self.policy) for c in SystemConfig}
        self._outcomes: list[TrialOutcome] = []
        self._table = AttackTable()
        self._seen_a5_generators: set[str] = set()

    def run(self, trial: AttackTrial) -> list[TrialOutcome]:
        """Evaluate one trial under all three configurations.

        Inadmissible trials are recorded and excluded from the rates -- they
        are yield statistics, not attack outcomes.
        """
        if trial.attack is AttackType.A5_STYLE_ADAPTIVE and trial.text_generator:
            self._seen_a5_generators.add(trial.text_generator)

        outcomes: list[TrialOutcome] = []
        for config in SystemConfig:
            cell = self._table.cell(trial.attack, config)

            if not trial.admissible:
                cell.n_excluded_inadmissible += 1
                continue

            policy = self._policies[config]
            result = fuse(trial.branch_scores(config, policy), policy)
            outcome = TrialOutcome(
                trial_id=trial.trial_id,
                attack=trial.attack,
                config=config,
                result=result,
            )
            outcomes.append(outcome)
            self._outcomes.append(outcome)

            cell.n_trials += 1
            if outcome.attacker_succeeded:
                cell.n_accepted += 1
            elif result.decision is Decision.BORDERLINE:
                cell.n_borderline += 1
            if trial.simulated:
                cell.n_simulated += 1

        return outcomes

    def run_all(self, trials: Iterable[AttackTrial]) -> None:
        for trial in trials:
            self.run(trial)

    def record_yield(self, attack: AttackType, stats: Any) -> None:
        """Attach clone yield statistics for an attack row."""
        self._table.yields[attack] = stats

    def table(self) -> AttackTable:
        """The accumulated table, with generator notes attached."""
        if self._seen_a5_generators == {"template"}:
            note = "A5:template"
            if note not in self._table.notes:
                self._table.notes.append(note)
        return self._table

    def outcomes(self) -> list[TrialOutcome]:
        return list(self._outcomes)

    def per_speaker_breakdown(self, trials: list[AttackTrial]) -> dict[str, dict[str, float]]:
        """IAPMR under the full system, per speaker.

        A mean hides the case that matters: if the full system stops every
        attack on 24 speakers and none on the 25th, that speaker is
        completely unprotected and the mean says 96%. Report the worst
        speaker, not just the average.
        """
        by_speaker: dict[str, list[bool]] = defaultdict(list)
        by_trial = {t.trial_id: t for t in trials}
        for outcome in self._outcomes:
            if outcome.config is not SystemConfig.FULL:
                continue
            trial = by_trial.get(outcome.trial_id)
            if trial is not None:
                by_speaker[trial.speaker_id].append(outcome.attacker_succeeded)

        return {
            speaker: {
                "n_trials": float(len(results)),
                "iapmr": sum(results) / len(results) if results else 0.0,
            }
            for speaker, results in by_speaker.items()
        }


__all__ = [
    "AttackSuite",
    "AttackTable",
    "AttackTrial",
    "CellResult",
    "MIN_TRIALS_PER_CELL",
    "SystemConfig",
    "TrialOutcome",
    "policy_for",
    "wilson_interval",
]
