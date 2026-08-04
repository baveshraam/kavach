"""Attack generation and detection for the spoofing evaluation.

WHY THIS EXISTS
---------------
A voice authentication paper that only reports EER on genuine-vs-impostor
trials has not shown the system is secure -- it has shown it can tell two
cooperative people apart. The claim KAVACH actually makes is stronger:

    a code-switching profile is hard to steal even when the voice and the
    secret have both already been stolen.

That claim is only testable against attackers who *have* stolen the voice and
the secret. This package builds them.

THE THREAT MODEL
----------------
Five attackers of increasing capability, each strictly stronger than the last::

    A1  replay          has a recording of the victim
    A2  splice          has recordings, cuts them into a new answer
    A3  clone           has ~30s of victim audio, no knowledge of the answer
    A4  clone+knowledge A3 plus the answer, scraped from social media
    A5  style-adaptive  A4 plus an LLM told to imitate the victim's switching

A4 is the headline. If the acoustic branch and the knowledge branch both
accept a clone that knows the answer, and the CSBG rejects it, the
contribution is real. A5 is the credibility test: a reviewer will ask what
happens when the attacker also copies the style, so it gets run and reported
whatever the outcome.

A5 IS SPLIT IN TWO, AND THAT SPLIT IS A RESULT
----------------------------------------------
An attacker cannot read the victim's enrolled CSBG -- that lives inside the
authentication server. They can only estimate it from speech they can
overhear. So A5 runs in two conditions:

    A5-oracle    attacker is handed the victim's exact enrolled CSBG.
                 Physically unrealisable; it is the *upper bound* on attacker
                 power and answers "is this defence breakable in principle?"

    A5-observed  attacker estimates the CSBG from N seconds of public speech
                 (a social-media video). This is the realistic condition.

Sweeping N in A5-observed produces the same curve as the enrolment-stability
experiment, read from the other side: **the speech a defender needs to enrol a
usable CSBG is the speech an attacker needs to steal one.** Whether that
number is 30 seconds or 5 minutes decides whether the defence is practical,
and it is one measurement answering both questions.

SIMULATED SIGNALS ARE NOT RECORDED SIGNALS
------------------------------------------
`replay.simulate_replay` and the splice level-matching apply signal
processing that *approximates* a loudspeaker-to-microphone chain. They are for
developing and unit-testing the detectors before the lab session. A number
from a simulated replay is not a replay result, and the paper must report
attacks recorded through real hardware. Every simulated artefact carries
`provenance["simulated"] = True`; `suite.AttackSuite.report()` refuses to mark
a table row as paper-ready while any simulated trial is in it.

ETHICS AND SCOPE
----------------
This is anti-spoofing evaluation in the ASVspoof tradition: attacks are built
solely to measure a defence, against consenting enrolled participants, on a
system the authors control. Cloned audio stays in `data/attacks/` (git-ignored
alongside the biometric data), is never played to a third party, and is
destroyed with the corpus under the consent terms in the project's ethics
section. The style-adaptation prompt in `text.py` needs the victim's CSBG,
which only the system owner holds -- it is not an attack anyone can mount from
outside.
"""

from __future__ import annotations

from enum import Enum


class AttackType(str, Enum):
    """The five threat-model attackers, ordered by capability."""

    A1_REPLAY = "A1_replay"
    A2_SPLICE = "A2_splice"
    A3_CLONE = "A3_clone"
    A4_CLONE_KNOWLEDGE = "A4_clone_knowledge"
    A5_STYLE_ADAPTIVE = "A5_style_adaptive"

    @property
    def label(self) -> str:
        """Short human-readable name for tables and the UI."""
        return _LABELS[self]

    @property
    def has_victim_voice(self) -> bool:
        """True when the attacker can produce audio in the victim's voice.

        All five can, which is the point: the acoustic branch is assumed
        compromised throughout. The attacks differ in what they know and how
        they speak, not in whose voice comes out.
        """
        return True

    @property
    def has_answer(self) -> bool:
        """True when the attacker knows the correct answer to the challenge.

        The A3 -> A4 step. This is what makes A4 the headline: it is the first
        attacker the knowledge branch cannot stop.
        """
        return self in (AttackType.A4_CLONE_KNOWLEDGE, AttackType.A5_STYLE_ADAPTIVE)

    @property
    def adapts_style(self) -> bool:
        """True when the attacker attempts to imitate the victim's CSBG."""
        return self is AttackType.A5_STYLE_ADAPTIVE

    @property
    def is_synthetic_speech(self) -> bool:
        """True when the audio is TTS rather than recorded human speech.

        A1 and A2 are made of the victim's real voice, so a synthesis
        detector cannot see them -- which is exactly why they are in the
        threat model despite being the least sophisticated attacks.
        """
        return self in (
            AttackType.A3_CLONE,
            AttackType.A4_CLONE_KNOWLEDGE,
            AttackType.A5_STYLE_ADAPTIVE,
        )


_LABELS: dict[AttackType, str] = {
    AttackType.A1_REPLAY: "Replay",
    AttackType.A2_SPLICE: "Cut-and-paste splice",
    AttackType.A3_CLONE: "Voice clone, no knowledge",
    AttackType.A4_CLONE_KNOWLEDGE: "Voice clone + knowledge",
    AttackType.A5_STYLE_ADAPTIVE: "Style-adaptive clone",
}


class StyleSource(str, Enum):
    """Where an A5 attacker got the victim's code-switching style.

    See the A5 discussion in the module docstring: `ORACLE` is the
    unrealisable upper bound, `OBSERVED` is the condition an actual attacker
    can reach.
    """

    ORACLE = "oracle"
    """Handed the victim's enrolled CSBG. Upper bound on attacker power."""

    OBSERVED = "observed"
    """Estimated from a limited sample of overheard public speech."""

    NONE = "none"
    """No style information. The attacker speaks in their own idiolect."""


# Re-exported below the enum definitions on purpose: the submodules import
# AttackType and StyleSource from this package, so they must already exist in
# the namespace by the time these lines run.
from .clone import (  # noqa: E402
    CLONE_ATTACKS,
    CloneBackend,
    CloneBatchStats,
    CloneQualityReport,
    EchoCloner,
    SynthesisRequest,
    XTTSCloner,
    check_language_support,
    screen_clone,
)
from .replay import (  # noqa: E402
    ReplayChannel,
    ReplayDetector,
    ReplayReport,
    SpectralCues,
    cue_separation,
    fit_spectral_thresholds,
    simulate_replay,
)
from .splice import (  # noqa: E402
    SpliceConfig,
    SpliceReport,
    detect_clicks,
    detect_digital_silence,
    detect_splice,
    splice_segments,
)
from .suite import (  # noqa: E402
    AttackSuite,
    AttackTable,
    AttackTrial,
    CellResult,
    SystemConfig,
    TrialOutcome,
    policy_for,
    wilson_interval,
)
from .text import (  # noqa: E402
    AttackerAnswer,
    AttackTextGenerator,
    StyleProfile,
    describe_style,
    estimate_style_from_speech,
)

__all__ = [
    "AttackType",
    "StyleSource",
    # clone
    "CLONE_ATTACKS",
    "CloneBackend",
    "CloneBatchStats",
    "CloneQualityReport",
    "EchoCloner",
    "SynthesisRequest",
    "XTTSCloner",
    "check_language_support",
    "screen_clone",
    # replay
    "ReplayChannel",
    "ReplayDetector",
    "ReplayReport",
    "SpectralCues",
    "cue_separation",
    "fit_spectral_thresholds",
    "simulate_replay",
    # splice
    "SpliceConfig",
    "SpliceReport",
    "detect_clicks",
    "detect_digital_silence",
    "detect_splice",
    "splice_segments",
    # suite
    "AttackSuite",
    "AttackTable",
    "AttackTrial",
    "CellResult",
    "SystemConfig",
    "TrialOutcome",
    "policy_for",
    "wilson_interval",
    # text
    "AttackTextGenerator",
    "AttackerAnswer",
    "StyleProfile",
    "describe_style",
    "estimate_style_from_speech",
]
