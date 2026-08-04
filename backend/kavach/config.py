"""Central configuration.

Every tunable in one place so the paper can report exact settings and a
reviewer can reproduce them. Values are overridable by environment variable
(prefix `KAVACH_`) or a `.env` file.

Thresholds here are *starting points derived from reasoning*, not fitted
values. Anything used in a reported result must be fitted on a dev split and
the fitted value reported -- see `kavach.fusion.calibration`. Tuning a
threshold on the test speakers and reporting the result would leak.
"""

from __future__ import annotations

from pathlib import Path

from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

_REPO_ROOT = Path(__file__).resolve().parents[2]


class Settings(BaseSettings):
    """KAVACH runtime configuration."""

    model_config = SettingsConfigDict(
        env_prefix="KAVACH_",
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # ------------------------------------------------------------- storage
    data_dir: Path = Field(default=_REPO_ROOT / "data")
    audio_dir: Path = Field(default=_REPO_ROOT / "data" / "raw")
    attack_dir: Path = Field(default=_REPO_ROOT / "data" / "attacks")
    db_path: Path = Field(default=_REPO_ROOT / "data" / "kavach.db")

    # ----------------------------------------------------------------- ASR
    whisper_model: str = "large-v3"
    whisper_compute_type: str = "int8"
    """int8 keeps large-v3 viable on CPU. Use float16 on a GPU."""

    whisper_device: str = "auto"
    whisper_language: str | None = None
    """None lets Whisper auto-detect. For code-mixed Tamil-English, forcing
    'ta' generally produces better Tamil but can suppress English segments --
    evaluate both on your own audio and report which you used."""

    suppress_numerals: bool = True
    """Emit number words rather than digits. Digits are language-neutral, so
    '5' loses the fact that the speaker chose English or Tamil to say it --
    and NUMBER is one of the most discriminative classes. See
    lid.rules.tag_token."""

    # ----------------------------------------------------- speaker embedding
    ecapa_model: str = "speechbrain/spkrec-ecapa-voxceleb"
    embedding_device: str = "cpu"
    target_sample_rate: int = 16_000
    min_audio_seconds: float = 1.0
    """Below this, an ECAPA embedding is too unstable to verify against."""

    max_audio_seconds: float = 30.0

    # ----------------------------------------------------------------- LLM
    llm_model: str = "claude-opus-5"
    llm_tagging_effort: str = "low"
    """Tagging is well-specified and high-volume. Thinking stays ON --
    disabling it on Opus 5 risks internal tags leaking into output; lowering
    effort saves the same cost without that failure mode."""

    llm_challenge_effort: str = "medium"
    """Challenge generation needs more care: the question must be natural,
    answerable from the SKG, and target a specific semantic class."""

    # ------------------------------------------------------------- decision
    #: STARTING POINTS ONLY. Fit on a dev split before reporting anything.
    speaker_threshold: float = 0.62
    """ECAPA cosine similarity. ~0.6-0.7 is the usual operating range for
    this checkpoint on clean audio; phone recordings sit lower."""

    csbg_threshold: float = 0.0
    """Cohort-normalised LLR. 0.0 = 'as likely this speaker as the average
    impostor', the natural neutral point for a z-normed LLR."""

    knowledge_threshold: float = 0.70
    """Cross-lingual answer match. Deliberately not 1.0 -- exact match fails
    constantly under code-mixed ASR, which is the point of the matcher."""

    fused_threshold: float = 0.55
    borderline_margin: float = 0.05
    """Scores within this of the threshold are reported BORDERLINE rather than
    a hard accept/reject. An honest 'not sure' beats a coin flip on a
    security decision."""

    # ------------------------------------------------------------- CSBG
    lid_confidence_floor: float = 0.0
    """Drop language-choice tokens below this LID confidence. 0.0 keeps
    everything. Raising it trades coverage for annotation quality -- run the
    ablation before committing to a value."""

    min_enrolment_seconds: float = 180.0
    """Below 3 minutes the CSBG is too sparse to be reliable. Enrolment warns
    rather than blocks -- the stability analysis (proposal 5.3) exists to
    replace this guess with a measured number."""

    sparse_class_threshold: float = 5.0
    min_scored_tokens: int = 5
    """Below this the CSBG branch reports insufficient_evidence and fusion
    falls back to the other branches rather than trusting a noisy score."""

    # -------------------------------------------------------- challenge
    challenge_ttl_seconds: int = 60
    """A challenge must expire, or a captured one could be replayed later.
    Long enough for a real answer, short enough to bound the attack window."""

    max_challenge_reuse: int = 0
    """0 = never reuse a challenge for a speaker. Reuse would let an attacker
    who observed one login replay the answer."""

    # ------------------------------------------------------------- API
    api_host: str = "127.0.0.1"
    api_port: int = 8000
    cors_origins: list[str] = Field(default_factory=lambda: ["http://localhost:5173"])

    @field_validator("data_dir", "audio_dir", "attack_dir")
    @classmethod
    def _resolve(cls, v: Path) -> Path:
        return v.expanduser().resolve()

    def ensure_dirs(self) -> None:
        """Create storage directories. Safe to call repeatedly."""
        for d in (self.data_dir, self.audio_dir, self.attack_dir):
            d.mkdir(parents=True, exist_ok=True)

    def reportable(self) -> dict[str, object]:
        """The settings that belong in the paper's reproducibility section."""
        return {
            "whisper_model": self.whisper_model,
            "whisper_language": self.whisper_language,
            "suppress_numerals": self.suppress_numerals,
            "ecapa_model": self.ecapa_model,
            "llm_model": self.llm_model,
            "speaker_threshold": self.speaker_threshold,
            "csbg_threshold": self.csbg_threshold,
            "knowledge_threshold": self.knowledge_threshold,
            "fused_threshold": self.fused_threshold,
            "lid_confidence_floor": self.lid_confidence_floor,
            "min_enrolment_seconds": self.min_enrolment_seconds,
            "challenge_ttl_seconds": self.challenge_ttl_seconds,
        }


_settings: Settings | None = None


def get_settings() -> Settings:
    """Process-wide settings singleton."""
    global _settings
    if _settings is None:
        _settings = Settings()
    return _settings


def reset_settings() -> None:
    """Clear the singleton. Tests only."""
    global _settings
    _settings = None


__all__ = ["Settings", "get_settings", "reset_settings"]
