"""Builders shared by the clone-bank tests. Not a test module."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from kavach.attacks import AttackType
from kavach.attacks.bank import AUDIO_DIR, CloneBank, CloneClip, sha256_file
from kavach.audio import Audio, save_wav
from kavach.csbg.ontology import Language, SemanticClass
from kavach.csbg.tokens import Token

SR = 16_000


def tone(seconds: float = 3.0, freq: float = 140.0, amp: float = 0.3, seed: int = 0) -> Audio:
    rng = np.random.default_rng(seed)
    t = np.arange(int(SR * seconds)) / SR
    x = amp * np.sin(2 * np.pi * freq * t) + 0.01 * rng.standard_normal(len(t))
    return Audio(x.astype(np.float32), SR, f"tone{freq}")


def token_dicts() -> list[dict]:
    """Stored-token wire dicts, built from the real enums so values cannot drift."""
    toks = [
        Token("naan", Language.TA, SemanticClass.OTHER, 0.9, 0, 300),
        Token("hometown", Language.EN, SemanticClass.OTHER, 0.9, 300, 800),
    ]
    return [
        {
            "text": t.text,
            "language": t.language.value,
            "semanticClass": t.semantic_class.value,
            "lidConfidence": t.lid_confidence,
            "startMs": t.start_ms,
            "endMs": t.end_ms,
        }
        for t in toks
    ]


def new_bank(tmp_path: Path, victim: str = "S04", speaker_id: str = "spk_victim") -> CloneBank:
    return CloneBank.new(
        tmp_path / "clones" / victim, victim, speaker_id, allowed_victims=[victim]
    )


def add_clip(
    bank: CloneBank,
    *,
    fact_key: str | None = "hometown",
    attack: AttackType = AttackType.A4_CLONE_KNOWLEDGE,
    annotated: bool = True,
    similarity: float = 0.8,
    admissible: bool = True,
    answer_score: float | None = 0.9,
    flags: list[str] | None = None,
    speaker: str = "attacker_1",
) -> CloneClip:
    clip_id = f"clone_{len(bank.clips):08x}"
    rel = f"{AUDIO_DIR}/{clip_id}__SYNTHETIC.wav"
    path = bank.root / rel
    save_wav(tone(seed=len(bank.clips)), path)
    clip = CloneClip(
        clip_id=clip_id,
        attack=attack,
        backend="fake",
        audio_path=rel,
        sha256=sha256_file(path),
        duration_sec=3.0,
        created_at="2026-10-02T00:00:00+00:00",
        fact_key=fact_key,
        source={"kind": "teammate_speech", "file": f"{fact_key}.wav", "speaker": speaker},
    )
    if annotated:
        clip.transcript = "naan hometown Thanjavur"
        clip.tokens = token_dicts()
        clip.annotation_source = "LLM"
        clip.ecapa_similarity = similarity
        clip.threshold = 0.62
        clip.admissible = admissible
        clip.answer_score = answer_score
        clip.flags = list(flags or [])
    bank.add(clip)
    return clip
