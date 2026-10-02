from __future__ import annotations

import pytest

from clone_helpers import add_clip, tone
from kavach.api.attacks import run_attack
from kavach.api.pipeline import Pipeline
from kavach.api.store import Store
from kavach.attacks import AttackType
from kavach.attacks.bank import CloneBank
from kavach.audio import save_wav
from kavach.config import Settings
from test_api import ENGLISH_NUMBERS, TAMIL_NUMBERS, enrol_tokens, speaker_utterances

VICTIM = "S04"


def bank_at(settings, victim):
    return CloneBank.new(
        settings.attack_dir / "clones" / VICTIM, VICTIM, victim, allowed_victims=[VICTIM]
    )


def lab(tmp_path, *, bank_flag=True, clips=5, annotated=True, **clip_kw):
    settings = Settings(
        data_dir=tmp_path,
        audio_dir=tmp_path / "raw",
        attack_dir=tmp_path / "attacks",
        db_path=tmp_path / "kavach.db",
        demo_attack_bank=bank_flag,
        clone_victims=[VICTIM],
    )
    store = Store(settings.db_path, settings.audio_dir)
    victim = store.create_speaker({"display_name": f"{VICTIM} · Victim"})["id"]
    enrol_tokens(store, victim, speaker_utterances(TAMIL_NUMBERS, n=10))
    for i in range(3):
        other = store.create_speaker({"display_name": f"Other{i}"})["id"]
        enrol_tokens(store, other, speaker_utterances(ENGLISH_NUMBERS, n=10))
    pipeline = Pipeline(store, settings)
    bank = bank_at(settings, victim)
    for _ in range(clips):
        add_clip(bank, annotated=annotated, similarity=0.9, **clip_kw)
    bank.save()
    return store, pipeline, settings, victim, bank


def a4(store, pipeline, victim, trials=40):
    return run_attack(
        attack=AttackType.A4_CLONE_KNOWLEDGE,
        speaker_id=victim,
        trials=trials,
        store=store,
        pipeline=pipeline,
    )


def test_a4_uses_the_bank_and_says_so(tmp_path) -> None:
    store, pipeline, _, victim, _ = lab(tmp_path)
    run = a4(store, pipeline, victim)
    assert run.acoustic_source == "measured"
    assert any("Measured, not modelled" in n for n in run.notes)


def test_trials_are_capped_at_the_number_of_distinct_clips(tmp_path) -> None:
    """Resampling five clips to forty trials would give an interval narrower
    than the evidence."""
    store, pipeline, _, victim, _ = lab(tmp_path, clips=5)
    run = a4(store, pipeline, victim, trials=40)
    assert run.trials <= 5
    assert any("capped" in n.lower() or "distinct clips" in n for n in run.notes)


def test_yield_is_printed_with_the_row(tmp_path) -> None:
    store, pipeline, _, victim, _ = lab(tmp_path)
    run = a4(store, pipeline, victim)
    assert any("Attack yield" in n for n in run.notes)
    assert run.yield_rate is not None


def test_the_run_is_still_simulated_and_not_paper_ready(tmp_path) -> None:
    store, pipeline, _, victim, _ = lab(tmp_path)
    run = a4(store, pipeline, victim)
    assert run.simulated is True
    assert any("paper" in n.lower() for n in run.notes)


def test_without_the_flag_the_run_is_modelled(tmp_path) -> None:
    store, pipeline, _, victim, _ = lab(tmp_path, bank_flag=False)
    assert a4(store, pipeline, victim).acoustic_source == "modelled"


def test_a3_stays_modelled_because_the_bank_has_only_a4_clips(tmp_path) -> None:
    store, pipeline, _, victim, _ = lab(tmp_path)
    run = run_attack(
        attack=AttackType.A3_CLONE, speaker_id=victim, trials=20, store=store, pipeline=pipeline
    )
    assert run.acoustic_source == "modelled"


def test_unannotated_clips_are_not_used(tmp_path) -> None:
    store, pipeline, _, victim, _ = lab(tmp_path, annotated=False)
    assert a4(store, pipeline, victim).acoustic_source == "modelled"


def test_a_tampered_bank_is_reported_not_silently_replaced(tmp_path) -> None:
    store, pipeline, settings, victim, bank = lab(tmp_path)
    save_wav(tone(seed=42), bank.audio_file(bank.clips[0]))
    run = a4(store, pipeline, victim)
    assert run.acoustic_source == "modelled"
    assert any("could not be used" in n for n in run.notes), run.notes


def test_a_bank_for_another_speaker_is_not_used(tmp_path) -> None:
    store, pipeline, settings, victim, bank = lab(tmp_path)
    other = store.create_speaker({"display_name": "S09 · Someone"})["id"]
    enrol_tokens(store, other, speaker_utterances(TAMIL_NUMBERS, n=10))
    run = run_attack(
        attack=AttackType.A4_CLONE_KNOWLEDGE,
        speaker_id=other,
        trials=20,
        store=store,
        pipeline=pipeline,
    )
    assert run.acoustic_source == "modelled"
