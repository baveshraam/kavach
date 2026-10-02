"""The bank is the trust boundary for clone clips: what it refuses matters most."""

from __future__ import annotations

import json

import pytest

from clone_helpers import add_clip, new_bank, tone
from kavach.attacks import AttackType
from kavach.attacks.bank import (
    BANK_FILE,
    BankError,
    CloneBank,
    check_allowed,
    resolve_speaker_id,
)
from kavach.audio import save_wav


def reload(bank: CloneBank, **kw) -> CloneBank:
    return CloneBank.load(bank.root, allowed_victims=["S04"], **kw)


class TestRoundTrip:
    def test_save_and_load_keep_everything(self, tmp_path) -> None:
        bank = new_bank(tmp_path)
        add_clip(bank)
        bank.save()
        loaded = reload(bank, expected_speaker_id="spk_victim")
        assert loaded.victim == "S04" and loaded.victim_speaker_id == "spk_victim"
        assert loaded.clips[0].to_dict() == bank.clips[0].to_dict()
        assert loaded.clips[0].attack is AttackType.A4_CLONE_KNOWLEDGE

    def test_save_is_atomic(self, tmp_path) -> None:
        bank = new_bank(tmp_path)
        add_clip(bank)
        bank.save()
        assert not list(bank.root.glob("*.tmp"))

    def test_the_file_carries_the_pseudonym_and_marks_itself_synthetic(self, tmp_path) -> None:
        bank = new_bank(tmp_path)
        add_clip(bank)
        bank.save()
        data = json.loads((bank.root / BANK_FILE).read_text(encoding="utf-8"))
        assert data["victim"] == "S04" and data["synthetic"] is True
        assert data["clips"][0]["attack"] == "A4_clone_knowledge"


class TestRefusals:
    def test_a_victim_not_on_the_allowlist_is_refused(self, tmp_path) -> None:
        bank = new_bank(tmp_path)
        add_clip(bank)
        bank.save()
        with pytest.raises(BankError, match="allowlist"):
            CloneBank.load(bank.root, allowed_victims=["S09"])

    def test_an_empty_allowlist_refuses_everyone(self, tmp_path) -> None:
        with pytest.raises(BankError, match="allowlist"):
            check_allowed("S04", [])

    def test_a_missing_audio_file_is_refused(self, tmp_path) -> None:
        bank = new_bank(tmp_path)
        clip = add_clip(bank)
        bank.save()
        bank.audio_file(clip).unlink()
        with pytest.raises(BankError, match="missing"):
            reload(bank)

    def test_a_changed_audio_file_is_refused(self, tmp_path) -> None:
        """The hash is what proves a clip is the one that was screened."""
        bank = new_bank(tmp_path)
        clip = add_clip(bank)
        bank.save()
        save_wav(tone(seed=99), bank.audio_file(clip))
        with pytest.raises(BankError, match="hash"):
            reload(bank)

    def test_a_bank_for_a_different_speaker_is_refused(self, tmp_path) -> None:
        bank = new_bank(tmp_path)
        add_clip(bank)
        bank.save()
        with pytest.raises(BankError, match="speaker"):
            reload(bank, expected_speaker_id="spk_someone_else")

    def test_an_unknown_bank_version_is_refused(self, tmp_path) -> None:
        bank = new_bank(tmp_path)
        add_clip(bank)
        bank.save()
        path = bank.root / BANK_FILE
        data = json.loads(path.read_text(encoding="utf-8"))
        data["bank_version"] = "99"
        path.write_text(json.dumps(data), encoding="utf-8")
        with pytest.raises(BankError, match="version"):
            reload(bank)

    def test_a_path_that_escapes_the_bank_is_refused(self, tmp_path) -> None:
        bank = new_bank(tmp_path)
        add_clip(bank)
        bank.save()
        path = bank.root / BANK_FILE
        data = json.loads(path.read_text(encoding="utf-8"))
        data["clips"][0]["audio_path"] = "../../outside.wav"
        path.write_text(json.dumps(data), encoding="utf-8")
        with pytest.raises(BankError):
            reload(bank)

    def test_new_refuses_to_overwrite_a_bank(self, tmp_path) -> None:
        bank = new_bank(tmp_path)
        add_clip(bank)
        bank.save()
        with pytest.raises(BankError, match="exists"):
            CloneBank.new(bank.root, "S04", "spk_victim", allowed_victims=["S04"])
        CloneBank.new(bank.root, "S04", "spk_victim", allowed_victims=["S04"], overwrite=True)


class TestSelection:
    def test_unannotated_clips_are_never_measured(self, tmp_path) -> None:
        bank = new_bank(tmp_path)
        add_clip(bank, annotated=False)
        assert bank.measured() == [] and bank.match("hometown") is None

    def test_a_flagged_clip_is_never_measured_or_matched(self, tmp_path) -> None:
        """A transcript Whisper translated is a fabricated language choice."""
        bank = new_bank(tmp_path)
        add_clip(bank, flags=["looks_translated: all English"])
        assert bank.measured() == [] and bank.match("hometown") is None

    def test_match_picks_the_admissible_clip_with_the_best_similarity(self, tmp_path) -> None:
        bank = new_bank(tmp_path)
        add_clip(bank, similarity=0.70)
        best = add_clip(bank, similarity=0.85)
        add_clip(bank, similarity=0.95, admissible=False)
        assert bank.match("hometown").clip_id == best.clip_id

    def test_match_only_returns_a4_clips_for_that_fact(self, tmp_path) -> None:
        bank = new_bank(tmp_path)
        add_clip(bank, attack=AttackType.A3_CLONE, fact_key=None)
        add_clip(bank, fact_key="college")
        assert bank.match("hometown") is None
        assert bank.match("college") is not None

    def test_covered_facts_lists_only_what_can_be_answered(self, tmp_path) -> None:
        bank = new_bank(tmp_path)
        add_clip(bank, fact_key="hometown")
        add_clip(bank, fact_key="college", admissible=False)
        add_clip(bank, fact_key="favouriteFood", annotated=False)
        assert bank.covered_facts() == ["hometown"]


class TestYield:
    def test_yield_is_admissible_over_annotated(self, tmp_path) -> None:
        bank = new_bank(tmp_path)
        add_clip(bank, admissible=True)
        add_clip(bank, admissible=True)
        add_clip(bank, admissible=False)
        add_clip(bank, annotated=False)
        s = bank.yield_summary()
        assert (s.generated, s.annotated, s.admissible) == (4, 3, 2)
        assert s.yield_rate == pytest.approx(2 / 3)

    def test_yield_is_none_when_nothing_is_annotated(self, tmp_path) -> None:
        bank = new_bank(tmp_path)
        add_clip(bank, annotated=False)
        assert bank.yield_summary().yield_rate is None

    def test_sources_counts_distinct_attackers(self, tmp_path) -> None:
        bank = new_bank(tmp_path)
        add_clip(bank, speaker="attacker_1")
        add_clip(bank, speaker="attacker_1")
        add_clip(bank, speaker="attacker_2")
        assert bank.yield_summary().sources == 2


class TestResolveSpeaker:
    SPEAKERS = [
        {"id": "spk_a", "display_name": "S04 · Someone"},
        {"id": "spk_b", "display_name": "S08 · Other"},
        {"id": "spk_c", "display_name": "S040 · Not S04"},
    ]

    def test_it_matches_the_pseudonym_prefix_only(self) -> None:
        assert resolve_speaker_id(self.SPEAKERS, "S04") == "spk_a"

    def test_it_refuses_when_there_is_no_match(self) -> None:
        with pytest.raises(BankError):
            resolve_speaker_id(self.SPEAKERS, "S11")

    def test_it_refuses_when_there_are_two(self) -> None:
        with pytest.raises(BankError):
            resolve_speaker_id(self.SPEAKERS + [{"id": "spk_d", "display_name": "S04"}], "S04")
