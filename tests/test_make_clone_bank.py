from __future__ import annotations

import json

import pytest

from clone_helpers import tone
from kavach.attacks import AttackType
from kavach.attacks.bank import BANK_FILE, BankError, CloneBank
from kavach.attacks.clone import VoiceConverter
from kavach.attacks.make_clone_bank import generate_bank, main
from kavach.audio import Audio


class FakeConverter:
    """Stands in for kNN-VC: same words, 'new voice' = a different pitch tone."""

    def name(self) -> str:
        return "fake"

    def convert(self, source: Audio, target_reference: list[Audio]) -> Audio:
        return tone(seconds=source.duration_sec, freq=210.0)


def sources(*facts: str) -> dict[str, Audio]:
    return {f: tone(seconds=3.0, seed=i) for i, f in enumerate(facts)}


def generate(tmp_path, **kw):
    args = dict(
        victim="S04",
        victim_speaker_id="spk_victim",
        converter=FakeConverter(),
        target_reference=[tone(seconds=8.0)],
        sources=sources("hometown", "college"),
        facts=["hometown", "college", "favouriteFood"],
        out_root=tmp_path / "clones" / "S04",
        allowed_victims=["S04"],
    )
    args.update(kw)
    return generate_bank(**args)


def test_the_fake_satisfies_the_protocol() -> None:
    assert isinstance(FakeConverter(), VoiceConverter)


def test_one_a4_clip_per_covered_fact(tmp_path) -> None:
    report = generate(tmp_path)
    bank = CloneBank.load(report.bank.root, allowed_victims=["S04"])
    assert sorted(c.fact_key for c in bank.clips) == ["college", "hometown"]
    assert all(c.attack is AttackType.A4_CLONE_KNOWLEDGE for c in bank.clips)
    assert report.uncovered_facts == ["favouriteFood"]


def test_clips_are_marked_synthetic_in_the_file_name(tmp_path) -> None:
    report = generate(tmp_path)
    assert all("SYNTHETIC" in c.audio_path for c in report.bank.clips)


def test_nothing_in_the_bank_carries_a_name(tmp_path) -> None:
    report = generate(tmp_path)
    text = (report.bank.root / BANK_FILE).read_text(encoding="utf-8")
    assert json.loads(text)["victim"] == "S04"
    assert "·" not in text, "a display name leaked into the bank"


def test_a_victim_not_on_the_allowlist_is_refused_before_any_work(tmp_path) -> None:
    class Exploding(FakeConverter):
        def convert(self, *a, **k):
            raise AssertionError("converted audio for someone who has not agreed")

    with pytest.raises(BankError, match="allowlist"):
        generate(tmp_path, converter=Exploding(), allowed_victims=[])


def test_too_little_target_audio_is_refused(tmp_path) -> None:
    with pytest.raises(BankError, match="reference"):
        generate(tmp_path, target_reference=[tone(seconds=2.0)])


def test_short_but_usable_target_audio_warns(tmp_path) -> None:
    report = generate(tmp_path)  # 8 s: usable, far under the recommended amount
    assert any("minutes" in w for w in report.warnings)


def test_a_second_run_does_not_silently_replace_the_bank(tmp_path) -> None:
    generate(tmp_path)
    with pytest.raises(BankError, match="exists"):
        generate(tmp_path)
    generate(tmp_path, overwrite=True)


def test_the_cli_refuses_a_victim_who_is_not_allowed_without_touching_a_model(
    tmp_path, monkeypatch, capsys
) -> None:
    import kavach.attacks.make_clone_bank as mod

    monkeypatch.setattr(
        mod, "build_converter", lambda name: (_ for _ in ()).throw(AssertionError("loaded a model"))
    )
    monkeypatch.setenv("KAVACH_DATA_DIR", str(tmp_path))
    code = main(["--victim", "S04", "--sources", str(tmp_path), "--backend", "knn_vc"])
    assert code == 2
    assert "allowlist" in capsys.readouterr().err
