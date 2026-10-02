"""The impostor cohorts behind the false-accept numbers: public corpora, embedded once, cached.

The numbers in the calibration report are only as good as how these were built, so the build is code
with tests, not a scratch script: which clips were taken, how they were cut, and that a re-run is the same.
"""

from __future__ import annotations

import numpy as np
import pytest
import soundfile as sf

from kavach.cohort import Item, embed_cohort, enumerate_libri, enumerate_slr65, load_cohort, save_cohort
from kavach.embedding import SpeakerEmbedding


def write(path, seconds, sr=16000, seed=0):
    path.parent.mkdir(parents=True, exist_ok=True)
    t = np.linspace(0, seconds, int(sr * seconds), endpoint=False)
    x = 0.3 * np.sin(2 * np.pi * (150 + seed) * t) * (0.6 + 0.4 * np.sin(2 * np.pi * 3 * t)) + 0.01 * np.random.default_rng(seed).standard_normal(len(t))
    sf.write(str(path), x.astype("float32"), sr)


@pytest.fixture()
def libri(tmp_path):
    base = tmp_path / "LibriSpeech" / "dev-clean"
    for spk in ("100", "200"):
        for chap in ("1", "2"):
            for k in range(6):
                write(base / spk / chap / f"{spk}-{chap}-{k:04d}.flac", seconds=3.0 + 4 * (k % 3), seed=k)
    return tmp_path


class FakeEmbedder:
    def __init__(self):
        self.seen = []

    def embed(self, audio, **kw):
        self.seen.append(audio.duration_sec)
        v = np.zeros(192)
        v[len(self.seen) % 192] = 1.0
        return SpeakerEmbedding(v)


def test_libri_speakers_are_prefixed_and_chapters_are_the_sessions(libri):
    items = enumerate_libri(libri, "dev-clean", per_speaker=100)
    assert {i.speaker for i in items} == {"L100", "L200"}
    assert {i.session for i in items if i.speaker == "L100"} == {"100-1", "100-2"}


def test_clips_outside_the_duration_window_are_left_out(libri):
    items = enumerate_libri(libri, "dev-clean", per_speaker=100, min_seconds=4.0, max_seconds=12.0)
    import soundfile as sf2
    assert items and all(4.0 <= sf2.info(i.path).duration <= 12.0 for i in items)


def test_the_per_speaker_cap_applies_and_the_choice_is_reproducible(libri):
    a = enumerate_libri(libri, "dev-clean", per_speaker=3, seed=1)
    b = enumerate_libri(libri, "dev-clean", per_speaker=3, seed=1)
    c = enumerate_libri(libri, "dev-clean", per_speaker=3, seed=2)
    assert [i.path for i in a] == [i.path for i in b]
    assert len([i for i in a if i.speaker == "L100"]) == 3
    assert [i.path for i in a] != [i.path for i in c]


def test_slr65_speakers_come_from_the_file_name(tmp_path):
    for spk in ("00008", "00023"):
        for k in range(3):
            write(tmp_path / "male" / f"tam_{spk}_{k:011d}.wav", seconds=4.0, seed=k)
    items = enumerate_slr65(tmp_path, "male", per_speaker=10)
    assert {i.speaker for i in items} == {"Ttam_00008", "Ttam_00023"}
    assert {i.session for i in items} == {"one"}


def test_whole_clips_are_embedded_once_each(libri):
    items = enumerate_libri(libri, "dev-clean", per_speaker=100)
    emb = FakeEmbedder()
    c = embed_cohort(items, emb)
    assert len(c.vec) == len(items) == len(emb.seen)


def test_chunking_cuts_each_clip_into_probe_length_pieces_and_drops_a_short_tail(libri):
    items = [Item("L100", "100-1", str(next((libri / "LibriSpeech" / "dev-clean" / "100" / "1").glob("*.flac"))))]
    write(__import__("pathlib").Path(items[0].path), seconds=12.0)       # 12 s: two 5 s chunks, a 2 s tail dropped
    emb = FakeEmbedder()
    c = embed_cohort(items, emb, chunk_seconds=5.0)
    assert len(c.vec) == 2
    assert all(abs(d - 5.0) < 0.2 for d in c.dur)


def test_the_cache_round_trips(libri, tmp_path):
    c = embed_cohort(enumerate_libri(libri, "dev-clean", per_speaker=2), FakeEmbedder())
    out = save_cohort(c, tmp_path / "cache" / "libri-dev__ecapa.npz")
    back = load_cohort(out)
    assert back.vec.shape == c.vec.shape and list(back.speaker) == list(c.speaker)


def test_an_existing_cache_is_not_overwritten_silently(libri, tmp_path):
    c = embed_cohort(enumerate_libri(libri, "dev-clean", per_speaker=2), FakeEmbedder())
    p = save_cohort(c, tmp_path / "x.npz")
    with pytest.raises(FileExistsError):
        save_cohort(c, p)


def test_an_empty_corpus_is_an_error_not_an_empty_cache(tmp_path):
    with pytest.raises(ValueError, match="no clips"):
        embed_cohort([], FakeEmbedder())
