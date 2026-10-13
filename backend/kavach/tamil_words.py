"""The Tamil word bank for the experimental mixed-language challenge, and Unicode-safe comparison.

    python -m kavach.tamil_words check          # bank sanity: size, duplicates, confusable pairs

A mixed challenge is ten English words and two Tamil words read in order (`mixed_phrase`). This module owns
the Tamil half: ~100 common, concrete words (Tamil script, a romanisation to help the reader, an English
gloss), and the one rule for deciding that two Tamil strings are the same word.

The bank is text only and is committed. The enrolled speaker's RECORDINGS of these words are not: they live
in the git-ignored Studio store (`data/studio/<speaker>/`), and `data/tamil_words/words.jsonl` indexes them.
Those recordings are enrolment and evaluation data (does the recogniser hear this speaker say this word?);
no login ever compares a probe to them, so replaying one proves nothing.

Comparison, and why it is this strict. Two different Tamil words are often one or two combining marks apart
(a vowel sign, a pulli), so edit-distance leniency would equate words that mean different things. Tamil
strings are therefore compared only after Unicode NFC and removal of zero-width characters, and then by
EXACT equality; a near match is allowed only for long words and only when the bank has no other word that
close (`confusable_pairs` proves it for the whole bank and a test pins it).
"""

from __future__ import annotations

import json
import re
import sys
import unicodedata
from dataclasses import dataclass
from difflib import SequenceMatcher
from pathlib import Path

TAMIL_BLOCK = re.compile(r"[஀-௿]")
_INVISIBLE = dict.fromkeys(map(ord, "​‌‍⁠﻿"), None)
_NOT_TAMIL = re.compile(r"[^஀-௿]")
_NOT_LATIN = re.compile(r"[^a-z]")

#: A near Tamil match needs at least this many Tamil code points and this similarity. Below the length every
#: word must match exactly: a short word one mark away from another is a different word.
NEAR_MIN_LEN = 6
NEAR_RATIO = 0.9


@dataclass(frozen=True, slots=True)
class TamilWord:
    word_id: str
    ta: str
    roman: str
    gloss: str

    def to_dict(self) -> dict[str, str]:
        return {"word_id": self.word_id, "text": self.ta, "romanised": self.roman, "gloss": self.gloss, "language": "ta"}


# (Tamil, romanisation as a reader would write it, English gloss). Concrete everyday nouns of two or more
# syllables where possible: isolated one-syllable words are too short for the recogniser to confirm.
_RAW: tuple[tuple[str, str, str], ...] = (
    # people
    ("அம்மா", "amma", "mother"), ("அப்பா", "appa", "father"), ("அண்ணன்", "annan", "elder brother"),
    ("தம்பி", "thambi", "younger brother"), ("அக்கா", "akka", "elder sister"), ("தங்கை", "thangai", "younger sister"),
    ("தாத்தா", "thaatha", "grandfather"), ("பாட்டி", "paatti", "grandmother"), ("நண்பன்", "nanban", "friend"),
    ("ஆசிரியர்", "aasiriyar", "teacher"), ("குழந்தை", "kuzhandhai", "child"), ("மருத்துவர்", "maruthuvar", "doctor"),
    # food and drink
    ("சாதம்", "saadham", "rice"), ("தண்ணீர்", "thanneer", "water"), ("தயிர்", "thayir", "curd"),
    ("இட்லி", "idli", "idli"), ("தோசை", "dhosai", "dosa"), ("உப்பு", "uppu", "salt"),
    ("சர்க்கரை", "sarkkarai", "sugar"), ("காபி", "kaapi", "coffee"), ("தேநீர்", "thaeneer", "tea"),
    ("மாம்பழம்", "maambazham", "mango"), ("வாழைப்பழம்", "vaazhaippazham", "banana"),
    ("தேங்காய்", "thengaai", "coconut"), ("வெங்காயம்", "vengaayam", "onion"), ("தக்காளி", "thakkaali", "tomato"),
    ("மிளகாய்", "milagaai", "chilli"), ("எண்ணெய்", "ennai", "oil"), ("முட்டை", "muttai", "egg"),
    ("இறைச்சி", "iraichi", "meat"),
    # home
    ("வீடு", "veedu", "house"), ("கதவு", "kadhavu", "door"), ("ஜன்னல்", "jannal", "window"),
    ("நாற்காலி", "naarkaali", "chair"), ("மேசை", "maesai", "table"), ("புத்தகம்", "puthagam", "book"),
    ("பேனா", "paenaa", "pen"), ("கடிகாரம்", "kadigaaram", "clock"), ("விளக்கு", "vilakku", "lamp"),
    ("சாவி", "saavi", "key"), ("குடை", "kudai", "umbrella"), ("செருப்பு", "seruppu", "sandal"),
    ("துணி", "thuni", "cloth"), ("படுக்கை", "padukkai", "bed"), ("கண்ணாடி", "kannaadi", "mirror"),
    # nature
    ("மரம்", "maram", "tree"), ("நதி", "nadhi", "river"), ("கடல்", "kadal", "sea"), ("மலை", "malai", "mountain"),
    ("வானம்", "vaanam", "sky"), ("சூரியன்", "sooriyan", "sun"), ("நிலா", "nilaa", "moon"),
    ("நட்சத்திரம்", "natchathiram", "star"), ("மழை", "mazhai", "rain"), ("காற்று", "kaatru", "wind"),
    ("நெருப்பு", "neruppu", "fire"), ("காடு", "kaadu", "forest"), ("தீவு", "theevu", "island"),
    ("தோட்டம்", "thottam", "garden"),
    # animals
    ("பூனை", "poonai", "cat"), ("மாடு", "maadu", "cow"), ("குதிரை", "kudhirai", "horse"),
    ("யானை", "yaanai", "elephant"), ("புலி", "puli", "tiger"), ("சிங்கம்", "singam", "lion"),
    ("குரங்கு", "kurangu", "monkey"), ("மயில்", "mayil", "peacock"), ("பறவை", "paravai", "bird"),
    ("பாம்பு", "paambu", "snake"), ("ஆடு", "aadu", "goat"), ("கோழி", "kozhi", "hen"),
    ("காகம்", "kaagam", "crow"), ("எலி", "eli", "rat"),
    # places and transport
    ("கிராமம்", "giraamam", "village"), ("பள்ளி", "palli", "school"), ("கோயில்", "koyil", "temple"),
    ("கடை", "kadai", "shop"), ("சந்தை", "sandhai", "market"), ("நூலகம்", "noolagam", "library"),
    ("மருத்துவமனை", "maruthuvamanai", "hospital"), ("பேருந்து", "paerundhu", "bus"),
    ("ரயில்", "rayil", "train"), ("சாலை", "saalai", "road"), ("பாலம்", "paalam", "bridge"),
    ("நகரம்", "nagaram", "city"),
    # time
    ("காலை", "kaalai", "morning"), ("மாலை", "maalai", "evening"), ("இரவு", "iravu", "night"),
    ("வாரம்", "vaaram", "week"), ("மாதம்", "maadham", "month"), ("வருடம்", "varudam", "year"),
    ("நேரம்", "naeram", "time"),
    # colours
    ("சிவப்பு", "sivappu", "red"), ("நீலம்", "neelam", "blue"), ("பச்சை", "pachchai", "green"),
    ("மஞ்சள்", "manjal", "yellow"), ("கருப்பு", "karuppu", "black"), ("வெள்ளை", "vellai", "white"),
    # other
    ("பணம்", "panam", "money"), ("தங்கம்", "thangam", "gold"), ("பரிசு", "parisu", "gift"),
)

#: Glosses are plain English. A recogniser that translates a Tamil word into English could land on one of the
#: ten English challenge words by accident, so `mixed_phrase.draw_tamil` refuses a Tamil word whose gloss is
#: (or sounds like) any English word drawn for the same challenge.
BANK: tuple[TamilWord, ...] = tuple(TamilWord(f"ta{i:03d}", ta, rom, gl) for i, (ta, rom, gl) in enumerate(_RAW, 1))


def normalise_tamil(text: str) -> str:
    """NFC, zero-width characters removed, everything outside the Tamil block dropped (spaces, digits, Latin, punctuation)."""
    return _NOT_TAMIL.sub("", unicodedata.normalize("NFC", text).translate(_INVISIBLE))


def normalise_english(text: str) -> str:
    """Lower-case Latin letters only: 'Tiger,' and 'tiger' are one word; accents and punctuation are dropped."""
    folded = unicodedata.normalize("NFKD", text).casefold()
    return _NOT_LATIN.sub("", "".join(c for c in folded if not unicodedata.combining(c)))


def is_tamil_token(token: str) -> bool:
    return bool(TAMIL_BLOCK.search(token))


def tamil_same(expected: str, heard: str) -> bool:
    """True if `heard` is the Tamil word `expected`: identical after normalisation, or (long words only) a near string."""
    a, b = normalise_tamil(expected), normalise_tamil(heard)
    if not a or not b:
        return False
    if a == b:
        return True
    if min(len(a), len(b)) < NEAR_MIN_LEN:
        return False
    return SequenceMatcher(None, a, b).ratio() >= NEAR_RATIO


# ------------------------------------------------------------------ pronunciation keys
# Speech recognition writes a spoken Tamil word as Tamil script, as a romanisation, or as something in between, so
# exact spelling is the wrong test. Both the expected word and what was heard are reduced to a rough sound key
# (consonant skeleton, vowel length and doubling ignored, voiced = voiceless as in Tamil) and compared with a small
# edit-distance allowance. The allowance is 0 for keys shorter than 4, 1 for 4-6, 2 above, and `DEMO_WORDS` is chosen so
# that no two of its keys are within it (pinned by a test), so "close enough" cannot turn one demo word into another.

_CONS = {
    "க": "k", "ங": "n", "ச": "s", "ஞ": "n", "ட": "t", "ண": "n", "த": "t", "ந": "n", "ப": "p", "ம": "m", "ய": "y",
    "ர": "r", "ல": "l", "வ": "v", "ழ": "l", "ள": "l", "ற": "r", "ன": "n", "ஜ": "s", "ஷ": "s", "ஸ": "s", "ஹ": "h",
}
_VOWEL = {"அ": "a", "ஆ": "a", "இ": "i", "ஈ": "i", "உ": "u", "ஊ": "u", "எ": "e", "ஏ": "e", "ஐ": "ai", "ஒ": "o", "ஓ": "o", "ஔ": "au"}
_SIGN = {"ா": "a", "ி": "i", "ீ": "i", "ு": "u", "ூ": "u", "ெ": "e", "ே": "e", "ை": "ai", "ொ": "o", "ோ": "o", "ௌ": "au"}
_PULLI = "்"
_DIGRAPHS = (("zh", "l"), ("th", "t"), ("dh", "t"), ("sh", "s"), ("ch", "s"), ("kh", "k"), ("ph", "p"), ("bh", "p"), ("gh", "k"),
             ("ee", "i"), ("ii", "i"), ("oo", "u"), ("uu", "u"), ("aa", "a"), ("ay", "ai"), ("ey", "ai"))
_VOICING = str.maketrans({"g": "k", "d": "t", "b": "p", "j": "s", "c": "k", "q": "k", "w": "v", "z": "s", "f": "p", "x": "k"})


def _tamil_to_latin(text: str) -> str:
    s, out, i = normalise_tamil(text), [], 0
    while i < len(s):
        ch = s[i]
        nxt = s[i + 1] if i + 1 < len(s) else ""
        if ch in _CONS:
            out.append(_CONS[ch])
            if nxt in _SIGN:
                out.append(_SIGN[nxt])
                i += 1
            elif nxt == _PULLI:
                i += 1
            else:
                out.append("a")
        elif ch in _VOWEL:
            out.append(_VOWEL[ch])
        i += 1
    return "".join(out)


def phonetic_key(token: str) -> str:
    """A rough sound key for a Tamil-script or Latin token (see the block comment above)."""
    s = _tamil_to_latin(token) if is_tamil_token(token) else normalise_english(token)
    for a, b in _DIGRAPHS:
        s = s.replace(a, b)
    s = s.translate(_VOICING)
    return re.sub(r"(.)\1+", r"\1", s)


def _edit(a: str, b: str) -> int:
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        cur = [i]
        for j, cb in enumerate(b, 1):
            cur.append(min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (ca != cb)))
        prev = cur
    return prev[-1]


def allowance(key: str) -> int:
    return 0 if len(key) < 4 else 1 if len(key) <= 6 else 2


def phonetic_same(expected: str, heard: str) -> bool:
    """True if `heard` (any script) sounds like the Tamil word `expected` within the small allowance for its length."""
    a, b = phonetic_key(expected), phonetic_key(heard)
    return bool(a) and bool(b) and _edit(a, b) <= allowance(a)


def heard_as(expected: str, heard: str) -> bool:
    """The one test the mixed login applies to a Tamil word: exact Tamil (`tamil_same`) or the same sound (`phonetic_same`)."""
    return tamil_same(expected, heard) or phonetic_same(expected, heard)


#: The Tamil words the demo asks for: simple, everyday, two or three syllables, known pronunciations, and far apart in sound.
DEMO_WORDS: tuple[str, ...] = ("தண்ணீர்", "வீடு", "மலை", "பூனை", "கடல்", "யானை", "புத்தகம்", "அம்மா")


def demo_pool() -> list[TamilWord]:
    return [w for w in BANK if w.ta in DEMO_WORDS]


def confusable_pairs(bank: tuple[TamilWord, ...] = BANK) -> list[tuple[str, str]]:
    """Distinct bank words that `tamil_same` would call equal. Must be empty."""
    return [(x.ta, y.ta) for i, x in enumerate(bank) for y in bank[i + 1:] if tamil_same(x.ta, y.ta)]


def by_id() -> dict[str, TamilWord]:
    return {w.word_id: w for w in BANK}


def words_index_path(data_dir: Path | str) -> Path:
    return Path(data_dir) / "tamil_words" / "words.jsonl"


def load_recorded(data_dir: Path | str) -> list[dict]:
    """Rows of the local recorded-words dataset (`words.jsonl`); empty if none has been built yet."""
    path = words_index_path(data_dir)
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def main(argv: list[str] | None = None) -> int:
    ids = [w.word_id for w in BANK]
    texts = [normalise_tamil(w.ta) for w in BANK]
    pairs = confusable_pairs()
    short = [w.ta for w in BANK if len(normalise_tamil(w.ta)) <= 3]
    print(f"{len(BANK)} words; duplicate ids {len(ids) - len(set(ids))}; duplicate texts {len(texts) - len(set(texts))}; "
          f"confusable pairs {len(pairs)}; very short (<=3 code points) {len(short)}")
    for a, b in pairs:
        print(f"  confusable: {a} ~ {b}")
    return 1 if pairs or len(set(texts)) != len(texts) else 0


if __name__ == "__main__":
    sys.exit(main())
