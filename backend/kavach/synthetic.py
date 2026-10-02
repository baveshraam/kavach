"""A synthetic stranger for the preflight: Windows' built-in text-to-speech.

The preflight cannot be the presenter, but it can be someone else saying exactly the words on
screen. A synthetic voice is nothing like a person, so this proves the plumbing (the words are
recognised, a wrong voice is rejected, other words are rejected) and not how the system treats a
human who resembles the presenter. Returns None where there is no such voice (not Windows, or no
SAPI voice installed): callers must treat that as "could not rehearse", never as a pass.
"""

from __future__ import annotations

import subprocess
import sys
import tempfile
from pathlib import Path

STRANGER_VOICE = "Microsoft Zira Desktop"


def speak(text: str, voice: str = STRANGER_VOICE, *, timeout: float = 60.0) -> bytes | None:
    """`text` spoken by `voice` as WAV bytes, or None when no synthetic voice is available."""
    if sys.platform != "win32":
        return None
    safe = text.replace("'", " ").replace('"', " ")
    with tempfile.TemporaryDirectory() as tmp:
        wav = Path(tmp) / "speak.wav"
        script = (
            "Add-Type -AssemblyName System.Speech; "
            "$s = New-Object System.Speech.Synthesis.SpeechSynthesizer; "
            f"$s.SelectVoice('{voice}'); $s.SetOutputToWaveFile('{wav}'); $s.Speak('{safe}'); $s.Dispose()"
        )
        try:
            subprocess.run(["powershell", "-NoProfile", "-Command", script], check=True, capture_output=True, timeout=timeout)
            return wav.read_bytes()
        except (OSError, subprocess.SubprocessError):
            return None
