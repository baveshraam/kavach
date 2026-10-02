"""Download every checkpoint the live demo needs, once, ahead of time.

    python -m kavach.prefetch

The live API never downloads a model mid-login (see
`SemanticMatcher.allow_download`): a login that starts a 1.9 GB download is a
login that never finishes. So whatever it needs has to be on disk already --
this fetches it, on a good connection, before the demo.

Safe to re-run: every step is a cache hit once complete, and a step that fails
is reported and skipped rather than stopping the others.
"""

from __future__ import annotations

import sys
import time


def _step(name: str, fn) -> bool:
    t = time.perf_counter()
    print(f"- {name} ...", flush=True)
    try:
        fn()
    except Exception as exc:  # noqa: BLE001 -- report and continue
        print(f"  FAILED: {type(exc).__name__}: {exc}", flush=True)
        return False
    print(f"  ok ({time.perf_counter() - t:.0f}s)", flush=True)
    return True


def main() -> int:
    from .config import get_settings

    settings = get_settings()
    results = []

    def whisper() -> None:
        from faster_whisper import WhisperModel

        WhisperModel(settings.whisper_model, device="cpu", compute_type="int8")

    def ecapa() -> None:
        from .embedding import ECAPAEmbedder

        ECAPAEmbedder().model  # noqa: B018 -- property triggers the load

    def labse() -> None:
        from sentence_transformers import SentenceTransformer

        SentenceTransformer("sentence-transformers/LaBSE")

    results.append(_step(f"Whisper {settings.whisper_model}", whisper))
    results.append(_step("ECAPA-TDNN (speechbrain)", ecapa))
    results.append(_step("LaBSE answer matcher (~1.9 GB)", labse))

    print("all present" if all(results) else "some models missing -- the demo degrades without them")
    return 0 if all(results) else 1


if __name__ == "__main__":
    sys.exit(main())
