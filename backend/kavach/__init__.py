"""KAVACH -- Knowledge-graph Anchored Voice Authentication for
Code-switched, Hybrid-language speakers.

A multi-factor voice authentication system that fuses:

    1. acoustic speaker identity      (ECAPA-TDNN embedding)
    2. code-switching idiolect        (Code-Switch Behaviour Graph)  <- the contribution
    3. knowledge possession           (Speaker Knowledge Graph challenge)
    4. response freshness             (challenge binding + timing)

Research target: SPELLL-2026, theme "Knowledge Graphs for low-resource
languages using LLM and Multimodal data".

Subpackage layout::

    csbg/       Code-Switch Behaviour Graph -- the research core. Pure numpy.
    corpus      Recorded-speech manifest, loader and elicitation protocol.
    lid/        Word-level language ID and semantic tagging.
    asr/        Speech recognition (faster-whisper).
    embedding/  ECAPA-TDNN speaker embeddings.
    skg/        Speaker Knowledge Graph (RDF triples).
    challenge/  Adaptive challenge generation.
    matcher/    Cross-lingual answer verification.
    fusion/     Score calibration and fusion.
    attacks/    Replay/splice/clone attack generation for evaluation.
    eval/       EER, DET curves, ablations, fairness slices.
    api/        FastAPI application.

Heavy dependencies (torch, speechbrain, faster-whisper) are imported lazily
inside the modules that need them, so `import kavach.csbg` stays fast and the
research core can be tested with `pip install -r requirements-core.txt`.
"""

__version__ = "0.1.0"

__all__ = ["__version__"]
