"""The three pretrained speaker embedders compared in the paper, behind one constructor.

All are frozen, used as published, and scored by cosine. Each has its own cohort cache so a template from one model is
never scored against strangers embedded by another (`cohort_dir`): ECAPA keeps the original `data/cohort/emb`.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from ..audio import Audio, AudioError, prepare_for_embedding
from ..embedding import ECAPAEmbedder, SpeakerEmbedding

#: name -> (checkpoint, family). Registered order is the order tables are printed in.
EMBEDDERS = {
    "ecapa": ("speechbrain/spkrec-ecapa-voxceleb", "speechbrain"),
    "resnet": ("speechbrain/spkrec-resnet-voxceleb", "speechbrain"),
    "wavlm": ("microsoft/wavlm-base-plus-sv", "wavlm"),
}


class WavLMSVEmbedder:
    """Microsoft's WavLM-base-plus-SV x-vector head (512-d). Same preprocessing as the SpeechBrain embedders."""

    def __init__(self, *, model_name: str = EMBEDDERS["wavlm"][0], device: str = "cpu",
                 min_seconds: float = 1.0, max_seconds: float = 30.0) -> None:
        self.model_name = model_name
        self.device = device
        self.min_seconds = min_seconds
        self.max_seconds = max_seconds
        self._model: Any = None

    def _load(self):
        from transformers import AutoFeatureExtractor, WavLMForXVector

        fe = AutoFeatureExtractor.from_pretrained(self.model_name)
        model = WavLMForXVector.from_pretrained(self.model_name).to(self.device).eval()
        return fe, model

    @property
    def model(self) -> Any:
        if self._model is None:
            self._model = self._load()
        return self._model

    def embed(self, audio: Audio, *, preprocess: bool = True) -> SpeakerEmbedding:
        import torch

        prepared = prepare_for_embedding(audio, max_seconds=self.max_seconds) if preprocess else audio
        if prepared.duration_sec < self.min_seconds:
            raise AudioError(f"Audio is {prepared.duration_sec:.2f}s after trimming; at least {self.min_seconds}s is needed.")
        fe, model = self.model
        with torch.no_grad():
            inputs = fe(prepared.samples, sampling_rate=prepared.sample_rate, return_tensors="pt")
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            vec = model(**inputs).embeddings.squeeze().detach().cpu().numpy()
        return SpeakerEmbedding(vector=vec, duration_sec=prepared.duration_sec, source=prepared.source)


def make_embedder(name: str, *, device: str = "cpu"):
    if name not in EMBEDDERS:
        raise ValueError(f"unknown embedder {name!r}; known: {', '.join(EMBEDDERS)}")
    checkpoint, family = EMBEDDERS[name]
    if family == "wavlm":
        return WavLMSVEmbedder(model_name=checkpoint, device=device)
    return ECAPAEmbedder(model_name=checkpoint, device=device)


def cohort_dir(data_dir: Path | str, name: str) -> Path:
    """Where this embedder's cached cohort embeddings live. ECAPA keeps the original directory."""
    base = Path(data_dir) / "cohort"
    return base / "emb" if name == "ecapa" else base / f"emb_{name}"
