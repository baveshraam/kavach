"""kNN-VC voice conversion (bshall/knn-vc): WavLM features + a HiFi-GAN vocoder.

Converts *speech* into the target speaker's voice by replacing each source frame
with the average of its nearest frames of the target's speech. The words, the
timing and the code-switching style are the source speaker's; only the voice
changes -- which is exactly attack A4.

Runs in the clone environment (`.venv-clone`, CUDA torch). `torch` is imported
inside the methods so this module imports anywhere; the first `convert` call
downloads the WavLM and vocoder checkpoints through torch hub.
"""

from __future__ import annotations

import tempfile
from pathlib import Path

import numpy as np

from ...audio import Audio, save_wav

OUTPUT_SAMPLE_RATE = 16_000


class KnnVcConverter:
    def __init__(self, *, device: str = "cuda", topk: int = 4, prematched: bool = True) -> None:
        self._device = device
        self._topk = topk
        self._prematched = prematched
        self._model = None

    def name(self) -> str:
        return "knn_vc"

    @property
    def model(self):
        if self._model is None:
            import torch

            self._model = torch.hub.load(
                "bshall/knn-vc",
                "knn_vc",
                prematched=self._prematched,
                trust_repo=True,
                pretrained=True,
                device=self._device,
            )
        return self._model

    def convert(self, source: Audio, target_reference: list[Audio]) -> Audio:
        model = self.model
        with tempfile.TemporaryDirectory() as d:
            src_path = save_wav(source, Path(d) / "source.wav")
            ref_paths = [
                str(save_wav(a, Path(d) / f"ref_{i}.wav")) for i, a in enumerate(target_reference)
            ]
            query = model.get_features(str(src_path))
            matching_set = model.get_matching_set(ref_paths)
            wav = model.match(query, matching_set, topk=self._topk)
        samples = wav.detach().cpu().numpy().astype(np.float32).ravel()
        return Audio(samples, OUTPUT_SAMPLE_RATE, "knn_vc")
