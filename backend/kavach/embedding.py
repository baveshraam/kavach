"""ECAPA-TDNN speaker embeddings -- the baseline biometric.

This is the branch KAVACH must *beat* in the attack evaluation: the paper's
central claim is that a voice clone defeats this and does not defeat the
CSBG. It therefore needs to be a fair, competently-configured baseline, not a
strawman. Deliberately weakening it would invalidate the result.

Uses the pretrained SpeechBrain checkpoint (`spkrec-ecapa-voxceleb`) as a
frozen feature extractor -- no training, standard practice, explicitly
supported by the model card. torch/speechbrain are imported lazily.

Reference EER on VoxCeleb1-O for this checkpoint is ~0.8-0.9%. **Do not
expect anything close on phone-recorded code-mixed Tamil-English.** That
benchmark is clean, English-centric, and matched-condition. The gap is a
finding to report (proposal 5.5, the fairness audit), not a bug.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from .audio import Audio, AudioError, prepare_for_embedding

#: Dimensionality of the SpeechBrain ECAPA-TDNN output.
EMBEDDING_DIM = 192


@dataclass(slots=True)
class SpeakerEmbedding:
    """An L2-normalised speaker embedding."""

    vector: np.ndarray
    duration_sec: float = 0.0
    source: str = ""

    def __post_init__(self) -> None:
        v = np.asarray(self.vector, dtype=np.float64).ravel()
        norm = np.linalg.norm(v)
        if norm < 1e-9:
            raise ValueError(
                "Degenerate (zero-norm) speaker embedding. The audio was probably "
                "silent -- check audio.check_quality before embedding."
            )
        # Normalise on construction so cosine similarity is a plain dot
        # product and no caller can forget to normalise.
        self.vector = v / norm

    def similarity(self, other: SpeakerEmbedding) -> float:
        """Cosine similarity in [-1, 1]. Higher means more likely same speaker."""
        return float(np.dot(self.vector, other.vector))

    def to_list(self) -> list[float]:
        return self.vector.tolist()

    @classmethod
    def from_list(cls, values: list[float], **kw: Any) -> SpeakerEmbedding:
        return cls(np.asarray(values, dtype=np.float64), **kw)


@dataclass(slots=True)
class SpeakerTemplate:
    """A speaker's enrolled acoustic identity: the centroid of their embeddings.

    Averaging several utterances is standard and materially more robust than
    a single embedding -- it averages out phonetic content and session noise,
    leaving speaker identity. Individual embeddings are retained so the
    template can be rebuilt if a bad recording is later removed.
    """

    speaker_id: str
    centroid: SpeakerEmbedding
    embeddings: list[SpeakerEmbedding] = field(default_factory=list)

    @classmethod
    def from_embeddings(
        cls, speaker_id: str, embeddings: list[SpeakerEmbedding]
    ) -> SpeakerTemplate:
        if not embeddings:
            raise ValueError(f"Cannot build a template for {speaker_id!r} with no embeddings.")
        mean = np.mean([e.vector for e in embeddings], axis=0)
        return cls(
            speaker_id=speaker_id,
            centroid=SpeakerEmbedding(mean, source=f"centroid[{len(embeddings)}]"),
            embeddings=list(embeddings),
        )

    def score(self, probe: SpeakerEmbedding) -> float:
        """Cosine similarity of a probe against the centroid."""
        return self.centroid.similarity(probe)

    @property
    def self_consistency(self) -> float:
        """Mean pairwise similarity among the enrolled embeddings.

        A diagnostic worth surfacing at enrolment: a low value means the
        recordings disagree about who this speaker is -- usually a mislabelled
        clip, a second person in the room, or wildly varying recording
        conditions. Catching that at enrolment is far cheaper than debugging
        it as unexplained EER later.
        """
        if len(self.embeddings) < 2:
            return 1.0
        sims = [
            self.embeddings[i].similarity(self.embeddings[j])
            for i in range(len(self.embeddings))
            for j in range(i + 1, len(self.embeddings))
        ]
        return float(np.mean(sims))

    def to_dict(self) -> dict[str, Any]:
        return {
            "speaker_id": self.speaker_id,
            "centroid": self.centroid.to_list(),
            "embeddings": [e.to_list() for e in self.embeddings],
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> SpeakerTemplate:
        return cls(
            speaker_id=data["speaker_id"],
            centroid=SpeakerEmbedding.from_list(data["centroid"]),
            embeddings=[SpeakerEmbedding.from_list(v) for v in data.get("embeddings", [])],
        )


class ECAPAEmbedder:
    """Wrapper around SpeechBrain's pretrained ECAPA-TDNN.

    The model is loaded on first use, not at construction, so importing this
    module stays cheap and the API server starts without waiting on a
    checkpoint download.
    """

    def __init__(
        self,
        *,
        model_name: str = "speechbrain/spkrec-ecapa-voxceleb",
        device: str = "cpu",
        cache_dir: str | Path | None = None,
        min_seconds: float = 1.0,
        max_seconds: float = 30.0,
    ) -> None:
        self.model_name = model_name
        self.device = device
        self.cache_dir = Path(cache_dir) if cache_dir else Path("pretrained_models")
        self.min_seconds = min_seconds
        self.max_seconds = max_seconds
        self._model: Any = None

    @property
    def model(self) -> Any:
        """Lazily load the checkpoint (downloads on first use)."""
        if self._model is None:
            try:
                from speechbrain.inference.speaker import EncoderClassifier
            except ImportError:
                try:
                    from speechbrain.pretrained import EncoderClassifier  # type: ignore
                except ImportError as exc:  # pragma: no cover - environment issue
                    raise ImportError(
                        "speechbrain is required for speaker embeddings. "
                        "Install with `pip install -r requirements.txt`."
                    ) from exc

            self._model = EncoderClassifier.from_hparams(
                source=self.model_name,
                savedir=str(self.cache_dir / self.model_name.replace("/", "_")),
                run_opts={"device": self.device},
            )
        return self._model

    def embed(self, audio: Audio, *, preprocess: bool = True) -> SpeakerEmbedding:
        """Extract an embedding from one recording.

        Args:
            audio: 16 kHz mono audio.
            preprocess: Apply trim/normalise/truncate. Keep True unless the
                caller has already run `prepare_for_embedding` -- enrolment
                and verification must preprocess identically.

        Raises:
            AudioError: If the clip is too short to embed reliably.
        """
        import torch

        prepared = (
            prepare_for_embedding(audio, max_seconds=self.max_seconds)
            if preprocess
            else audio
        )

        if prepared.duration_sec < self.min_seconds:
            raise AudioError(
                f"Audio is {prepared.duration_sec:.2f}s after trimming; at least "
                f"{self.min_seconds}s is needed for a stable embedding. "
                "Ask the speaker to re-record with a longer utterance."
            )

        with torch.no_grad():
            wav = torch.from_numpy(prepared.samples).unsqueeze(0)
            out = self.model.encode_batch(wav)
            vec = out.squeeze().detach().cpu().numpy()

        return SpeakerEmbedding(
            vector=vec, duration_sec=prepared.duration_sec, source=prepared.source
        )

    def embed_many(self, clips: list[Audio], **kw: Any) -> list[SpeakerEmbedding]:
        """Embed several clips, skipping those too short rather than failing.

        Enrolment routinely includes one clip where the speaker stopped early;
        losing that clip is better than losing the whole enrolment session.
        """
        out: list[SpeakerEmbedding] = []
        for clip in clips:
            try:
                out.append(self.embed(clip, **kw))
            except AudioError:
                continue
        return out

    def enrol(self, speaker_id: str, clips: list[Audio], **kw: Any) -> SpeakerTemplate:
        """Build a speaker template from enrolment recordings.

        Raises:
            AudioError: If no clip produced a usable embedding.
        """
        embeddings = self.embed_many(clips, **kw)
        if not embeddings:
            raise AudioError(
                f"No usable embeddings for {speaker_id!r} from {len(clips)} clips. "
                "All were too short or silent -- check audio.check_quality output."
            )
        return SpeakerTemplate.from_embeddings(speaker_id, embeddings)


def verify(
    template: SpeakerTemplate, probe: SpeakerEmbedding, *, threshold: float
) -> tuple[bool, float]:
    """Threshold a probe against a template.

    Returns:
        (accepted, cosine_similarity).
    """
    score = template.score(probe)
    return score >= threshold, score


def score_matrix(
    templates: dict[str, SpeakerTemplate], probes: dict[str, SpeakerEmbedding]
) -> tuple[list[float], list[float]]:
    """All-pairs scoring for evaluation.

    Returns:
        (genuine_scores, impostor_scores), split by whether the probe's
        speaker id matches the template's.
    """
    genuine: list[float] = []
    impostor: list[float] = []
    for spk_id, template in templates.items():
        for probe_id, probe in probes.items():
            score = template.score(probe)
            (genuine if probe_id == spk_id else impostor).append(score)
    return genuine, impostor


__all__ = [
    "SpeakerEmbedding",
    "SpeakerTemplate",
    "ECAPAEmbedder",
    "EMBEDDING_DIM",
    "verify",
    "score_matrix",
]
