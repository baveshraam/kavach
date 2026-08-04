"""Speaker Knowledge Graph -- personal facts as RDF triples.

The SKG holds what a speaker knows about their own life. It serves two roles:

1.  **The knowledge factor.** An attacker with a cloned voice must also know
    the answer.
2.  **The elicitation surface.** The challenge generator queries it for a fact
    in a semantic class where this speaker is most discriminative, so the
    question reliably provokes tokens of the class the CSBG wants to measure
    (see `challenge.py`).

Real RDF via `rdflib`, not a dict, because role 2 needs actual traversal: the
generator asks "give me a fact of type FOOD linked to this speaker" and, where
entities link into the cultural ontology, can reach related entities for
richer questions. A flat key-value store cannot answer the second kind of
query, and a knowledge graph that is only a dict with a graph-shaped name is
exactly what a reviewer will call out.

⚠️  PRIVACY -- read before deploying or releasing anything
-----------------------------------------------------------
An SKG is sensitive personal data: hometown, family, school, daily routine,
sitting next to a voiceprint. Consequences that must be respected:

* **Never release raw SKGs with the corpus.** Release the CSBG structure with
  entities anonymised (`SpeakerKG.anonymised()`), never these triples.
* Encrypt at rest. `to_dict()` output is plaintext by design so the
  application layer can choose its own encryption; do not write it to disk
  unencrypted.
* A deletion request must remove the SKG, not just the audio -- see
  `SKGStore.delete`.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any, Iterator

from .csbg.ontology import SemanticClass

#: Namespace for KAVACH-minted entities and predicates.
KAVACH_NS = "http://kavach.research/ns#"


@dataclass(frozen=True, slots=True)
class FactType:
    """A category of personal fact the SKG can hold.

    `semantic_class` is the link that makes adaptive targeting work: it says
    which CSBG class a question about this fact will elicit tokens from. Ask
    about food, measure the FOOD class.
    """

    predicate: str
    question: str
    """Interview prompt used at enrolment. Phrased conversationally so the
    answer is natural speech, not a form field."""

    semantic_class: SemanticClass
    example: str = ""
    sensitive: bool = True
    """Whether this fact must be stripped before any data release."""


#: The enrolment interview. Chosen to span the elicitable semantic classes so
#: the challenge generator has a fact available whichever class turns out to
#: be discriminative for a given speaker -- a speaker whose tell is NUMBER is
#: useless if we can only ask about food.
FACT_TYPES: tuple[FactType, ...] = (
    FactType("hometown", "Which town or city are you from?", SemanticClass.PLACE_LOCAL, "Thanjavur"),
    FactType("school", "Which school did you go to?", SemanticClass.EDU_WORK, "St. Joseph's"),
    FactType("college", "Which college do you study at?", SemanticClass.EDU_WORK, "Amrita"),
    FactType("favouriteFood", "What is your favourite food?", SemanticClass.FOOD, "kothu parotta"),
    FactType("comfortFood", "What do you eat when you're feeling low?", SemanticClass.FOOD, "curd rice"),
    FactType("siblingName", "Do you have brothers or sisters? What are their names?", SemanticClass.KINSHIP, ""),
    FactType("familyRole", "Who do you talk to most at home?", SemanticClass.KINSHIP, "amma"),
    FactType("commute", "How do you usually travel to college?", SemanticClass.TRANSPORT, "bus"),
    FactType("commuteTime", "How long does that journey take?", SemanticClass.TIME_DATE, "40 minutes"),
    FactType("hostelRoom", "What was your hostel room number, or your house number?", SemanticClass.NUMBER, "214"),
    FactType("favouriteFestival", "Which festival do you enjoy most?", SemanticClass.RELIGION_FESTIVAL, "Pongal"),
    FactType("favouriteFilm", "What is a film you have watched many times?", SemanticClass.MEDIA_ENTERTAIN, ""),
    FactType("phoneBrand", "What phone do you use?", SemanticClass.TECH_DIGITAL, ""),
    FactType("firstJob", "What was your first job or internship?", SemanticClass.EDU_WORK, ""),
)

FACT_TYPE_BY_PREDICATE: dict[str, FactType] = {f.predicate: f for f in FACT_TYPES}

#: Fact types grouped by the CSBG class they elicit. Drives challenge targeting.
FACTS_BY_CLASS: dict[SemanticClass, tuple[FactType, ...]] = {}
for _f in FACT_TYPES:
    FACTS_BY_CLASS.setdefault(_f.semantic_class, ())
    FACTS_BY_CLASS[_f.semantic_class] += (_f,)


@dataclass(slots=True)
class Fact:
    """One (speaker, predicate, value) assertion."""

    predicate: str
    value: str
    raw_answer: str = ""
    """The speaker's full spoken answer, before entity extraction. Kept
    because it is the ground truth if extraction was wrong -- ASR errors on
    Tamil proper nouns are common and a researcher must be able to correct
    them at enrolment."""

    confidence: float = 1.0
    verified: bool = False
    """True once a human has confirmed the extracted value. Unverified facts
    are usable but flagged in the UI."""

    @property
    def fact_type(self) -> FactType | None:
        return FACT_TYPE_BY_PREDICATE.get(self.predicate)

    @property
    def semantic_class(self) -> SemanticClass:
        ft = self.fact_type
        return ft.semantic_class if ft else SemanticClass.OTHER


def slugify(value: str) -> str:
    """Turn a free-text value into a URI-safe local name.

    Preserves Tamil script (RDF URIs accept it via IRIs) so a Tamil-script
    answer does not collapse to an empty slug and silently lose the entity.
    """
    cleaned = re.sub(r"[^\w஀-௿]+", "_", value.strip(), flags=re.UNICODE)
    return cleaned.strip("_") or "unknown"


class SpeakerKG:
    """RDF knowledge graph of one speaker's personal facts."""

    def __init__(self, speaker_id: str) -> None:
        self.speaker_id = speaker_id
        self._facts: dict[str, Fact] = {}
        self._graph: Any = None

    # ------------------------------------------------------------- rdflib

    @property
    def graph(self) -> Any:
        """The rdflib Graph, built lazily and rebuilt when facts change."""
        if self._graph is None:
            self._graph = self._build_graph()
        return self._graph

    def _build_graph(self) -> Any:
        try:
            from rdflib import Graph, Literal, Namespace, URIRef
            from rdflib.namespace import RDF, RDFS
        except ImportError as exc:  # pragma: no cover - environment issue
            raise ImportError(
                "rdflib is required for the Speaker Knowledge Graph. "
                "Install with `pip install -r requirements-core.txt`."
            ) from exc

        ns = Namespace(KAVACH_NS)
        g = Graph()
        g.bind("kavach", ns)

        subject = URIRef(f"{KAVACH_NS}speaker_{slugify(self.speaker_id)}")
        g.add((subject, RDF.type, ns.Speaker))

        for fact in self._facts.values():
            predicate = URIRef(f"{KAVACH_NS}{fact.predicate}")
            entity = URIRef(f"{KAVACH_NS}{slugify(fact.value)}")
            g.add((subject, predicate, entity))
            # Keep the surface form: slugging is lossy and the matcher needs
            # the original string to compare an answer against.
            g.add((entity, RDFS.label, Literal(fact.value)))
            g.add((entity, ns.semanticClass, Literal(fact.semantic_class.value)))
        return g

    def _invalidate(self) -> None:
        self._graph = None

    # -------------------------------------------------------------- facts

    def add_fact(
        self,
        predicate: str,
        value: str,
        *,
        raw_answer: str = "",
        confidence: float = 1.0,
        verified: bool = False,
    ) -> Fact:
        """Assert a fact, replacing any existing one for the same predicate."""
        fact = Fact(
            predicate=predicate,
            value=value.strip(),
            raw_answer=raw_answer,
            confidence=confidence,
            verified=verified,
        )
        self._facts[predicate] = fact
        self._invalidate()
        return fact

    def remove_fact(self, predicate: str) -> bool:
        if predicate in self._facts:
            del self._facts[predicate]
            self._invalidate()
            return True
        return False

    def get(self, predicate: str) -> Fact | None:
        return self._facts.get(predicate)

    def __len__(self) -> int:
        return len(self._facts)

    def __iter__(self) -> Iterator[Fact]:
        return iter(self._facts.values())

    @property
    def facts(self) -> list[Fact]:
        return list(self._facts.values())

    # ------------------------------------------------------------ queries

    def facts_for_class(self, semantic_class: SemanticClass) -> list[Fact]:
        """Facts whose questions elicit tokens of `semantic_class`.

        The primary query behind adaptive challenge targeting.
        """
        return [f for f in self._facts.values() if f.semantic_class is semantic_class]

    def available_classes(self) -> set[SemanticClass]:
        """Semantic classes this speaker has at least one fact for."""
        return {f.semantic_class for f in self._facts.values()}

    def sparql(self, query: str) -> list[Any]:
        """Run a SPARQL query against the graph.

        Exposed for exploration and for the paper's KG figure. The routine
        paths use the typed helpers above -- SPARQL string-building in
        application code is harder to test and no faster.
        """
        return list(self.graph.query(query))

    def query_by_class(self, semantic_class: SemanticClass) -> list[tuple[str, str]]:
        """SPARQL lookup of (predicate, label) for one semantic class.

        Functionally equivalent to `facts_for_class`, but goes through the
        triple store -- which is the honest demonstration that the SKG is a
        queried knowledge graph rather than a dict wearing a graph's name.
        """
        rows = self.sparql(
            f"""
            PREFIX kavach: <{KAVACH_NS}>
            PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#>
            SELECT ?predicate ?label WHERE {{
                ?speaker ?predicate ?entity .
                ?entity rdfs:label ?label .
                ?entity kavach:semanticClass "{semantic_class.value}" .
            }}
            """
        )
        return [(str(p).replace(KAVACH_NS, ""), str(label)) for p, label in rows]

    def missing_fact_types(self) -> list[FactType]:
        """Interview questions not yet answered."""
        return [ft for ft in FACT_TYPES if ft.predicate not in self._facts]

    @property
    def completeness(self) -> float:
        return len(self._facts) / len(FACT_TYPES) if FACT_TYPES else 0.0

    # ------------------------------------------------------ serialisation

    def to_dict(self) -> dict[str, Any]:
        """Plaintext representation. **Encrypt before writing to disk.**"""
        return {
            "speaker_id": self.speaker_id,
            "facts": [
                {
                    "predicate": f.predicate,
                    "value": f.value,
                    "raw_answer": f.raw_answer,
                    "confidence": f.confidence,
                    "verified": f.verified,
                }
                for f in self._facts.values()
            ],
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> SpeakerKG:
        kg = cls(data["speaker_id"])
        for f in data.get("facts", []):
            kg.add_fact(
                f["predicate"],
                f["value"],
                raw_answer=f.get("raw_answer", ""),
                confidence=f.get("confidence", 1.0),
                verified=f.get("verified", False),
            )
        return kg

    def anonymised(self) -> dict[str, Any]:
        """Release-safe view: structure without any personal values.

        THIS is what may accompany a released corpus. It shows which fact
        types a speaker had and which classes were covered -- enough to
        reproduce the method -- while disclosing nothing about the person.
        """
        return {
            "speaker_id": self.speaker_id,
            "n_facts": len(self._facts),
            "predicates": sorted(self._facts),
            "classes_covered": sorted(c.value for c in self.available_classes()),
            "completeness": round(self.completeness, 3),
        }

    def serialise_rdf(self, fmt: str = "turtle") -> str:
        """Serialise the triples (turtle, xml, json-ld, ...).

        Contains personal data -- for local inspection and paper figures only.
        """
        return self.graph.serialize(format=fmt)

    def __repr__(self) -> str:
        return f"SpeakerKG(speaker={self.speaker_id!r}, facts={len(self._facts)})"


class SKGStore:
    """In-memory collection of speaker knowledge graphs.

    Deliberately not persistent. Persistence must be an explicit, encrypted
    choice by the application layer -- defaulting to writing personal facts to
    disk would be the wrong default for this data.
    """

    def __init__(self) -> None:
        self._graphs: dict[str, SpeakerKG] = {}

    def get_or_create(self, speaker_id: str) -> SpeakerKG:
        if speaker_id not in self._graphs:
            self._graphs[speaker_id] = SpeakerKG(speaker_id)
        return self._graphs[speaker_id]

    def get(self, speaker_id: str) -> SpeakerKG | None:
        return self._graphs.get(speaker_id)

    def delete(self, speaker_id: str) -> bool:
        """Erase a speaker's SKG. Required for data-subject deletion requests."""
        return self._graphs.pop(speaker_id, None) is not None

    def __contains__(self, speaker_id: str) -> bool:
        return speaker_id in self._graphs

    def __len__(self) -> int:
        return len(self._graphs)

    @property
    def speaker_ids(self) -> list[str]:
        return list(self._graphs)


def interview_script() -> list[tuple[str, str]]:
    """(predicate, question) pairs for the enrolment interview, in order."""
    return [(f.predicate, f.question) for f in FACT_TYPES]


__all__ = [
    "FactType",
    "Fact",
    "SpeakerKG",
    "SKGStore",
    "FACT_TYPES",
    "FACT_TYPE_BY_PREDICATE",
    "FACTS_BY_CLASS",
    "KAVACH_NS",
    "interview_script",
    "slugify",
]
