# KAVACH
### Knowledge-graph Anchored Voice Authentication for Code-switched, Hybrid-language speakers

> **Status, 2026-10-03.** This is the research proposal as written on 2026-08-04 and is not rewritten. Since then: a 12-speaker pilot corpus was recorded and annotated; the CSBG scored 26.3% EER on scripted speech (a script classifier) and 50.0% on free speech (chance), so the fallback framing of §12 applies; and the project's direction became evidence for a single enrolled speaker behind a gated login. What was built, decided and measured since is in `PROJECT.md` (sections 3.9-3.12, 4 and 6), `DEMO_HARDENING.md` and the top of `HANDOFF.md`. The deadline discussion below is as of August and has been overtaken by events.


**Target venue:** SPELLL-2026 — 5th International Conference on Speech and Language Technologies for Low-Resource Languages, Amrita Vishwa Vidyapeetham, Coimbatore, 17–19 December 2026.
**Conference theme:** *Knowledge Graphs for low-resource languages using LLM and Multimodal data.*

---

## ⚠️ Read this first: the deadline

Today is **2026-08-04**. The conference is **2026-12-17**. For a December conference with (likely) Springer proceedings, the paper submission deadline is almost certainly **late August to late September 2026**, with camera-ready in October.

**Before you write a single line of code, go to the SPELLL-2026 site and confirm:**
- Paper submission deadline
- Page limit and template (Springer LNCS/CCIS vs. ACL style)
- Whether there is a short-paper / work-in-progress track (a 4–6 page short paper is a very realistic target for this timeline)
- Whether dataset/resource papers are accepted as a category (this matters — see §5.4)

The plan in §11 assumes a **~8-week runway to a late-September deadline**. If the deadline is sooner, jump straight to the "Minimum Publishable Unit" in §12.

---

## 1. The one-paragraph pitch

Voice biometrics authenticates *how you sound*. Voice cloning has made how-you-sound cheap to forge. **KAVACH adds a second, orthogonal biometric that cloning does not capture: how you code-switch.** Bilingual Tamil–English speakers switch languages in stable, idiosyncratic, semantically-conditioned ways — one speaker says numbers in English but kinship terms in Tamil; another does the reverse; a third switches on discourse markers only. We encode each speaker's switching behaviour as a per-speaker **Code-Switch Behaviour Graph (CSBG)** — a knowledge graph over semantic concept classes and language-choice edges — and use it as a soft biometric. To *elicit* natural code-switching at authentication time (rather than hoping for it), an LLM traverses a small personal **Speaker Knowledge Graph (SKG)** to generate an unbounded, unpredictable, culturally-grounded challenge question in code-mixed Tamil–English. The result is a three-factor voice authentication system — *timbre* + *knowledge* + *switching idiolect* — that we show rejects state-of-the-art voice clones which defeat ECAPA-TDNN alone.

**The headline claim we are trying to earn:** *a cloned voice can reproduce a speaker's spectrum but not their sociolinguistic idiolect, and in low-resource code-switched settings this gap is large enough to be a usable security signal.*

---

## 2. Why the old proposal fails and what survives

| Old component | Verdict | Fate in KAVACH |
|---|---|---|
| ECAPA-TDNN speaker verification (pretrained) | Not novel. Baseline. | **Kept** — becomes the baseline system we must beat, and one branch of the fusion. |
| Random digit-string challenge-response | Not novel. Industry standard since IVR banking. Also: language-neutral, so it tests nothing linguistic. | **Replaced** by KG-grounded LLM challenge generation (§4.3). |
| Exact string match on ASR output | Breaks completely on Tamil–English code-mixed ASR (high WER). | **Replaced** by cross-lingual semantic + phonetic matching (§4.4). This becomes a *contribution*, because we can measure how badly exact-match fails. |
| "We tested on Tamil code-mixed speech" | Observation, not contribution. | **Elevated** — becomes a released annotated corpus + a fairness audit (§5.4, §5.5). |
| "Not a defence against deepfakes" (stated limitation) | This was the honest scope boundary. | **Inverted into the central research question.** The CSBG is precisely the deepfake defence the old proposal said it couldn't build — and it needs no spoof-labelled training corpus. |

The old proposal's fatal move was scoping *out* the interesting problem. KAVACH scopes it back in through a side door that doesn't require ASVspoof-scale data or GPU training.

---

## 3. Novelty — verified against the literature (searched 2026-08-04)

> **This section was revised after an actual literature search.** Findings below are from five targeted searches. Full source list in §13. **Read §3.4 before committing.**

### 3.1 Tier 1 — SURVIVES. This is the paper's spine.

> **Code-switching behaviour as a soft biometric for authentication and anti-spoofing.**

Five searches across speaker verification, anti-spoofing, behavioural biometrics, and code-switching literature returned **no work using code-switching patterns as an authentication or spoof-detection signal.** The code-switching field is almost entirely ASR and language identification; the speaker-verification field is almost entirely acoustic. The gap is real.

**However — an important nuance that changes how you must frame this.** The underlying premise is *already established* in the code-switching literature: it is a known and cited finding that **"code-switching is a speaker-dependent behaviour, where the frequency by which the foreign language is embedded differs across speakers."** Prior work has clustered speakers by code-switching attitude for language modelling (Zhang et al., *Code-Switching Attitude Dependent Language Modeling*).

This is **good news for feasibility** — the effect you depend on is documented, so it probably works. But it means you **cannot claim to have discovered that code-switching varies by speaker.** That is known. Your claim must be narrower and sharper:

> *We are the first to quantify whether speaker-dependent code-switching carries **enough** discriminative information to function as a security factor, and the first to show it survives voice-cloning attacks that defeat acoustic speaker verification.*

The contribution is the **measurement and the security application**, not the observation. Frame it that way in the abstract or a reviewer will write "this is an obvious application of a known phenomenon."

### 3.2 Tier 2 — **CUT THIS CLAIM. Prior art found.**

> ~~KG-grounded dynamic challenge generation~~

**US Patent 10,362,016 — "Dynamic knowledge-based authentication"** does essentially this: it builds a knowledge graph of a user's real-life events and generates authentication challenges from selected nodes in that graph. Also relevant: **US Patent 7,289,957**, verifying a speaker using random combinations of previously-supplied units.

**Do not present KG-based challenge generation as novel.** Cite the patent, describe the module as an engineering component built on known technique, and move on. Presenting it as a contribution and having a reviewer find this patent would damage your credibility on the Tier-1 claim, which is the one that matters.

**What remains defensible** is one narrow slice: the challenge is **adaptively targeted at the semantic class where this speaker is most discriminative**, closing a loop between the biometric and the probe. The patent generates challenges to test *knowledge*; you generate them to *elicit a measurable biometric signal*. That is a real difference — but it is one sentence in the paper, not a contribution bullet.

### 3.3 Tier 3 — SURVIVES, and is now more important than before

Code-switch-tolerant matching under high-WER Tamil–English ASR; the fairness audit; the attack suite. These are safe.

**One caveat on the corpus contribution:** Microsoft/SpeechOcean released ~60 hours of code-switched speech covering Tamil-English, Telugu-English and Gujarati-English (Interspeech 2018 / MUCS shared task lineage). **Verify whether it has per-speaker labels with multiple utterances per speaker** — you need that structure, and it is not confirmed. If it does, use it and drop the "first corpus" framing (your contribution becomes the *speaker-and-switch-annotated authentication* layer). If it doesn't, your corpus contribution stands and is stronger.

### 3.4 Closest structural prior art — you MUST cite and differentiate

**"Modeling the Bilingual Phonological Idiolect Through a Translanguaging Lens"** builds a **graph-based idiolect model of a bilingual speaker using network science.** Graph + bilingual + idiolect — structurally the nearest thing to the CSBG that exists.

It differs from KAVACH in three ways, and you should state them explicitly in Related Work:
1. It models **phonological** variation (speech-sound substitution patterns); the CSBG models **lexical-semantic language choice** (which language for which concept class).
2. Its purpose is descriptive/clinical (language acquisition); yours is **discriminative** (verification under attack).
3. It does not evaluate identification, spoofing, or security at all.

Being blindsided by this in review is avoidable. Cite it in your third paragraph.

### 3.5 What we will NOT claim
Not a new speaker-embedding architecture. Not beating ASVspoof SOTA. Not that CSBG works as a standalone primary biometric — it is a *soft* biometric and a fusion component. Not that KG challenge generation is new (§3.2). Reviewers punish overclaiming far harder than modest scope.

---

## 4. System architecture

Five modules. Modules A and E are the paper; B, C, D are the machinery that makes them possible.

```
                   ┌──────────────── ENROLMENT ────────────────┐
  audio ──► [B] ASR + word-level LID ──► [A] CSBG builder ──► G_s (speaker graph)
                                    └──► [C] SKG builder  ──► K_s (personal KG, RDF triples)

                   ┌──────────────── AUTHENTICATION ───────────┐
  K_s ──► [D] LLM challenge generator ──► "உங்க college-la first year-la எந்த hostel-la இருந்தீங்க?"
                                                       │
  response audio ──┬──► ECAPA-TDNN ──────────────► S_spk    (timbre)
                   ├──► [B] ASR+LID ──► [A] CSBG ─► S_csbg  (switching idiolect)  ★
                   ├──► [B] ASR ──► [E] X-ling matcher ─► S_know (knowledge)      ★
                   └──► timing/session ──────────► S_live   (freshness)
                                                       │
                                              [F] score fusion ──► ACCEPT / REJECT
```

### 4.1 Module A — The Code-Switch Behaviour Graph (technical core)

For speaker `s`, define graph `G_s = (V, E, w)`.

**Nodes:**
- `V_C` — a fixed ontology of ~20 semantic concept classes:
  `NUMBER, TIME_DATE, KINSHIP, FOOD, PLACE_LOCAL, PLACE_GLOBAL, TECH_DIGITAL, EDU_WORK, MONEY_COMMERCE, EMOTION_STATE, BODY_HEALTH, TRANSPORT, RELIGION_FESTIVAL, MEDIA_ENTERTAIN, DISCOURSE_MARKER, POLITENESS, QUANTITY_MEASURE, ACTION_VERB, FUNCTION_WORD, NAMED_ENTITY, OTHER`
- `V_L` — language nodes: `{TA, EN, NEUTRAL}`

**Edge type 1 — lexical-choice edges** `c → l`, weight `P_s(lang = l | class = c)`.
> *Captures: "this speaker uses English for NUMBER 94% of the time but Tamil for KINSHIP 88% of the time."*

**Edge type 2 — switch-transition edges** `(c_i, l_i) → (c_j, l_j)`, weight = observed bigram transition probability over consecutive tokens.
> *Captures where in the semantic flow the speaker actually flips language — the switch points, not just the totals.*

**Graph-level attributes** (established, citable code-mixing metrics — verify exact citations):
- **CMI** (Code-Mixing Index, Gambäck & Das 2014): for an utterance of `n` tokens with `u` language-neutral tokens and `max(w_i)` tokens in the dominant language:
  `CMI = 100 × (1 − max(w_i) / (n − u))` for `n > u`, else `0`
- **I-index** (integration/switch-point fraction, Guzmán et al. 2017): `#switch_points / (n − 1)`
- **Matrix-language ratio**: fraction of utterances where Tamil supplies the morphosyntactic frame (Myers-Scotton MLF)

**Scoring.** Given a login utterance `x`, build mini-graph `G_x`, then:

```
S_csbg(x, s) = 1 − [ α·JSD_lex(G_x, G_s) + β·JSD_trans(G_x, G_s) + γ·Δ_metrics(G_x, G_s) ]

  JSD_lex   = frequency-weighted mean Jensen-Shannon divergence over lexical-choice
              distributions, taken only over classes actually observed in x
  JSD_trans = JS divergence over switch-transition distributions (Laplace-smoothed)
  Δ_metrics = normalised |CMI_x − CMI_s| and |I_x − I_s|
```

Then **cohort-normalise (z-norm)** against the CSBGs of all other enrolled speakers so the score is discriminative rather than absolute. `α, β, γ` tuned on a dev split.

> **Why a graph and not a flat feature vector?** Three reasons, and you must be able to defend this to a reviewer: (1) the switch-transition structure is inherently relational — it is a labelled transition system, not a bag of numbers; (2) the graph is *inspectable and explainable* — you can show a user or an auditor exactly which semantic class caused a rejection, which a 512-d embedding cannot do; (3) it composes with the SKG in the same store, letting the challenge generator query "which concept class discriminates this speaker best, and do I have a personal entity in that class to ask about?" — the graph is *actively queried*, not decorative. If you can't defend (3), the KG framing looks bolted-on for the theme, which reviewers will notice.

**Sparsity is the main technical risk.** With ~4–5 minutes of speech per speaker, edge-type-2 counts are thin. Mitigations: back off to edge type 1 when transition counts are below threshold; Laplace smoothing; hierarchical class collapse (20 classes → 6 super-classes). Turn this into a result — see §5.3.

### 4.2 Module B — ASR + word-level language ID

- **ASR:** `faster-whisper` large-v3 (int8, CPU-viable) as baseline. Also evaluate AI4Bharat IndicWhisper / IndicConformer for Tamil — *verify current availability and licence*.
- **Word-level LID** is the accuracy bottleneck for the whole CSBG. Three-stage:
  1. **Script heuristic** — Tamil script → TA, Latin script → EN. Free and highly accurate when Whisper preserves script.
  2. **Romanised-Tamil detection** — Whisper often romanises Tamil. Handle with a transliteration round-trip (`indic-transliteration`) + Tamil lexicon lookup.
  3. **LLM adjudication** — send the token sequence to an LLM for per-token `{TA, EN, NEUTRAL, NAMED_ENTITY}` tagging *and* semantic-class assignment in one call. This is where the "LLM" in the conference theme does real work, not decoration.
- **You must validate the LID.** Hand-annotate ~500 tokens and report LID accuracy. Every downstream number depends on it, and a reviewer will ask.

### 4.3 Module C+D — Speaker KG and LLM challenge generation

**SKG construction (enrolment):** a short guided interview (10–12 questions: hometown, school, favourite food, family, college, commute, festivals). Store as RDF triples via `rdflib`:
```
:speaker_07  :hometown        :Thanjavur .
:speaker_07  :favouriteFood   :Kothu_Parotta .
:Kothu_Parotta a :Food ; :region :TamilNadu .
:speaker_07  :college         :Amrita_Coimbatore .
```
Link entities to a lightweight Tamil cultural ontology (foods, festivals, districts, kinship terms) so the LLM can generate *contextually rich* questions and so the graph has real structure rather than being a flat key-value store.

**Challenge generation (login):**
1. SPARQL-query the SKG for a fact the speaker knows.
2. Query the CSBG for the semantic class where this speaker is **most discriminative** (highest divergence from the cohort mean) *and* has the sparsest data.
3. Prompt the LLM: generate a natural code-mixed Tamil–English question whose answer will require tokens from that target class.
4. Never reuse a challenge; log all issued challenges per speaker.

> This is the elegant bit: **the challenge is adaptively targeted at the semantic classes that best identify this particular speaker.** The system asks about food when food is your tell, and about numbers when numbers are your tell. Frame this as *active biometric probing* — it's a nice, quotable idea.

### 4.4 Module E — Code-switch-tolerant answer verification

Exact string match dies on code-mixed ASR. Replace with a three-way match against the SKG-expected answer:
1. **Cross-lingual semantic similarity** — LaBSE or an Indic SBERT variant (*verify model availability*), cosine over sentence embeddings. Handles "Thanjavur" vs "தஞ்சாவூர்".
2. **Phonetic similarity** — transliterate both to a common phonetic representation (ISO-15919 via `indic-transliteration`), then normalised Levenshtein. Handles ASR spelling errors and script inconsistency.
3. **Entity linking** — check whether the SKG's expected entity (or an alias) appears at all.

`S_know = max(w_sem·sim_sem, w_pho·sim_pho, w_ent·hit_ent)`

**Ablation this.** Exact-match vs. semantic-only vs. phonetic-only vs. full. The false-rejection rate of exact match on Tamil–English code-mixed answers is, on its own, a publishable observation for a low-resource venue.

### 4.5 Module F — Fusion

`S_final = w1·S_spk + w2·S_csbg + w3·S_know`, weights from logistic regression on a dev split. Liveness is a **gate**, not a weighted term. Report EER for each branch alone and every fusion combination.

#### 4.5.1 A weighted average cannot express the paper's claim — found while building

This surfaced when the A4 experiment was first wired up, and it is worth a paragraph in the paper because it is not obvious.

With weights (0.40, 0.30, 0.30) and a decision threshold of 0.55, an A4 attacker who fools the acoustic branch (0.85) and knows the answer (0.90) already sits at `0.4(0.85) + 0.3(0.90) = 0.61` from those two branches alone. The CSBG carries weight 0.30, so it can move the total by at most 0.30 — and it moves it *downward* only as far as its own score allows. **No CSBG score, not even exactly zero, pulls that trial below threshold.** Under plain linear fusion the CSBG is structurally incapable of stopping the one attack it exists to stop.

Raising the CSBG's weight until the arithmetic works is tuning to a single adversary, and it charges the cost to genuine speakers who happen to answer tersely.

The real error is a category one. The CSBG score is a **log-likelihood ratio against a background model**, so a strongly negative value is not weak support for identity — it is affirmative evidence *against* it. Averaging that into a pool of positive evidence discards its sign. This is the same mistake `liveness_is_gate` already avoids.

So fusion gains a **veto**: a branch scoring below a floor rejects outright, regardless of the weighted sum. Two safeguards, both testable:

- an **unavailable** branch never vetoes — a probe too short to score cannot reject anyone;
- the floor sits far below the fusion threshold, so it fires on strong contrary evidence rather than on a merely unimpressive score.

**A veto is a false-reject risk, and the paper must price it.** Report the fitted floor *and* the FRR it costs, and run the ablation with vetoes disabled (`veto_thresholds={}`) so the veto's contribution is a measured quantity rather than an architectural assumption. An unmeasured veto is not a result.

#### 4.5.2 The veto floor was set on an imagined scale — found while wiring the API

The veto above shipped with a floor of 0.15 on the branch's [0, 1] scale, chosen by reasoning about that scale in the abstract. When the scorer and the fusion layer were finally connected end to end, it turned out **the veto could never fire.**

The CSBG branch score is a length-normalised LLR squashed by a logistic with scale 2.0, so a branch score `s` means a raw LLR of `2·ln(s/(1−s))`. Three landmarks:

| Branch score | Raw LLR | Reading |
|---|---|---|
| 0.50 | 0.00 | speaker model and background explain the probe equally well |
| 0.35 | −1.24 | background ≈3.5× more likely — strong evidence against |
| 0.15 | −3.47 | background ≈32× more likely |

Measured on the reference corpus, an impostor who contradicts the claimed speaker in *every* measured class scores **0.29**, because the logistic compresses hard once the LLR passes −2. A genuine speaker scores 0.93. So a floor at 0.15 sat *below the impostor mode*: the mechanism was in the code, was covered by tests, and would not have fired once on a real probe.

Two things made this invisible for as long as it was:

- the fusion tests use hand-written branch scores (A4 was written as `csbg=0.05`), so they never asked what the scorer emits;
- the scorer tests never look at fusion.

Nothing connected the two modules, and the constant lived in the one that could not check it. `tests/test_api.py::TestVetoCalibration` now closes that gap by asserting the floor sits above a decisive impostor and well below a genuine speaker, computed by the real scorer.

**The methodological point is worth a sentence in the paper, and it generalises past this project:** a threshold expressed in units of a score is only meaningful once you have measured what that score actually does. Two correct modules with a constant passed between them is exactly where this hides.

The floor is now 0.35 — the point at which the background model is about 3.5× more likely than the claimed speaker's. It remains a **starting point, not a fitted value**; §4.5.1's requirement to report the fitted floor and its FRR cost is unchanged.

---

## 5. Experiments — this is what makes or breaks the paper

### 5.1 The headline experiment (the money table)

Build a threat model with five attackers and show which branches stop which:

| | Attack | Attacker capability | ECAPA alone | +Knowledge | +CSBG (full) |
|---|---|---|---|---|---|
| **A1** | Replay | Old recording of the speaker | ✗ accepts | ✓ rejects | ✓ rejects |
| **A2** | Cut-and-paste splice | Splices recorded words into the answer | ✗ accepts | ~ partial | ✓ rejects |
| **A3** | Voice clone, *no* knowledge | XTTS-v2 / F5-TTS clone from 30s of audio | ✗ accepts | ✓ rejects | ✓ rejects |
| **A4** | Voice clone, *with* knowledge | Clone + attacker has scraped the answer from social media | ✗ accepts | ✗ accepts | **✓ rejects ← the claim** |
| **A5** | **Style-adaptive clone** | Clone + knowledge + LLM prompted to imitate the speaker's code-switch style | ✗ | ✗ | **? report honestly** |

**A4 is the paper.** If ECAPA + knowledge-check both accept a clone that knows the answer, and CSBG catches it, you have a real result.

**A5 is your credibility.** A reviewer *will* ask "what if the attacker also clones the switching style?" Run it. Report it even if it partially breaks the system. Include it as an explicit limitation and as future work (adversarial CSBG hardening). Preempting the obvious attack is worth more than hiding it.

#### 5.1.1 The one sentence the whole table rests on

> **The knowledge branch checks the fact. The CSBG checks the wrapper around it. The fact is stealable; the wrapper is the biometric.**

The challenge asks "what is your mother's name?" and the *fact* is "Lakshmi" — scrapeable off a birthday post. But nobody answers a question with a bare noun. The speaker says *"en amma peru Lakshmi"* or *"my mother's name is Lakshmi"* or *"amma name Lakshmi"*, and which of those they say, reliably, across sessions, is what the CSBG measures. The attacker knows the name. They do not know the sentence.

**This imposes a hard constraint on the challenge set:** a challenge answerable in one word has no wrapper and gives the CSBG nothing to measure. Questions must elicit a phrase. Report mean answer length per challenge template alongside the per-class results — a null result on a one-word challenge is a failure of the question, not of the hypothesis.

#### 5.1.2 A5 splits in two, and the split is itself a result

An attacker cannot read the victim's enrolled CSBG — it lives on the authentication server. They can only estimate it from speech they can overhear. So A5 runs twice:

| Condition | Attacker's style source | What it answers |
|---|---|---|
| **A5-oracle** | Handed the victim's exact enrolled CSBG | Upper bound on attacker power. Physically unrealisable. "Is this defence breakable *in principle*?" |
| **A5-observed** | Estimated from *N* seconds of public speech (a social-media video) | The condition a real attacker can actually reach. |

Sweeping *N* in A5-observed produces the same curve as the §5.3 enrolment-stability experiment, read from the other side:

> **The speech a defender needs to enrol a usable CSBG is the speech an attacker needs to steal one.**

One measurement answers both questions, and which number it lands on (30 seconds? five minutes?) decides whether the defence is practical at all. If a usable CSBG takes five minutes of speech to estimate, most victims are safe from most attackers. If it takes thirty seconds, anyone with a public Instagram is exposed. **Put this curve in the paper.**

Two failure modes in running this, both found while implementing it:

**An eavesdropping budget larger than the corpus is not a budget.** If *N* exceeds the speech the victim has on record, the attacker was handed all of it — which is the *oracle* condition, and filing that number in the observed row overstates the realistic attacker by however much the truncation would have cost. `api.attacks` now labels the run ORACLE whenever the budget fails to bind, and says so in the run's notes. Anyone sweeping *N* must check the same thing: the curve's right-hand tail is oracle, not observed, and where it stops being observed depends on the speaker.

**What the A5-observed curve actually measures is within-speaker consistency.** The attacker's estimate converges on the enrolled graph exactly as fast as the speaker is self-consistent. A speaker who says numbers in English 100% of the time is fully characterised by a handful of tokens; a speaker who does it 70% of the time needs far more speech before the estimate stabilises. So:

> **The more reliably a speaker code-switches, the better the biometric works and the more cheaply it is stolen.**

That tension is real, it is not obvious, and it is the most interesting thing in the A5 row. It also predicts the shape of the result: the CSBG's discriminative power and its stealability are driven by the *same* per-speaker quantity, so the per-speaker scatter of "CSBG margin" against "seconds to steal" should be positively correlated. **Plot it.** If it is, the defence has a characterisable weak population — the speakers it protects best — and naming that is worth more than a better mean.

#### 5.1.3 Report attack yield next to every clone row, or the table means nothing

The easiest way to accidentally fake this paper's headline result: generate clones so bad that ECAPA rejects them unaided, then report "A4 rejected" as if the CSBG did it. **An attack that fails because the clone was bad is not evidence the defence works.**

So: a clone counts as an A3/A4/A5 trial **only if the acoustic branch accepts it**. Clones that fail that screen are excluded from the rate and reported separately as **attack yield** — the fraction of generated clones that fooled ECAPA.

"A4: 0/40 accepted" means nothing at 5% yield and everything at 90% yield.

Low yield is publishable, but as a *different* claim: "open TTS cannot clone code-switched Tamil–English well enough to defeat a standard speaker verifier" is the low-resource-ness-as-defensive-asset finding (§8). It supports the framing. It is not a result about the CSBG, and the two must not be blurred.

*Note on TTS: verify Tamil support before committing. Coqui XTTS-v2's language list needs checking for `ta`; AI4Bharat IndicTTS and commercial multilingual TTS are fallbacks. If Tamil TTS quality is poor, that is itself a finding — see above and §8.*

#### 5.1.4 Four guards, enforced in code

`attacks.suite.AttackTable.paper_ready()` refuses to certify a row that has any of:

1. **simulated attacks** — a signal-processed pseudo-replay is not a replay; record through real hardware;
2. **inadmissible clones counted as rejections** — see §5.1.3;
3. **template-written A5** — A5 is a claim about a *capable* adversary, so its text must come from the LLM path;
4. **too few trials** — every cell carries a 95% Wilson interval (not normal-approximation: these rates sit at 0 and 1, where the normal interval runs outside [0,1] and is simply wrong), and cells below 30 trials are flagged.

Also report **per-speaker IAPMR, not just the mean.** If the full system stops every attack on 24 speakers and none on the 25th, that speaker is completely unprotected and the mean reads 96%.

### 5.2 Core verification results
- EER / minDCF for: ECAPA alone, CSBG alone, each pairwise fusion, full fusion
- DET curves
- Genuine vs. impostor score distributions per branch
- **Ablation over CSBG components:** lexical edges only / transition edges only / metrics only / full

#### 5.2.1 A veto breaks the usual way of reporting operating points

`eval/ablation.py` fits everything on a dev split and reports on test. Two things about it are not standard and both must be stated in the paper, because a reviewer who assumes the standard version will read the table wrong.

**Trials that cross the dev/test boundary are discarded, not assigned.** An impostor trial pairing a dev speaker's *model* with a test speaker's *probe* has one foot on each side; putting it in test means the threshold was partly fitted on that model. At `dev_fraction=0.4` this drops roughly half the impostor trials. Fewer trials is the cheaper error.

**"FAR at 1% FRR" does not exist for a system with a veto — report it as unattainable, not as 100%.** A veto rejects some genuine trials at *every* threshold, so it puts a hard floor under FRR. If that floor is 1.25%, no threshold reaches 1% FRR and the metric is undefined. The natural implementation returns 1.0 for "infeasible", which prints as `100.00` beside the baseline's `46.70` — and a reader compares them as measured rates, concluding fusion is catastrophic at an operating point fusion cannot occupy. It now returns NaN and renders `n/a`, with the veto count beside it.

The same subtlety has a second face. A vetoed trial is one no threshold accepts, i.e. a score of −∞ — which is what makes a DET curve possible for a vetoed system at all. But the empirical curve is normally anchored at "one below the minimum score", and one below −∞ is still −∞, so every vetoed trial gets accepted at the bottom of the sweep and the floor vanishes. The anchor must be built from the lowest **finite** score. Both of these were found by writing the test that asserts the operating point is unattainable, not by reading the code.

**The veto ablation needs a health warning attached to it.** Disabling the veto can *lower* EER while making the attack table worse, because EER is computed over genuine-vs-impostor trials and cannot see the attack rows. A row reading "vetoes disabled: −1.65% EER" with no context reads as "the veto is harmful". It is a trade, and §5.1 is the other half of it.

#### 5.2.2 The acoustic baseline is confirmed competent on this audio — and its pilot number must not be reported

The paper's central claim is that a clone defeats ECAPA and does not defeat the CSBG. That claim is only worth anything if the ECAPA baseline is a fair one, so it was checked against the real recordings as soon as the branch was wired up (`eval/branches.py`, pilot corpus, 4 speakers x 14 utterances, one session each):

- **Template self-consistency 0.85-0.95** per speaker across 9 enrolment clips. This is the diagnostic worth keeping in the enrolment flow: a low value means the clips disagree about who the speaker is, which in practice is a mislabelled file or a second voice in the room. Nothing here is mislabelled.
- **100% branch coverage** — every trial scored, no clip too short or unreadable.
- Genuine mean 0.937, impostor mean 0.632, and **no impostor scored above the weakest genuine trial**.

**That last number is not a result and must never be reported as one.** With four speakers there are 20 genuine and 60 impostor trials, drawn from one recording sitting per speaker — same phone, same room, same few minutes. Perfect separation under those conditions is the expected outcome for any competent embedder and carries no information about the system. What it does establish is that the audio path, the preprocessing and the checkpoint are all working, so a later failure to separate can be attributed to the corpus or the method rather than to plumbing. Report it, if at all, as a sanity check with the speaker count attached.

The reportable version of this row needs cross-session probes and enough speakers for the EER to be stable; §5.1's headline table is where it belongs.

#### 5.2.3 The answer matcher's rescaling was necessary, and is now measured

`matcher.py` rescales LaBSE cosine similarity from 0.5 rather than using it raw, on the reasoning that multilingual sentence embeddings put unrelated text well above zero. Measured on the real model: `Thanjavur` vs `தஞ்சாவூர்` scores **0.862** after rescaling, and `Thanjavur` vs `kothu parotta` scores **0.0**. The cross-script match — the case none of the other three matchers can handle — survives, and the unrelated pair floors out instead of reading as a partial match. Without the rescale the second pair would have contributed a non-trivial score to every impostor trial.

#### 5.2.4 The scripted pilot separates speakers for the wrong reason — measured

End-to-end run on the pilot: **7 speakers x 14 utterances**, one session each, 4395 tokens tagged by `gemini-3.1-flash-lite`, zero guessed. The pipeline works. The numbers do not mean what they appear to.

Tamil share of choice tokens, per speaker: **0.69, 0.03, 0.90, 0.63, 0.05, 0.03, 0.06**. A spread of 0.87 looks like a spectacular result and is an artifact of the elicitation design. Each speaker reads a *different* script (`SpeakerRecord.script_id`), and those scripts differ in language balance — four of the seven are English-dominant, with Tamil surviving only in food and festival vocabulary (`idli`, `sambar`, `poriyal`, `kuzhambu`, `Pongal`, `kolam`, `paal`). Speaker S02's FUNCTION_WORD class is 0.00 Tamil over 405 tokens, which is the correct answer for someone reading English prose and tells you nothing about how they speak.

So on this corpus a CSBG is a script classifier. `Corpus.reportability` already blocks any §5.1 claim on SCRIPTED provenance, and this is the magnitude of what it is blocking: separation that would otherwise look like the headline result. **Free-speech sessions are not an improvement to the corpus, they are the corpus.**

**And that separation does not convert into verification accuracy.** Run with `--real-branches --within-session`, the CSBG alone scores **26.32% EER [15.79–42.11]** against 50% for guessing, and no veto floor bought any FAR reduction inside the 2% FRR budget. This is the more useful half of the finding, and it cuts against the optimistic reading of the spread: a per-speaker aggregate can differ enormously while carrying almost nothing per probe, and verification only ever sees one probe. Anyone quoting the 0.87 spread as evidence the idea works should quote this row in the same breath.

The acoustic branch reads **0.00% EER** on the same run, which is equally not a result: enrolment and probe come from the same sitting, so it is measuring how well ECAPA memorises one microphone, one codec and one room. Both numbers need a second session before either means anything.

Two things are worth keeping from the run anyway, because neither depends on the script:

- **Word-level LID and semantic tagging work on real code-mixed Tamil ASR output.** All 21 classes populated, 0 guessed tokens over 4395.
- **37 tokens were English that Whisper wrote in Tamil script**, recovered by the confidence override rather than recorded as Tamil choices the speaker never made: `மாணிங்` (morning), `ஏவினிங்` (evening), `நைட்டுக்கு` (night), `யூஸ்வலி` (usually), `சிக்ஸ்` (six), `தெட்டிக்கு` (thirty), `பஸ்டாப்புக்கு` (bus stop), `போன்ல` (phone), `யூஸ்` (use). They concentrate in TIME_DATE (7), ACTION_VERB (6) and QUANTITY_MEASURE (4). `kavach.inspect_corpus --translit` lists them for checking; a wrong entry there is an English label on a word genuinely spoken in Tamil, which is the same bias pointing the other way.

The ASR came out well where it worked: **median WER 5.3%, mean 9.4%** over the 57 utterances that have both a reference and a usable transcript. Quote the median. The mean is dragged by two utterances where the speaker paraphrased the script instead of reading it, which is a deviation from the protocol rather than an ASR error, and WER cannot tell the two apart.

It also failed in a way worth stating plainly, because it is invisible in that number: **on 2 of 98 utterances Whisper translated the Tamil into English instead of transcribing it.** `CODE_MIX_PROMPT` exists to prevent exactly this and is a hint, not a guarantee. The output is fluent, correct English, so nothing downstream objects — the utterance simply enters the graph as a speaker who chose English for every token. One was recovered by re-decoding; the other returned zero Tamil on three attempts and is excluded from the corpus with its reason recorded, which is why the run reports 97 usable utterances rather than 98. `Transcript.looks_translated()` now catches this at transcription time and, deliberately, without needing a reference — the free-speech sessions this project depends on have none.

### 5.3 CSBG stability analysis (a strong secondary result)
> **How much code-switched speech does a speaker need before their CSBG converges?**

Plot CSBG self-similarity and EER against enrolment duration (30s, 1m, 2m, 3m, 5m). This answers the sparsity worry head-on, and it is genuinely useful to anyone else who wants to build on the idea. Also report **cross-session stability** — does a speaker's CSBG hold up when recorded weeks apart, or does it drift with topic and mood? (Be prepared for the honest answer to be "it drifts somewhat.")

### 5.4 Resource contribution
A released corpus: **~25–30 bilingual Tamil–English speakers × ~5 minutes**, with:
- Word-level language tags (TA/EN/NEUTRAL/NE)
- Semantic-class annotations
- Speaker IDs, session metadata, device metadata
- The attack recordings (replay, splices, clones) as a paired spoof set

Low-resource venues weight resource contributions heavily. **This alone can carry an accepted paper** even if the CSBG results come out weaker than hoped. Treat it as your insurance policy — and check whether SPELLL has a dedicated resource-paper track.

*Get consent forms signed before you record anything. Biometric voice data + personal knowledge-graph facts is genuinely sensitive — see §9.*

### 5.5 Fairness audit
Pretrained ECAPA is trained on English-centric VoxCeleb. Measure and report:
- EER on monolingual Tamil vs. monolingual English vs. code-mixed trials
- Score degradation by speaker's dominant language
- Gender and device breakdown

If verification is measurably worse for code-mixed speech, that is a bias finding on-theme for SPELLL's stated ethics objective — and it strengthens the motivation for a language-aware second factor.

---

## 6. Datasets

**Primary — self-collected (required).** 25–30 speakers minimum. Below ~20 the EER confidence intervals are too wide to claim anything. Recruiting bilingual Tamil–English speakers at an Indian engineering college is very feasible; budget 2 weeks.

**Supplementary — verify each of these actually exists and is licensed before relying on it:**
- **IndicSUPERB** (AI4Bharat) — includes speaker-identification tasks across Indian languages; useful as a baseline reference point
- **Common Voice Tamil** — monolingual Tamil, useful for the fairness comparison arm
- **Shrutilipi / IndicVoices** (AI4Bharat) — large-scale Indic speech
- **MUCS 2021** code-switching ASR challenge data — Hindi-English and Bengali-English (not Tamil, but useful for a cross-lingual generalisation experiment if time allows)
- **DravidianLangTech / FIRE code-mixed datasets** — mostly *text*, not speech, but useful for building the semantic-class lexicon and validating the LID

> ⚠️ I have not verified the current contents, licences, or availability of any of these. Check each one yourself before citing it. Do not put a dataset in the paper you haven't downloaded.

**Explicitly not needed:** ASVspoof. We generate our own attack set, which is the point — no spoof-labelled corpus required.

---

## 7. Feasibility

**What makes this actually buildable in 8 weeks:**
- No model training. ECAPA, Whisper, LaBSE, the LLM — all frozen, inference only.
- The CSBG is counting, smoothing, and divergence — a few hundred lines of NumPy/NetworkX, not a research codebase.
- The LLM does the hard NLP (word-level LID + semantic tagging) via API, which removes the single biggest engineering risk.
- Attack generation is off-the-shelf TTS.
- Data collection is the long pole, and it is parallelisable — start recruiting in week 1, before the code is finished.

**Real risks, ranked:**

| Risk | Severity | Mitigation |
|---|---|---|
| CSBG doesn't discriminate (EER too high to be useful) | **Critical** | Run a 5-speaker pilot in week 2 before committing. If the score distributions don't separate at all, pivot to the corpus + fairness paper (§12). |
| CSBG drifts across sessions | High | Measure it (§5.3) and report honestly; frame as soft biometric requiring periodic re-enrolment. |
| Word-level LID accuracy too low | High | Hand-validate early; LLM adjudication + script heuristics should get you well past useful. |
| Too few speakers | High | Start recruiting week 1. 20 is the floor. |
| Tamil TTS quality too poor to make A4 a fair test | Medium | Reframe as a finding (§8); use best available multilingual TTS and report its quality honestly. |
| A5 breaks the system | Medium | Report it. It's a limitation, not a disqualification — every biometric has an adaptive-attacker bound. |
| SpeechBrain/PyTorch/torchaudio version hell | Low but annoying | Pin versions in week 1. This bit the old proposal too. |

---

## 8. The framing hook: low-resource-ness as a security asset

Worth a paragraph in the introduction and probably in the abstract, because it's memorable and it's exactly SPELLL's worldview:

> The prevailing narrative treats low-resource status purely as a deficit. In the security setting it is partly an **asset**. Voice-cloning and TTS systems are trained overwhelmingly on high-resource languages; their Tamil output is measurably worse than their English output, and their code-switched output is worse still, because the training data for naturalistic Tamil–English switching barely exists. Meanwhile the *defender* — the genuine speaker — code-switches natively and effortlessly. KAVACH deliberately operates in the region of linguistic space where the attacker's models are weakest and the legitimate user is strongest. Data scarcity, from the attacker's side, is a defensive moat.

Be careful to also state the counter-argument, because it's real: this moat **erodes as multilingual TTS improves**, so the contribution is a method and a measurement of the current gap, not a permanent defence. Say so explicitly.

---

## 9. Ethics (SPELLL lists this as an explicit objective — do not skip it)

- **Consent:** written informed consent covering voice biometric collection, personal-fact collection, retention period, and release of the derived corpus. Get institutional ethics approval if your department requires it — start this in week 1, it has latency.
- **The SKG is sensitive PII.** Personal facts about family, home, and education, in a graph, next to a voiceprint. Store hashed/encrypted; never release the raw SKG in the corpus, only the CSBG structure with entities anonymised.
- **Right to withdraw** — speakers can request deletion of their data and their derived graphs.
- **Bias:** report the fairness audit (§5.5) as a first-class result, not a footnote.
- **Dual-use:** the attack suite (§5.1) is generated to evaluate a defence. Release the *defence* code and the *genuine* corpus; consider withholding or gating the generated clone set, and state your release decision and reasoning in the paper.
- **Accessibility limitation:** a challenge-response system that requires speaking a fluent code-mixed answer disadvantages speakers with speech disabilities, and the knowledge factor disadvantages users with memory impairment. State this.

---

## 10. Tech stack and repo layout

```
speech_processing/
├── backend/
│   ├── api/               FastAPI app, routes, schemas (Pydantic)
│   ├── asr/               faster-whisper wrapper, VAD, audio preprocessing
│   ├── lid/               word-level language ID (script → translit → LLM adjudication)
│   ├── embedding/         SpeechBrain ECAPA-TDNN wrapper, cosine scoring
│   ├── csbg/              graph construction, smoothing, JSD scoring, cohort z-norm
│   ├── skg/               rdflib triple store, SPARQL queries, Tamil cultural ontology
│   ├── challenge/         LLM challenge generation, targeting policy, replay ledger
│   ├── matcher/           cross-lingual semantic + phonetic + entity answer matching
│   ├── fusion/            score calibration and logistic-regression fusion
│   ├── attacks/           replay, splice, TTS clone, style-adaptive clone generators
│   └── eval/              EER/minDCF/DET, ablations, fairness slices, stability curves
├── kavach/                React + Vite + TypeScript UI. `src/api/types.ts` is the
│                          contract: `backend/api/schemas.py` mirrors it field for
│                          field, and `tests/test_api.py` parses the .ts file and
│                          fails the build when the two drift. TypeScript cannot
│                          check a JSON payload at runtime, so nothing in the
│                          frontend build catches a renamed field — it shows up as
│                          a blank panel in a demo.
├── data/
│   ├── raw/               enrolment + login recordings (gitignored)
│   ├── attacks/           generated spoof audio (gitignored)
│   └── annotations/       word-level LID + semantic class tags
├── notebooks/             analysis, figures for the paper
└── paper/                 LaTeX, figures, tables
```

| Layer | Choice |
|---|---|
| Backend | Python 3.11, FastAPI, Uvicorn |
| Speaker embedding | SpeechBrain `spkrec-ecapa-voxceleb` |
| ASR | `faster-whisper` large-v3 (int8) + IndicWhisper/IndicConformer comparison |
| LLM | API-based (word-level LID + semantic tagging + challenge generation) |
| Sentence embeddings | LaBSE / Indic SBERT (*verify*) |
| Transliteration | `indic-transliteration` (ISO-15919) |
| Knowledge graph | `rdflib` for the SKG (real triples + SPARQL), `networkx` for CSBG scoring |
| TTS (attacks) | Coqui XTTS-v2 / F5-TTS / IndicTTS (*verify Tamil support*) |
| Storage | SQLite + filesystem audio store |
| Metrics | scikit-learn, `pyeer` or custom EER/DET |
| Frontend | React + Vite + TypeScript, Recharts for DET/score plots, Cytoscape.js for graph viz |

**Multimodality, stated honestly:** KAVACH fuses three modalities — **acoustic** (waveform → embedding), **lexical** (ASR text), and **structural** (knowledge graph). It is *not* audio-visual. Say "multi-representational fusion of speech, text, and graph modalities" and don't let a reviewer catch you implying video.

---

## 11. Eight-week plan (from 2026-08-04)

| Week | Milestone |
|---|---|
| **1** | Confirm SPELLL deadline & template. Start ethics approval + speaker recruitment. Pin the environment; get ECAPA + Whisper running end-to-end on 2 test clips. |
| **2** | **PILOT (go/no-go).** Record 5 speakers × 3 min. Build the LID + semantic tagger. Build CSBG v1. **Check: do genuine and impostor CSBG scores separate at all?** If no → pivot per §12. |
| **3** | Full data collection (target 25–30 speakers). SKG interview + storage. Challenge generator. |
| **4** | Answer matcher (semantic + phonetic + entity). Fusion layer. Full pipeline runs end-to-end. |
| **5** | Attack generation A1–A5. Hand-annotate the LID validation set. |
| **6** | Full evaluation: EER/DET, all ablations, stability curves, fairness slices. Generate every figure. |
| **7** | Write the paper. Frontend wiring + demo video. |
| **8** | Internal review, tighten, submit. Buffer for the inevitable. |

The frontend is deliberately late — it's for the demo and the figures, not for the science. Don't let it eat weeks 2–5.

---

## 12. Minimum Publishable Unit (the fallback)

If the pilot in week 2 shows CSBG doesn't discriminate, **do not force it.** Pivot to a paper that is guaranteed publishable at a low-resource venue and still fits the theme:

> **"A Tamil–English Code-Switched Speech Corpus for Voice Authentication, with a Knowledge-Graph-Grounded Challenge Framework and a Fairness Audit of Pretrained Speaker Embeddings"**

That paper needs only: the corpus (§5.4), the KG+LLM challenge generation and the code-switch-tolerant matcher (§4.3–4.4, with the exact-match failure quantified), and the fairness audit (§5.5). All three are low-risk and independent of whether the CSBG works. You then report the CSBG as a *negative result with analysis* — which, done properly, is respectable and reviewers at resource-focused venues accept it.

Decide this at the end of week 2, not week 6.

---

## 13. Related work to position against

Read these before writing the intro; they define the gap you're claiming.

- **Speaker embeddings:** Desplanques et al., *ECAPA-TDNN*, Interspeech 2020. Snyder et al., *x-vectors*, ICASSP 2018.
- **Idiolect as biometric (your closest prior art — cite it prominently and honestly):** Doddington, *Speaker recognition based on idiolectal differences between speakers*, Eurospeech 2001. This is the work you are extending into the multilingual/code-switch setting. Do not pretend it doesn't exist.
- **Code-switching theory:** Myers-Scotton, *Duelling Languages* (Matrix Language Frame model), 1993.
- **Code-mixing metrics:** Gambäck & Das, *On measuring the complexity of code-mixing*, ICON 2014 (CMI). Guzmán et al., *Metrics for modeling code-switching across corpora*, Interspeech 2017 (I-index, M-index, burstiness). Barnett et al. (M-index).
- **Anti-spoofing:** the ASVspoof 2019/2021 challenge papers (Todisco et al., Yamagishi et al.) — cite to position, not to compete.
- **Voice cloning threat:** recent zero-shot TTS/VC work (XTTS, VALL-E-family, F5-TTS) and audio-deepfake detection surveys.
- **Indic speech resources:** AI4Bharat's IndicSUPERB / IndicVoices papers.
- **Code-switched ASR:** MUCS 2021 challenge overview and Indic code-switching ASR literature.

### Literature search — COMPLETED 2026-08-04

Five searches run across speaker verification, anti-spoofing, behavioural biometrics, code-switching, and KG-based authentication. Result: **Tier-1 claim survives; Tier-2 claim killed by patent prior art** (see §3).

**Must-cite sources found, with why each matters:**

| Source | Why it matters to you |
|---|---|
| [Doddington, *Speaker recognition based on idiolectal differences*, Eurospeech 2001](https://www.isca-archive.org/eurospeech_2001/doddington01_eurospeech.pdf) | The acknowledged ancestor. Word n-gram idiolect for speaker ID. Cite prominently and position as "we extend this to the multilingual/code-switch setting." |
| [US Patent 10,362,016 — Dynamic knowledge-based authentication](https://image-ppubs.uspto.gov/dirsearch-public/print/downloadPdf/10362016) | **Kills the Tier-2 claim.** KG of personal life events → generated auth challenges. Cite it, don't compete with it. |
| [US Patent 7,289,957](https://image-ppubs.uspto.gov/dirsearch-public/print/downloadPdf/7289957) | Random-combination challenge-response speaker verification. Confirms the *original* proposal's core was already patented. |
| [Modeling the Bilingual Phonological Idiolect Through a Translanguaging Lens](https://pmc.ncbi.nlm.nih.gov/articles/PMC12611410/) | **Closest structural prior art.** Network-science graph of a bilingual idiolect. Differentiate explicitly (§3.4). |
| [An Investigation of Code-Switching Attitude Dependent Language Modeling](https://www.researchgate.net/publication/236219497_An_Investigation_of_Code-Switching_Attitude_Dependent_Language_Modeling) | Establishes speakers have measurable, clusterable CS styles. Your premise — cite as motivation, not as your finding. |
| [Modeling Topics and Sociolinguistic Variation in Code-Switched Discourse](https://arxiv.org/pdf/2512.03334) | Recent work on what conditions switching (topic, sociolinguistics). Supports your semantic-class conditioning design. |
| [A Survey of Threats Against Voice Authentication and Anti-Spoofing Systems](https://arxiv.org/html/2508.16843v2) | Recent threat-model survey. Use to frame the attack taxonomy in §5.1. |
| [Vulnerabilities of Audio-Based Biometric Authentication Against Deepfake Speech Synthesis](https://arxiv.org/html/2601.02914v1) | Recent evidence that cloning defeats commercial ASV. This is your motivation paragraph. |
| [What You Read Isn't What You Hear: Linguistic Sensitivity in Deepfake Speech Detection](https://www.researchgate.net/publication/392085487_What_You_Read_Isn't_What_You_Hear_Linguistic_Sensitivity_in_Deepfake_Speech_Detection) | Shows *linguistic content* affects spoof detection — adjacent support for "linguistic signal carries security-relevant information." |
| [ASVspoof 5](https://www.sciencedirect.com/science/article/pii/S0885230825000506) | Current anti-spoofing benchmark. Cite to position, not to compete. |
| [SwitchLingua code-switching dataset](https://arxiv.org/html/2506.00087v1) | Large multi-ethnic CS dataset — check whether it has speaker labels you could reuse. |
| Microsoft/SpeechOcean 60h CS corpus (Tamil-English incl.) | **Verify speaker labels.** Affects whether your corpus contribution stands (§3.3). |

**Still to check yourself:** Google Scholar and the ACL Anthology directly (my searches were general web). Search `"code-switching" speaker verification`, `code-mixing biometric`, and `language choice authentication` there. Also check the [code-switching-papers reading list](https://github.com/gentaiscool/code-switching-papers) — it is comprehensive and will tell you fast if something was missed.

---

## 14. Paper skeleton

1. **Introduction** — voice cloning breaks timbre-based verification; low-resource code-switched speech offers an orthogonal, hard-to-clone signal; the asset framing (§8).
2. **Related Work** — §13, ending with the explicit gap statement.
3. **Code-Switch Behaviour Graphs** — formalism, ontology, edges, scoring, cohort normalisation.
4. **KG-Grounded Adaptive Challenge Generation** — SKG, SPARQL traversal, LLM prompting, discriminative targeting.
5. **Code-Switch-Tolerant Verification** — the matcher, and why exact match fails.
6. **Corpus** — collection protocol, annotation, statistics, release terms.
7. **Experiments** — threat model, headline table, ablations, stability, fairness.
8. **Discussion & Limitations** — A5, drift, accessibility, moat erosion.
9. **Ethics.**
10. **Conclusion.**

**Figures that will sell it:** (a) the architecture diagram from §4; (b) a real CSBG rendered as a graph for two contrasting speakers, side by side — this is your Figure 1 and it should be beautiful; (c) DET curves per branch; (d) the CSBG stability-vs-duration curve; (e) the attack table as a heatmap.

---

## 15. What I'd do in the next 48 hours

1. Confirm the SPELLL-2026 submission deadline and template. Everything else depends on it.
2. Run the literature search in §13. Kill or confirm the Tier-1 claim now.
3. Start the ethics/consent paperwork — it has the longest latency of anything here.
4. Post the speaker recruitment call.
5. Pin the Python environment and get ECAPA + Whisper producing output on one Tamil–English clip.

Then come back and I'll build the backend.
