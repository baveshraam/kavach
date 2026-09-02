## Verification results

Split: dev: 2 speakers / 16 trials | test: 3 speakers / 36 trials | cohort: 48 trials (z-norm only)
Branches measured: speaker_embedding, csbg, knowledge

```
weights    : speaker_embedding=1.000, csbg=0.000, knowledge=0.000
             (logistic regression on dev)
threshold  : 0.8733 (dev EER operating point)
veto       : none
```

| Configuration | EER % [95% CI] | minDCF | FAR@1%FRR | FRR@1%FAR | n gen/imp |
|---|---|---|---|---|---|
| ECAPA alone | 0.00 [0.00-0.00] | 0.0000 | 0.00 | 0.00 | 12/24  [!] |
| CSBG alone | 50.00 [33.33-70.83] | 0.8333 | n/a | 83.33 | 12/24 (6 vetoed)  [!] |
| + Knowledge | 0.00 [0.00-0.00] | 0.0000 | 0.00 | 0.00 | 12/24  [!] |
| + CSBG only | 0.00 [0.00-0.00] | 0.0000 | 0.00 | 0.00 | 12/24  [!] |
| Full fusion | 0.00 [0.00-0.00] | 0.0000 | 0.00 | 0.00 | 12/24  [!] |

`[!]` marks a configuration with fewer than 30 trials on a side; its interval is too wide to compare against another row.

minDCF parameters: p_target=0.05, c_miss=1.0, c_fa=1.0.

### Ablations

**Scope is not decoration.** Each Δ is against the un-ablated baseline *of its own scope*, and rows in different scopes are not comparable with each other. Scoring ablations are measured on the branch they change, because a branch that another one dominates shows +0.00 for every switch and the zero says nothing about the switch -- see `ablate_scoring`.

| Removed | Scope | EER % | Δ EER | Note |
|---|---|---|---|---|
| equal branch weights | full system | 16.67 | +16.67 | What the logistic-regression fit was worth. |
| low-signal classes included | csbg | 41.67 | -8.33 | FUNCTION_WORD, NAMED_ENTITY and OTHER put back. Excluding them is a hypothesis stated in ontology.LOW_SIGNAL_CLASSES; a positive delta confirms it on this data, a negative one retires it. |
| transition stream removed | csbg | 50.00 | +0.00 | P(lang given class and prev_lang) dropped, its weight redistributed. Transition cells are ~4x sparser than lexical ones, so this is the row that says whether sequence information survives short probes. |
| graph metrics removed | csbg | 50.00 | +0.00 | CMI and I-index dropped. They are already ramped down on short probes by csbg.scoring, so a near-zero delta here is expected and is a check on that ramp rather than a finding. |
| lexical stream removed | csbg | 50.00 | +0.00 | Transitions and metrics alone. The complement of the row above: together they say how much of the CSBG is carried by which language a class takes versus how the speaker moves between them. |
| cohort z-norm removed | csbg | 45.83 | -4.17 | Raw LLRs are not comparable across claimed speakers: an unusual speaker produces large-magnitude scores for everyone. Expect this row to read ~0 on a single branch and still matter in fusion. EER is rank-based, so a per-speaker rescale only moves it by *reordering* speakers against each other -- which needs the population to vary in score scale in the first place. Fusion is not rank-based: it adds the branch to others on a shared [0, 1] scale, where the shift is exactly what makes the sum comparable. A near-zero delta here alongside a large one in the configuration table is the expected shape, not a contradiction. |

### CSBG stability against enrolment budget

Read left to right: how much speech a defender needs. Read as an attacker's eavesdropping budget: how much they need to steal it.

| Utterances | ~seconds | EER % [95% CI] |
|---|---|---|
| 2 | 48 | 47.06 [35.29-63.71] |
| 5 | 120 | 45.59 [34.15-58.49] |

### Caveats

- CSBG scores are cohort z-normalised. Statistics are fitted from impostor trials whose *probe* is a dev speaker, which covers test models (3/3 test speakers) without using a test probe -- see `cohort_fitting_trials`. Report un-normalised results too: z-norm usually helps materially, and hiding that is hiding a design decision.
- No veto floor bought any FAR reduction inside the 2% FRR budget on dev. Report that the veto was fitted and discarded -- that is a result about the branch, not an omission.

## Corpus coverage

5 speakers | 3503 tokens (2805 language choices)

A class needs about 4 observations before a speaker's own evidence outweighs the backoff prior (SmoothingConfig.class_alpha).

| Class | Tokens | Speakers with own evidence |
|---|---|---|
| NUMBER | 74 | 5/5 |
| TIME_DATE | 220 | 5/5 |
| KINSHIP | 69 | 5/5 |
| FOOD | 178 | 5/5 |
| PLACE_LOCAL | 102 | 5/5 |
| PLACE_GLOBAL | 20 | 2/5 |
| TECH_DIGITAL | 30 | 3/5 |
| EDU_WORK | 85 | 5/5 |
| MONEY_COMMERCE | 39 | 4/5 |
| EMOTION_STATE | 90 | 5/5 |
| BODY_HEALTH | 54 | 5/5 |
| TRANSPORT | 32 | 4/5 |
| RELIGION_FESTIVAL | 39 | 4/5 |
| MEDIA_ENTERTAIN | 67 | 5/5 |
| DISCOURSE_MARKER | 129 | 5/5 |
| POLITENESS | 3 | 0/5 |
| QUANTITY_MEASURE | 214 | 5/5 |
| ACTION_VERB | 520 | 5/5 |
| FUNCTION_WORD | 558 | 5/5 |
| NAMED_ENTITY | 2 | 0/5 |
| OTHER | 280 | 5/5 |

| Prompt | Utterances | Mean tokens | Target hit rate |
|---|---|---|---|
| p01_family | 5 | 54.6 | 100% |
| p02_food | 5 | 43.2 | 100% |
| p03_commute | 5 | 50.0 | 100% |
| p04_money | 5 | 45.0 | 80% |
| p05_phone | 5 | 53.8 | 100% |
| p06_study | 5 | 74.2 | 80% |
| p07_festival | 5 | 62.8 | 100% |
| p08_travel | 5 | 50.4 | 80% |
| p09_film | 5 | 42.6 | 80% |
| p10_numbers | 5 | 41.0 | 100% |
| p11_health | 5 | 60.8 | 100% |
| p12_market | 5 | 47.6 | 100% |
| p13_hometown | 5 | 66.6 | 100% |
| p14_control_name | 5 | 8.0 | 100% |
