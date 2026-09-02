# Experiment run 2026-08-18T21:50:34+00:00

**These numbers are NOT reportable.**

- commit `5028cfc93fc1`
- python 3.11.15 on macOS-26.6.1-arm64-arm-64bit
- corpus `kavach_freespeech_v1` (RECORDED), 5 speakers, 70 utterances
- seed 0, dev fraction 0.4, 500 bootstrap resamples
- word-level LID accuracy: **unmeasured** (no `--goldset`)
- acoustic (ECAPA): scored 900 of 900 trials (100.0%)
- knowledge (answer matcher): scored 72 of 900 trials (8.0%)

## Why these numbers may not be reported

1. 6 of 70 utterances are excluded and contribute to no graph (6x Whisper translated it into English rather than transcribing the Tamil; three re-decodes with a Tamil-script prompt returned either English again or hallucinated script, so its tokens would be language choices the speaker never made) -- report this count and these reasons alongside any result
2. 5 speakers have one session, so no cross-session trial exists for them and §5.3 cannot be computed on this corpus
3. the knowledge (answer matcher) branch scored only 8% of trials (knowledge (answer matcher): 72/900 trials scored (8.0%); unmeasured: claimed speaker never answered this prompt at enrolment x828); the unscored trials are fused from the remaining branches, so the table mixes two different systems
4. enrolment and probe speech were split within a session, which measures how well the estimator memorises one recording sitting

## Files

| Path | What it is |
|---|---|
| `results.json` | Every number. **The paper reads from here.** |
| `report.md` | The same run as prose tables |
| `tables/*.tex` | `tabular` bodies for `\input{}` |
| `figures/*.pdf` | Vector figures; `.png` beside each |

Nothing here is hand-edited. Re-run `python -m kavach.experiments` to regenerate.
