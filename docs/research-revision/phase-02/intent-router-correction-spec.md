# Intent Router Correction Specification — Sprint 2.4

## 1. Executive summary

The router is a fixed, regex-based heuristic with three labels: Current, Historical, and Trend. It is not a trained classifier and has no standalone accuracy, calibration, trigger-rate, confidence-distribution, or near-miss evidence. `tau = 0.40` is a fixed local conditional threshold, not a tuned or calibrated threshold. Crucially, a zero-match input defaults to Current at confidence 0.5 and consequently receives the current beta; it does not take the neutral fallback described for below-threshold confidence.

## 2. Current paper router claims

The paper characterizes the router as a three-category regex mechanism, defines normalized match-count confidence, and states that ambiguity falls back to neutral weighting at `tau = 0.4`. The mechanism exists, but the universal ambiguity/fallback statement is incomplete because the zero-match branch bypasses it.

## 3. Implementation evidence

- `scoring.py:39-58` defines fixed regex lists for Historical, Current, and Trend.
- `scoring.py:60-75` counts matching patterns and normalizes the winning count by the total match count when at least one pattern matches.
- `scoring.py:67-68` returns Current with confidence 0.5 for zero matches.
- `scoring.py:77-97` returns beta zero only for confidence strictly below 0.40; otherwise Current maps to +0.8, Historical to -0.3, and Trend to 0.0.
- `timer.py:48-88` applies the selected beta in the reusable retriever.
- `run_hard_negative_eval.py:205-220` uses the same scorer for its controlled fixture evaluation; `run_end_to_end_eval.py:70-148` uses the same classifier for the local target-subject-filtered protocol.

## 4. Intent categories

| Label | Selection mechanism | Beta when confidence is at least tau |
|---|---|---:|
| Current | highest fixed-regex match count, or zero-match default | +0.8 |
| Historical | highest fixed-regex match count | -0.3 |
| Trend | highest fixed-regex match count | 0.0 |

This is `SUPPORTED_AS_HEURISTIC`, not evidence of correct intent recognition.

## 5. Regex / pattern logic

The implementation lowercases the incoming text, evaluates fixed regular expressions, counts category matches, and selects the maximum count. The implementation contains no learned model, fitted parameters, probabilistic classifier, error model, or calibration routine. Ties rely on the insertion order of the category-score mapping; no tie-specific handling is implemented.

No standalone report measures false positives, false negatives, regex near misses, or coverage. Such claims are `REQUIRES_FUTURE_ANALYSIS`.

## 6. Confidence scoring

For at least one regex match, confidence is:

`winning category match count / total matches across all categories`.

This is a normalized heuristic match share, not a calibrated probability. For zero matches, no ratio is available; the implementation emits Current at 0.5. It must not be described as probability-like confidence, calibrated certainty, or a validated risk indicator.

## 7. tau threshold

`tau = 0.40` is the `TIMERScorer` constant. A confidence strictly below the threshold returns beta 0. It is a fixed local router setting and has no inspected tuning, calibration, sensitivity, or validation evidence.

The zero-match branch returns 0.5, so it does **not** neutralize through this threshold. The manuscript must not say all ambiguous or unmatched queries use the neutral fallback.

## 8. beta connection

Beta selects the temporal contribution in the scorer: +0.8 for Current, -0.3 for Historical, and 0.0 for Trend or below-threshold fallback. The temporal score is a mechanism implementation detail; it does not establish that beta selection is clinically appropriate, safe, or validated.

## 9. Fallback/default behavior

| Condition | Returned label/confidence | Beta outcome |
|---|---|---|
| No regex matches | Current / 0.5 | +0.8 |
| One or more matches, winning share below 0.40 | winning label / calculated share | 0.0 |
| One or more matches, winning share at least 0.40 | winning label / calculated share | label-specific beta |

The first row is the material correction to prior fallback wording. Separately, missing sidecar metadata defaults an offset to zero in `TIMERRetriever`; that is a sidecar-coverage behavior, not a router-confidence fallback.

## 10. What is supported

- `SUPPORTED_AS_HEURISTIC`: fixed pattern sets, count-based label selection, and the beta lookup are implemented.
- `SUPPORTED_WITH_LIMITATION`: confidence is a match-share calculation only for matched inputs; the below-threshold beta-zero condition exists.
- The controlled and Phase 9 evaluators invoke this mechanism, but their retrieval aggregates are not router-quality evaluations.

## 11. What is unsupported

- `UNSUPPORTED_ACCURACY_CLAIM`: no standalone router accuracy, precision, recall, or error analysis.
- `UNSUPPORTED_CALIBRATION_CLAIM`: no calibration procedure, reliability analysis, or confidence distribution.
- `REQUIRES_FUTURE_ANALYSIS`: trigger rate, zero-match rate, tie rate, false positives, false negatives, near misses, and impact on retrieval metrics.
- `REMOVE_OR_DEFER`: claims that tau makes routing safe, robust, validated, or tuned.

## 12. Required paper-correction instructions

1. Call the router fixed regex/rule-based heuristic logic, not a trained classifier.
2. State that `tau = 0.40` is a fixed local setting, not a tuned or validated threshold.
3. Correct any statement that ambiguity or no-match cases universally fall back to neutral beta.
4. If default behavior is described, state that zero matches default to Current at 0.5 in the inspected implementation.
5. Remove performance, calibration, clinical-safety, and reliability implications unless future analysis supplies evidence.

## 13. Unsafe claims

- “Classifier accuracy,” “calibrated confidence,” or “validated confidence threshold.”
- “Ambiguous or unmatched queries always use semantic-only/neutral routing.”
- “The safety threshold prevents incorrect temporal modulation.”
- Any claim that controlled or end-to-end retrieval aggregates independently validate the router.

## 14. Reviewer P19–P21 mapping

| Concern | Sprint 2.4 response direction |
|---|---|
| P19: router accuracy | No quality result exists; restrict to mechanism description and require a labelled evaluation. |
| P20: confidence distribution/trigger rate | No distribution, trigger, zero-match, or fallback-rate report exists; require instrumentation and sanitized aggregate reporting. |
| P21: regex near misses | No near-miss/error inventory exists; require a predefined annotation and error-analysis protocol. |

## 15. Future analysis requirements

Before any positive router-performance claim, conduct a separately approved, sanitized analysis with a defined label source, sampling plan, zero-match/tie/threshold accounting, confusion metrics, confidence calibration assessment, and the effect of router outcomes on ranking. This sprint neither designs final metrics nor runs that analysis.
