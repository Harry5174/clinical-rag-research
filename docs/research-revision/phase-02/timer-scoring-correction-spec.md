# TIMER Scoring Correction Specification — Sprint 2.3

## 1. Executive summary

The implemented TIMER score is `final_score = alpha * semantic_score + beta * exp(-lambda * offset_days)`, ranked descending. Its semantic input is protocol-dependent: the controlled hard-negative evaluator consumes supplied fixture scores, while Phase 9 consumes a transformed FAISS L2 distance. These must not be described as one universal cosine-like score.

Phase 9 applies `clip((D + 1) / 2)` to L2 distance `D`. The transform is nondecreasing in distance; it neither inverts distance nor establishes cosine similarity. Its use with descending TIMER sorting is a score-direction risk requiring separate analysis and a possible future code-correction/reproduction decision. This sprint changes neither code nor paper.

## 2. Current paper scoring claims

The paper presents a cosine-style semantic term fused with temporal decay, states fixed alpha/beta/lambda/tau values, characterizes negative beta as promoting older material, claims fixed `Ssem = 0.95` across hard-negative pairs, and asserts undocumented development-set tuning. The formula/constants are partly supported; the universal semantic-score meaning, blanket fixed-score statement, and tuning provenance are not.

## 3. Implementation scoring evidence

| Finding | Evidence |
|---|---|
| TIMER formula | `TIMERScorer.score_node()` computes alpha times supplied semantic input plus beta times exponential decay. |
| Decay | `compute_temporal_decay()` returns `exp(-lambda * max(0, offset_days))`. |
| Beta selection | Regex-derived Current, Historical, and Trend labels select beta; low confidence returns zero. |
| Reusable TIMER path | `TIMERRetriever` uses cross-encoder `rerank_score` when present, then sorts TIMER scores descending. |
| Controlled path | The hard-negative evaluator ranks fixture `semantic_score` values and uses them in TIMER scoring. |
| Phase 9 path | The evaluator transforms FAISS L2 output with `clip((D + 1) / 2)` before ablation scoring and descending sorting. |

## 4. TIMER formula truth

Safe implementation wording is:

`Score = alpha * S + beta(intent, confidence) * exp(-lambda * max(0, offset_days))`.

`S` is not invariant: it is a supplied fixture value in the controlled evaluator, a cross-encoder score when the optional reusable reranking path is used, and a transformed L2-derived evaluator value in Phase 9. A future paper edit must not apply one “raw cosine similarity” explanation to every path.

## 5. alpha / beta / lambda truth

| Parameter | Implemented setting | Evidence-bound description |
|---|---:|---|
| `alpha` | 0.6 in `TIMERScorer`; 1.0 for Phase 9 semantic-only | Fixed local setting. |
| `beta_current` | +0.8 | Adds a decaying recency term. |
| `beta_historical` | -0.3 | Subtracts the decaying recency term. |
| `beta_trend` | 0.0 | Neutral temporal contribution. |
| `lambda` | 0.005 | Fixed exponential-decay setting. |
| `tau` | 0.40 in `TIMERScorer` | Low confidence returns beta zero. |

Phase 9 explicitly supplies the same full TIMER alpha/beta configuration. The paper’s separate-development-set tuning claim is unsupported; parameter-selection, calibration, and sensitivity claims must be removed or deferred.

## 6. negative beta behavior

For historical intent, beta is -0.3. Decay is near one for recent candidates and approaches zero as offset grows. Thus negative beta penalizes recent candidates relative to otherwise equal older candidates; it does not add a positive old-document bonus.

Safe wording: “Historical intent applies a negative coefficient to the recency-decay term, reducing scores for more recent candidates relative to otherwise equal older candidates.” Do not call this clinically validated safety, guaranteed promotion, or a full remedy for semantic error.

## 7. semantic score meaning by protocol

| Protocol/path | Semantic input | Safe interpretation |
|---|---|---|
| Controlled hard-negative baseline | Fixture `semantic_score`, descending sort. | Supplied controlled value, not newly computed embedding retrieval. |
| Controlled hard-negative TIMER | Same fixture value in TIMER formula. | Mechanism-isolation input. Equal 0.95 fixtures occur in Semantic Collision and Real-World Mining; inspected generation paths use differing supplied values in Negation Recency and Terminology Drift. |
| Reusable `TIMERRetriever` | `rerank_score` when available. | Cross-encoder output with source-noted scale uncertainty. |
| Phase 9 evaluator | `clip((D + 1) / 2)` from FAISS L2 output. | Transformed L2-derived value; not established semantic similarity. |

The paper must correct its blanket fixed-0.95 statement. The suite is controlled throughout, but equal-0.95 behavior is scenario-specific.

## 8. L2-distance transform issue

Phase 9 computes `S_norm = clip((D + 1) / 2, 0, 1)` after `IndexFlatL2` search. Smaller L2 values are nearer, but this transform is nondecreasing in `D`: before clipping it increases with distance, and for `D >= 1` it is saturated at one.

It neither inverts L2 distance nor implements a demonstrated cosine/inner-product conversion. It is `REQUIRES_SCORE_ANALYSIS` and `REQUIRES_CODE_CORRECTION_DECISION`, not a valid description of semantic similarity, cosine similarity, inner-product similarity, calibration, or normalization.

## 9. score-direction analysis

| Stage | Direction |
|---|---|
| FAISS L2 output | Lower is better. |
| Phase 9 transform | Larger distance produces a larger or equal transformed value. |
| TIMER/ablation final sort | Higher final score is better. |

Within the unsaturated region, Phase 9’s semantic component favors larger L2 distance when other terms are equal; clipping may collapse distinct larger distances. This identifies risk but does not itself invalidate the reproduced aggregate without separately approved analysis or corrected reproduction.

## 10. safe paper wording constraints

- The local scorer combines a supplied semantic term with intent-modulated exponential temporal decay.
- The listed parameters are fixed local implementation/evaluation settings.
- Negative historical beta changes the recency term relative to equal semantic inputs.
- Controlled hard-negative outcomes are fixture-score mechanism-isolation results.
- Phase 9 aggregates are local target-subject-filtered results with an unresolved L2-score-treatment limitation.

## 11. unsafe paper wording

- One universal raw-cosine meaning for `Ssem`.
- Fixed 0.95 in every hard-negative scenario.
- Separate 20-query tuning provenance.
- Valid L2-to-cosine/inner-product conversion in Phase 9.
- Identical semantic-only baseline semantics across both protocols.
- Score calibration, sensitivity robustness, router reliability, or clinical-safety conclusions.

## 12. issues requiring future analysis or code decision

1. `REQUIRES_SCORE_ANALYSIS`: quantify transform/clipping impact and interaction with temporal terms.
2. `REQUIRES_CODE_CORRECTION_DECISION`: decide whether score direction requires an approved correction and reproduction.
3. `REQUIRES_FUTURE_ANALYSIS`: establish scale compatibility for cross-encoder scores and temporal terms if that path is reported.
4. `REMOVE_OR_DEFER`: unsupported tuning, calibration, and robustness claims.
