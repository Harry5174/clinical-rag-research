# Sprint 2.3 — TIMER Scoring and Evaluation Protocol Correction Specification

## 1. Sprint goal

Create a sanitized specification for honest TIMER-score and evaluation-protocol description, including controlled hard-negative fixtures, the reproduced Phase 9 protocol, and the unresolved L2 score-treatment risk.

## 2. Files inspected

- Approved Phase 01 manifests, correction matrix, reviewer-concern map, and exit record.
- Sprint 2.1 and Sprint 2.2 Phase 02 specifications.
- Frozen paper and bibliography for claim/citation context only.
- Relevant scorer, retriever, evaluator, metrics, and test files.

No raw data, result, index-content, or restricted report was inspected by content.

## 3. Commands run

- Required Git baseline/final checks; approved documentation inventories and reads.
- Approved paper/bibliography claim searches and source/script/test evidence searches.
- Read-only reviewer-concern and line-numbered source inspection.

No tests, experiments, reproductions, metric generation, compilation, network calls, staging, commits, or pushes ran.

## 4. Paper scoring/evaluation claims found

The paper describes one cosine-style semantic score, universal fixed 0.95 hard-negative scoring, tuned parameters, negative-beta temporal inversion, and an unconstrained/realistic Phase 9 framing. These claims require protocol-specific correction or limitation.

## 5. Implementation scoring evidence found

The scorer implements alpha times semantic input plus intent-selected beta times exponential decay. Hard-negative evaluation uses supplied fixtures; Phase 9 uses a clipped L2-derived value. Final sorting is descending, and target-subject filtering precedes top-50 selection.

## 6. TIMER scoring truth

The formula and fixed local constants are implemented. Negative historical beta penalizes recency relative to equal semantic inputs. The formula does not establish common source, scale, direction, or calibration for all semantic inputs.

## 7. Evaluation protocol truth

The 96.0%/52.5% result is controlled/simulated fixture-evaluator evidence. The 58%/80% and 22%/69% result is reproduced evidence for the local target-subject-filtered Phase 9 protocol with unresolved L2 score treatment.

## 8. Safe claims

- `SAFE_TO_DESCRIBE`: formula, fixed local constants, intent-dependent beta, and reproduced protocol flow.
- `SAFE_WITH_LIMITATION`: controlled hard-negative and filtered Phase 9 aggregates.
- `REQUIRES_METHOD_CORRECTION`: blanket fixed-0.95, cosine/inner-product semantics, and tuning wording.

## 9. Unsafe claims

- Universal raw-cosine `Ssem` meaning.
- Fixed 0.95 across all scenarios.
- Validated L2-to-similarity conversion.
- Unconstrained/deployment-like Phase 9 retrieval.
- Tuning, calibration, causal, or broad-superiority claims.

## 10. Future analysis/code-decision dependencies

- `REQUIRES_SCORE_ANALYSIS`: transform direction/clipping and temporal interaction.
- `REQUIRES_CODE_CORRECTION_DECISION`: whether to correct and reproduce.
- `REQUIRES_FUTURE_ANALYSIS`: score scale, router, sensitivity, failures, and comparators.
- `REMOVE_OR_DEFER`: unsupported tuning and overbroad score/deployment claims.

## 11. Scope confirmation

- No paper, bibliography, code, test, notebook, data, result, index, or configuration file was edited.
- No experiment, test, reproduction, metric generation, compilation, external call, commit, or push occurred.
- Documentation is sanitized and contains no raw clinical, patient, query, answer, retrieved-content, or result-row material.
