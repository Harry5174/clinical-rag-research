# Evaluation Protocol Correction Specification — Sprint 2.3

## 1. Executive summary

The hard-negative and Phase 9 results arise from distinct protocols. The hard-negative evaluation is controlled/simulated and ranks supplied fixtures. Phase 9 searches the full local index, filters to the target subject, retains 50 filtered candidates, applies TIMER ablation scoring, deduplicates note IDs, and evaluates top-1/top-5. Neither protocol establishes unconstrained clinical retrieval, deployment readiness, or broad comparator superiority.

## 2. Controlled hard-negative protocol

The evaluator loads a controlled fixture, ranks semantic-only output solely by supplied `semantic_score`, and applies TIMER scoring to the same supplied scores and temporal offsets. It calculates scenario and aggregate correctness. It does not perform live embedding encoding, FAISS candidate retrieval, or natural-score retrieval.

## 3. What 96.0% and 52.5% mean

The reproduced 96.0% TIMER and 52.5% semantic-only figures are authoritative existing aggregates for the current 200-item controlled/simulated fixture and current evaluator. They are top-ranked expected-retrieval outcomes in that exact protocol, not naturalistic end-to-end, clinical-utility, production, or untested-comparator results.

## 4. Fixed Ssem = 0.95 and artificiality caveat

Equal supplied `Ssem = 0.95` creates controlled semantic collisions in Semantic Collision and Real-World Mining. It is not universal: inspected generation paths use differing supplied scores in Negation Recency and Terminology Drift. Every hard-negative semantic score is a controlled evaluator input rather than a newly computed retrieval output.

Future paper wording must restrict fixed-0.95 language to the applicable scenarios and call them mechanism-isolation tests. Naturally varying semantic-retrieval claims require a separate approved analysis.

## 5. Phase 9 target-subject-filtered protocol

The evaluator encodes a query, searches the complete local FAISS index, obtains the expected item’s target subject, filters returned entries to that subject, and retains the first 50 entries after filtering. It computes temporal decay, applies every ablation, sorts final scores descending, and deduplicates note identifiers. The filter occurs after full-index search and before top-50 selection; it is a material local condition.

## 6. What 58% / 80% and 22% / 69% mean

- Full TIMER: 58% Accuracy@1 and 80% Recall@5.
- Semantic-only: 22% Accuracy@1 and 69% Recall@5.

These are authoritative existing only for the current local target-subject-filtered evaluator, source-linked local index/metadata, controlled input, and unresolved L2-score treatment. They must not be represented as general clinical-retrieval or deployment results.

## 7. Semantic-only baseline definitions

| Protocol | Definition | Relationship |
|---|---|---|
| Controlled hard-negative | Descending supplied fixture `semantic_score`; no temporal term. | Direct fixture-score baseline. |
| Phase 9 | `alpha = 1.0`, all beta values zero, descending transformed L2-derived evaluator value after filtering. | Distinct ablation baseline with unresolved score direction. |

The labels match, but their semantic-score sources do not. They must be described in their own protocol contexts.

## 8. Top-50 to top-5 protocol

For Phase 9: full-index L2 search; target-subject filter; first 50 filtered entries; TIMER score by configuration; descending final-score sort; note-ID deduplication; first note for Accuracy@1 and first five for Recall@5 membership. The optional cross-encoder is not invoked in this reproduced path.

## 9. What each protocol proves

- **Controlled hard-negative:** current fixture/evaluator behavior can isolate temporal-mechanism ranking under supplied-score scenarios.
- **Phase 9:** the local target-subject-filtered implementation reproduced its documented aggregate under the exact current protocol.

## 10. What each protocol does not prove

- Naturalistic semantic retrieval with naturally generated scores.
- Unfiltered, population-level, or deployment-ready clinical retrieval.
- Validity of the Phase 9 L2 score treatment.
- Router calibration, parameter robustness, causal explanation of scenario outcomes, clinical safety, or broad comparator superiority.

## 11. Required paper-correction instructions

1. Label the 200-item evaluation controlled/simulated and fixture-score-based.
2. Restrict fixed-0.95 wording to equal-score scenarios.
3. Retain 96.0%/52.5% only with the controlled-protocol boundary.
4. Report Phase 9 58%/80% and 22%/69% prominently but as target-subject-filtered local results.
5. State the full-search, filter-then-top-50, TIMER-score, deduplicated-top-5 flow.
6. Do not call Phase 9 unconstrained, cosine-scored, or free of the score-treatment limitation.
7. Remove/defer undocumented tuning and broad comparative/deployment inference.

## 12. Reviewer concern mapping

| Concern | Required response |
|---|---|
| P11–P13: fixed `Ssem = 0.95` | Correct the blanket claim; identify equal-score scenarios; separate mechanism isolation from natural-score efficacy. |
| P14: Phase 9 58% vs 22% | Give the reproduced local result prominence with filtering and score-treatment limitations. |
| P15: Negation Recency no gain | Report null result plainly; defer causal and safety interpretation. |
| P16: Terminology Drift n=20 | Qualify small-sample result; do not add unsupported interval or causal explanation. |
| P30–P31: missing comparators | Limit claims to tested semantic-only variants and disclose absent empirical comparators. |
