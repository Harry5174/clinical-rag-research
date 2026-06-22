# Methodology Gap Report — Sprint 2.1

## 1. Executive summary

The current implementation supports a reproducible, local controlled hard-negative protocol and a reproducible local target-subject-filtered Phase 9 protocol. It does not support several broader manuscript framings. The most urgent corrections are the FAISS metric, generated sidecar provenance, and precise evaluation-protocol wording. A second group of claims is implemented but requires narrower wording because tuning, router quality, sensitivity, broad baseline comparison, deployment relevance, or public release availability are not evidenced.

This report relies on sanitized Phase 01 records and read-only source inspection. It contains no raw notes, records, queries, answers, retrieved content, or raw result rows.

## 2. High-risk methodology mismatches

| Gap | Evidence | Methodology gap status | Required action |
|---|---|---|---|
| Paper says `IndexFlatIP` / exact-inner-product retrieval; inspected constructors and index use `IndexFlatL2` / `METRIC_L2`. | Phase 01 authoritative-results manifest; index preparation and base retrieval source. | METHOD_CORRECTION_REQUIRED | Correct method, table, and scoring-language references to L2 search over normalized embeddings. Do not preserve literal inner-product wording. |
| Paper’s sidecar discussion can imply observed temporal provenance; generator uses randomized offsets and a placeholder date. | Phase 01 provenance records; `scripts/create_sidecar.py`. | METHOD_CORRECTION_REQUIRED | Label the sidecar generated/simulated. Narrow or reconsider the “Virtual Graph” claim. |
| End-to-end evaluation is called realistic or unconstrained although the current code target-subject filters before retaining 50 candidates. | Phase 01 exact Phase 9 reproduction; `scripts/run_end_to_end_eval.py`. | EVALUATION_PROTOCOL_CLARIFICATION_REQUIRED | State target-subject filtering, full-index search, filtering order, and top-50 selection. Remove unconstrained/deployment-like framing. |
| The manuscript claims a separate 20-query tuning set and held-fixed hyperparameters without a supporting record. | Phase 01 claim matrix and manifest. | REMOVE_OR_DEFER_CLAIM | Remove/defer the tuning assertion until a documented development-set and analysis record exists. |
| The manuscript states that supplementary materials are provided, but no reviewed sanitized package exists. | Phase 01 manifest and safety record. | REMOVE_OR_DEFER_CLAIM | Remove/defer availability wording pending an approved supplement-release package. |

## 3. Medium-risk methodology mismatches

| Gap | Evidence | Methodology gap status | Required action |
|---|---|---|---|
| Normalized embeddings are used with L2 retrieval, while manuscript text uses cosine/inner-product language and Phase 9 applies `clip((D + 1) / 2)` to L2 output before descending TIMER sorting. The transform is increasing in distance and is not a demonstrated similarity conversion. | Index/retrieval/evaluator source; Phase 01 review; Sprint 2.3 score-direction inspection. | METHOD_CORRECTION_REQUIRED | Use exact L2-over-normalized-embeddings wording; retain an explicit score-treatment limitation and obtain methodology review plus a code-correction decision if analysis finds material impact. |
| Section-aware and header-propagation implementations exist, but the exact lineage from the 1,206-vector index to a specific chunking run is not durably established. | Chunking and index-builder source; Phase 01 manifest. | EVALUATION_PROTOCOL_CLARIFICATION_REQUIRED | Confirm exact index-build lineage before making a categorical header-propagation statement. |
| Router categories, regex matching, confidence computation, and a conditional tau fallback are implemented, but no router accuracy, calibration, trigger-rate, or near-miss analysis exists. A zero-match query returns Current at confidence 0.5 and therefore bypasses the below-0.40 neutral fallback. | `scoring.py`; Phase 01 reviewer mapping; Sprint 2.4 source inspection. | METHOD_CORRECTION_REQUIRED | Describe a heuristic mechanism only; correct the ambiguous-query fallback statement; defer performance/safety implications. |
| Fixed alpha, beta, lambda, and tau values are encoded, but sensitivity and tuning evidence are absent. | `scoring.py`; `run_end_to_end_eval.py`; Phase 01 claim matrix. | SCOPE_SOFTENING_REQUIRED | Call them fixed local settings, not tuned/validated choices. |
| The semantic-only baseline is implemented, but empirical alternatives such as simple temporal filters or related architectures are absent. | Evaluator source; Phase 01 reviewer mapping. | SCOPE_SOFTENING_REQUIRED | Limit comparative claims to the tested semantic-only baseline and acknowledge missing baselines. |
| The 165-subject / 1,206-chunk corpus size is supported, but manuscript framing can imply a broader retrieval setting. | Phase 01 manifest and paper inventory. | SCOPE_SOFTENING_REQUIRED | Retain counts and pair them with a local-scale limitation. |

## 4. Supported methodology claims

- BGE-base-en-v1.5 is selected by the inspected relevant index/retrieval/evaluation code, with a 768-dimensional inspected current index.
- The current index is L2-based, and relevant encoding code normalizes embeddings.
- Current, Historical, and Trend heuristic intent categories are implemented. Matched inputs use normalized match-count confidence and below-threshold confidence returns neutral beta; zero-match inputs instead default to Current at 0.5.
- TIMER’s current local configuration includes alpha 0.6, lambda 0.005, current beta +0.8, historical beta -0.3, and trend beta 0.0.
- The controlled hard-negative protocol has an authoritative existing aggregate of 96.0% TIMER versus 52.5% semantic-only, strictly for the reproduced controlled/simulated local protocol.
- The Phase 9 target-subject-filtered protocol has authoritative existing local aggregates of 58% / 80% full TIMER and 22% / 69% semantic-only for Accuracy@1 / Recall@5.

All supported claims above need their protocol boundaries retained; support for a mechanism is not support for deployment readiness, broad clinical utility, or exhaustive comparative performance.

## 5. Claims needing terminology reframing

- Replace `IndexFlatIP`, exact-inner-product, and unqualified cosine wording with precise description of L2 search over normalized embeddings.
- Describe temporal metadata as generated/simulated sidecar data, not source-observed clinical temporal provenance.
- Define “Virtual Graph” narrowly as a metadata-sidecar mechanism, unless a supervisor decides to remove or rename the label.
- Call the router heuristic/rule-based, not calibrated or demonstrated safe.
- Describe hard-negative performance as controlled/simulated stress-test evidence, not general clinical retrieval validation.

## 6. Claims needing evaluation-protocol clarification

- Separate the controlled hard-negative protocol from the target-subject-filtered Phase 9 end-to-end protocol.
- Specify that Phase 9 searches the local index, filters candidates to the target subject, then retains 50 filtered candidates and evaluates top-1/top-5 at note level.
- Clarify whether the optional two-stage cross-encoder architecture participated in reported evaluations; the reproduced Phase 9 evaluator does not invoke it.
- Confirm the exact chunking/index-build lineage before asserting that header propagation generated the reported current index.
- Pair the 165-subject / 1,206-chunk facts with their local-subset limitation.

## 7. Unsupported claims to remove/defer

- Separate 20-query development-set tuning and “held fixed” provenance.
- Availability of a reviewed supplementary package.
- Any claim that the existing local/restricted data, indexes, detailed results, or sidecar are publicly distributable.
- Any empirical claim of router calibration/accuracy, sensitivity robustness, broader baseline superiority, or deployment readiness that exceeds the inspected evidence.

## 8. Components needing future analysis

| Component | Reason | Needed outcome |
|---|---|---|
| L2-distance score transformation | End-to-end evaluator labels an inner-product-style conversion while consuming L2 search output. | Methodology review of whether the reported semantic score treatment is appropriate and how it should be described. |
| Hard-negative semantic-score scope | Paper describes fixed 0.95 scores across the suite, but equal 0.95 fixtures apply only to designated controlled scenarios; the evaluator consumes supplied scenario scores. | Correct the blanket description and specify controlled fixture-score use by scenario. |
| Exact index/chunking lineage | Multiple chunking strategies/builders exist; durable binding to the current evaluated index is incomplete. | A sanitized build/provenance record. |
| Router quality | No standalone quality, calibration, fallback, or regex-near-miss evidence. | A separate analysis plan and results before performance claims. |
| Hyperparameter tuning and sensitivity | Constants are implemented but undocumented as tuned; sensitivity is unmeasured. | A documented analysis plan before retaining tuning/robustness claims. |
| Comparative baselines | Only semantic-only variants are currently evidenced. | Approved baseline design and evaluation before comparative-superiority claims. |
| Target-subject filtering appropriateness | It is material to difficulty and interpretation. | Supervisor-approved protocol rationale before final manuscript wording. |

## 9. Recommended next Phase 02 sprints

1. **Sprint 2.2 — Methodology Correction Specification:** draft line-level future-edit instructions for L2 retrieval, normalized embeddings, sidecar provenance, protocol labels, corpus scope, hard-negative scope, and removal/deferment claims. No paper edits.
2. **Sprint 2.3 — Pipeline and Evaluation Protocol Specification:** resolve index/chunking lineage, candidate-flow wording, cross-encoder participation, score-treatment review inputs, and target-subject-filter protocol description.
3. **Sprint 2.4 — Analysis-Gap Plan:** propose, but do not execute, router, sensitivity, comparative-baseline, and failure/uncertainty analyses.
4. **Future supplement-release sprint:** obtain supervisor decisions and a sanitization/release manifest before any availability statement is drafted.

## Status summary

- `METHOD_CORRECTION_REQUIRED`: FAISS metric, generated sidecar provenance.
- `EVALUATION_PROTOCOL_CLARIFICATION_REQUIRED`: Phase 9 filtering, candidate flow, exact chunking/index lineage.
- `TERMINOLOGY_REFRAMING_REQUIRED`: normalized-embedding/L2 wording and Virtual Graph terminology.
- `SCOPE_SOFTENING_REQUIRED`: router, constants, corpus scale, controlled findings, baseline comparison, and deployment language.
- `REMOVE_OR_DEFER_CLAIM`: tuning provenance and supplementary-material availability.
- `NO_CHANGE_NEEDED`: none of the 24 required components are unconditionally change-free; even supported claims require protocol-bound framing.
