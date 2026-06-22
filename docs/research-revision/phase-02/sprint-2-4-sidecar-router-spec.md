# Sprint 2.4 — Sidecar, Virtual Graph, and Intent Router Specification

## 1. Sprint goal

Establish sanitized, implementation-grounded correction instructions for temporal sidecar provenance, Virtual Graph naming, intent-router behavior, confidence/tau wording, and reviewer concerns P08 and P19–P21. This is a specification-only sprint; it does not alter the manuscript, implementation, data, results, indexes, or evaluation outputs.

## 2. Files inspected

- Sanitized Phase 01 authority, claim-correction, and reviewer-concern records.
- Phase 02 truth table, methodology-gap report, retrieval/indexing, scoring, evaluation, and FAISS decision records.
- `latex_publication/v1/sn-article.tex` for targeted claim mapping and `sn-bibliography.bib` for context only.
- `scripts/create_sidecar.py`, `src/app/research/retrieval/scoring.py`, `src/app/research/retrieval/timer.py`, `two_stage.py`, the two evaluation scripts, and relevant router/TIMER tests.

No raw data, raw results, raw sidecar JSON, or index contents were inspected by content.

## 3. Commands run

- Git safety baseline: repository path, short status, branch, remotes, and whitespace check.
- Approved documentation inventories and read-only Phase 01/02 evidence review.
- Targeted paper/bibliography searches for sidecar, graph, router, confidence, threshold, and provenance claims.
- Source-only searches plus read-only inspection of the approved sidecar, scoring, router, evaluation, and test files.
- Source-only searches for graph primitives and router-quality evidence.

No tests, experiments, reproductions, builds, or paper compilation were run.

## 4. Paper sidecar/router claims found

The paper presents a temporal sidecar and “Virtual Graph” architecture, names Current/Historical/Trend routing, describes normalized confidence and tau fallback, and makes safety/novelty implications that exceed the inspected evidence. It also states that a tau threshold makes ambiguous queries neutral, which is not complete for zero-pattern inputs.

## 5. Implementation evidence found

The sidecar generator creates a note-keyed JSON structure with random category-driven offsets, a placeholder date, section data, and a heuristic intent label. The reusable retriever reads note offsets and sections. The scorer is fixed regex/rule logic with count-based confidence for matched inputs. Zero matches yield Current/0.5; below-0.40 confidence yields beta 0.

## 6. Temporal sidecar truth

`offset_days` is generated/randomized; `note_date` is a placeholder; `section` is copied from processed input; and `intent_ground_truth` is a section-derived heuristic label. These fields must not be presented as observed clinical temporal provenance. The sidecar is `SUPPORTED_BUT_NEEDS_REFRAMING` as an implementation artifact and `CONTRADICTED_BY_PROVENANCE` for observed-time claims.

## 7. Virtual Graph naming recommendation

Recommend `RENAME_TO_TEMPORAL_SIDECAR`, using **temporal metadata sidecar**. There is no inspected graph structure, edge/node representation, graph query, or traversal. Retain no graph novelty claim.

## 8. Intent router truth

The three categories are Current, Historical, and Trend. Fixed regex lists are counted, the largest count wins, and beta maps to +0.8, -0.3, or 0.0. This is `SUPPORTED_AS_HEURISTIC`; it is not a trained classifier or independently evaluated routing system.

## 9. Confidence/tau truth

For matched inputs, confidence equals winning category matches divided by all matches. `tau = 0.40` is a fixed local threshold: confidence below it yields beta 0. Zero matches do not follow that path; they return Current/0.5 and receive +0.8. Tau has no inspected tuning, calibration, sensitivity, or validation evidence.

## 10. Safe claims

- Generated local temporal metadata sidecar used by the scorer.
- Fixed regex/rule-based routing among three labels.
- Conditional below-threshold neutral beta behavior.
- Fixed local values for alpha, beta, lambda, and tau.
- Protocol-bound controlled and target-subject-filtered evaluation use of the router mechanism, without router-quality inference.

## 11. Unsafe claims

- Source-observed sidecar dates/offsets or clinical temporal provenance.
- Graph implementation, graph traversal, or graph novelty.
- Trained-classifier, router-accuracy, calibration, or safety claims.
- Universal neutral fallback for ambiguous/unmatched inputs.
- Tuned or validated tau threshold.

## 12. Reviewer concern mapping

- P08: rename to temporal metadata sidecar; remove graph and observed-provenance implications.
- P19: no standalone router accuracy evidence; require future labelled evaluation.
- P20: no confidence distribution, trigger rate, zero-match rate, or fallback-rate evidence; require future aggregate analysis.
- P21: no regex near-miss or error analysis; require future error-analysis protocol.

## 13. Future analysis dependencies

- Source-derived, sanitized temporal provenance if observed-time claims are desired.
- Router quality analysis covering labels, errors, zero matches, ties, threshold behavior, calibration, and ranking impact.
- Tuning/sensitivity evidence before any validated-parameter claim.
- Supervisor decision only if retaining “Virtual Graph” despite the recommendation.

## 14. Scope confirmation

Only the six approved Phase 02 documentation files were changed: four new Sprint 2.4 specifications and two prior records materially refined for zero-match router behavior. No paper, bibliography, code, tests, notebooks, data, results, indexes, or configuration files were changed.
