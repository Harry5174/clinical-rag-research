# Sprint 2.5 — Consolidated Methodology Revision Specification

## 1. Sprint goal

Consolidate approved Phase 02 methodology findings into an edit-ready handoff and decide whether the project can proceed to a separately gated paper-editing phase.

## 2. Files inspected

Sanitized Phase 01 authority, claim, and reviewer records; Phase 02 Sprints 2.1–2.4 specifications and decision records; current paper section/claim markers; and bibliography context. No code inspection was needed because the approved specifications were consistent.

## 3. Commands run

Ran the approved git baseline, sanitized-document inventories/read-only inspection, targeted paper section mapping, bibliography context search, and final git checks. No tests, experiments, reproductions, builds, or compilation were run.

## 4. Consolidated methodology findings

The manuscript mixes several supported local implementation facts with unsupported or overbroad methodological framing. The consolidated correction set separates exact implementation details, controlled/simulated evidence, local target-subject-filtered evidence, unresolved score treatment, and future-analysis-only claims.

## 5. Retrieval/indexing correction summary

- Correct literal `IndexFlatIP` / exact-inner-product language to `IndexFlatL2` / `METRIC_L2`.
- Retain normalized embedding facts, with only conditional normalized-vector ranking equivalence language.
- Do not call raw L2 output cosine, inner-product, or validated semantic similarity.
- For Phase 9, state full local-index search, target-subject filtering, then first 50 filtered candidates; final evaluation uses deduplicated top-1/top-5.
- State that the optional cross-encoder is not part of the reproduced Phase 9 path.
- Retain the exact current-index build-lineage caveat.

## 6. TIMER scoring correction summary

- Retain the local formula: alpha times protocol-specific semantic input plus intent-selected beta times exponential temporal decay.
- Treat alpha, beta, lambda, and tau as fixed local settings, not documented tuned choices.
- Describe negative historical beta as a recency penalty relative to equal semantic inputs, not a validated old-document bonus.
- Separate controlled supplied fixture scores from Phase 9 transformed L2-derived values.
- Preserve the score-treatment decision: the Phase 9 transform is not validated cosine/inner-product similarity and may affect ranking interpretation.

## 7. Evaluation protocol correction summary

- Label 96.0% versus 52.5% as the exact controlled/simulated fixture-score aggregate.
- Restrict equal-0.95 language to the applicable controlled scenarios.
- Label 58%/80% versus 22%/69% as the exact local target-subject-filtered Phase 9 aggregate with score-treatment limitation.
- Define semantic-only separately by protocol.
- Frame missing comparators, Negation Recency interpretation, and Terminology Drift uncertainty as limitations/future analysis.

## 8. Sidecar / Virtual Graph correction summary

- Use **temporal metadata sidecar**, not Virtual Graph.
- State generated/randomized `offset_days`, placeholder `note_date`, copied section metadata, and section-derived heuristic intent label.
- Do not present these fields as observed clinical temporal provenance.
- Do not claim graph structures, nodes, edges, traversal, graph reasoning, or graph novelty.

## 9. Intent router correction summary

- Describe fixed regex/rule-based Current, Historical, and Trend routing.
- State beta mapping and matched-input normalized match-share behavior as implementation details.
- State tau as a fixed local threshold, not a tuned/calibrated setting.
- Correct the universal neutral-fallback statement: zero matches default to Current at confidence 0.5 and receive the Current beta.
- Leave accuracy, calibration, trigger/fallback rates, and near misses to future analysis.

## 10. Chunking/corpus/index-lineage caveats

The local processed-subset corpus counts are descriptive, not production-scale evidence. Section-aware/header-propagation capability exists, but the exact build lineage for the current index is not durably established. Do not overstate chunking sophistication or use it to make unqualified current-index claims.

## 11. Unsupported claims to remove/defer

- Separate 20-query tuning-set and held-fixed provenance.
- Existing supplementary-material availability.
- Router quality/calibration/safety claims.
- Broad clinical utility, deployment readiness, or realistic/unconstrained evaluation claims.
- Graph-architecture claims.

## 12. Section-by-section edit plan summary

Apply the detailed edit table in `paper-methodology-edit-spec.md`: bound the Abstract/Introduction/Conclusion; correct Methods; label Experimental Setup and Results by protocol; strengthen Discussion/Limitations; and remove/defer unsupported Declarations/Supplement claims.

## 13. Future analysis/code-decision dependencies

- Score analysis before claims about Phase 9 semantic-score validity or temporal-effect magnitude; code/reproduction only if analysis supports a correction decision.
- Router study before router-quality claims.
- Comparator, sensitivity/tuning, and provenance studies before stronger related claims.
- Sanitized release work only if supplement availability will be asserted.

## 14. Phase 02 exit recommendation

**Phase 02 complete — proceed to a gated paper-editing phase.** Unresolved issues do not block scoped correction/removal/limitation editing; they do block stronger claims.

## 15. Scope confirmation

Only the four approved Sprint 2.5 documentation files were created. No paper, bibliography, code, tests, notebooks, data, results, indexes, or configuration files were changed. No experiments, reproductions, tests, compilation, commits, pushes, or external services were used.
