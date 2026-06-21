# Phase Lineage Map

## Timeline

| Phase / date | Commit or artifact | Intended representation | Current status |
|---|---|---|---|
| Phase 4, 2026-01-28 | dddee37 | validation/planning for hard-negative work | HISTORICAL |
| Initial hard-negative work, 2026-01-29 | 9b9473a | mining, preparation, controlled evaluator | HISTORICAL |
| Expanded hard-negative seed data, 2026-03-03 | 68c6b2 | configurable expansion of scenario seed data | HISTORICAL |
| Phase 5 v2, 2026-03-03 | d84f260 plus results/phase5_v2 | statistical hard-negative pipeline and 90.5% aggregate table | HISTORICAL; CONFLICT_UNRESOLVED |
| Phase 5, 2026-03-25 | 4b3ac68 plus results/phase5 | later 96.0% aggregate table and initial Phase 9 script | HISTORICAL; CONFLICT_UNRESOLVED |
| Phase 9 filtered path, 2026-03-26 | 1b6ab26 plus filtered result | target-subject-filtered end-to-end ablation | CURRENT implementation; NEEDS_REPRODUCTION |
| Sprint 1.2, 2026-06-21 | e8d926c | sanitized result-provenance record | CURRENT governance record |
| Sprint 1.3 | current documentation | lineage verification and future reproduction plan | CURRENT governance record |

## Hard-negative lineage

| Evidence | Phase 5 | Phase 5 v2 | Conclusion |
|---|---:|---:|---|
| aggregate rows | 200 | 200 | Same size |
| baseline-correct aggregate | 105 | 105 | Same baseline |
| TIMER-correct aggregate | 192 | 181 | 11-outcome difference |
| overall TIMER accuracy | 96.0% | 90.5% | Conflicting headlines |
| scenario ID overlap | 200 / 200 | 200 / 200 | Same result query-ID set |
| changed baseline outcomes | 0 | 0 | No baseline change |
| changed TIMER outcomes | 11 in terminology drift | 11 relative to Phase 5 | Difference concentrated in one scenario |
| evaluator source blob | current equals d84f260 | current equals d84f260 | No committed evaluator-source split |
| scoring behavior shape | unchanged through 4b3 and HEAD | unchanged through 4b3 and HEAD | No committed scoring behavior split |
| input/result Git tracking | ignored/untracked | ignored/untracked | No immutable input-to-result binding |

Decision: Phase 5 and Phase 5 v2 are historical conflicting aggregates. The local v2 input is a current reproduction candidate, but neither headline is CURRENT paper authority.

## Phase 9 lineage

| Evidence | Unfiltered | Filtered | Conclusion |
|---|---|---|---|
| script revision | initial March 25 implementation | March 26 implementation | Filtered is later |
| candidate protocol | direct top-50 search | complete search then target-subject filtering then top-50 | Material protocol change |
| output path | end_to_end_results.csv | end_to_end_results_filtered.csv | Current code produces filtered path |
| semantic-only result | 1% Accuracy@1 / 1% Recall@5 | 22% / 69% | Material result change |
| full TIMER result | 1% / 4% | 58% / 80% | Material result change |
| classification | SUPERSEDED | CURRENT; NEEDS_REPRODUCTION | Do not use unfiltered output for current protocol |

## Current implementation relationship

paper claim -> implementation -> result -> lineage status

- Controlled hard-negative claim -> scripts/run_hard_negative_eval.py -> Phase 5 or Phase 5 v2 aggregates -> CONFLICT_UNRESOLVED; NEEDS_REPRODUCTION
- End-to-end ablation claim -> scripts/run_end_to_end_eval.py -> results/phase9/end_to_end_results_filtered.csv -> CURRENT; NEEDS_REPRODUCTION
- FAISS metric claim -> index constructors and data/vector_store/poc_index.index -> IndexFlatL2 / METRIC_L2 -> NEEDS_PAPER_CORRECTION
- Sidecar provenance claim -> scripts/create_sidecar.py -> data/mocks/temporal_sidecar.json -> NEEDS_METHOD_CLARIFICATION

## Sensitive handling

Detailed local results, data, and historical reports that contain or may contain query, patient, retrieval, note, or chunk material are LOCAL_RESTRICTED_DO_NOT_PUBLISH. This map contains only aggregate and lineage evidence.

