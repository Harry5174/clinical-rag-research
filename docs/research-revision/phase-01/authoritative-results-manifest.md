# Sprint 1.2 Authoritative Results Manifest

## Handling boundary

This sanitized manifest records provenance, hashes, schema, and aggregate findings only. It does not contain raw clinical notes, patient rows, queries, answers, retrieved content, or note chunks. Artifacts with such fields remain local-only.

## Result artifacts

| Path | Type; size; SHA-256 prefix | Safe summary | Likely producer | Category | Authority | Sensitivity | Notes |
|---|---|---|---|---|---|---|---|
| results/phase5/summary_table.csv | CSV; 657 B; c5e33a5c41a74099 | 5 rows; n=200; baseline 52.5%; TIMER 96.0% | run_hard_negative_eval.py with the v2 input | CONTROLLED / SIMULATED | AUTHORITATIVE_EXISTING | aggregate-only | Sprint 1.4 reproduced this exact hash in isolated /tmp output with current evaluator, current v2 input, and captured environment/input hashes. |
| results/phase5/semantic_collision_results.csv | CSV; 6,214 B; 6aa8156bd52557e2 | 50 rows; baseline 25/50; TIMER 50/50 | same | CONTROLLED / SIMULATED | CONFLICT_UNRESOLVED | LOCAL_RESTRICTED_DO_NOT_PUBLISH | Query/retrieval fields not reproduced. |
| results/phase5/negation_recency_results.csv | CSV; 4,483 B; 68bc1d6a49fbc797 | 30 rows; baseline 30/30; TIMER 30/30 | same | CONTROLLED / SIMULATED | CONFLICT_UNRESOLVED | LOCAL_RESTRICTED_DO_NOT_PUBLISH | Byte-identical to v2 counterpart. |
| results/phase5/terminology_drift_results.csv | CSV; 3,011 B; c7b512169226d98e | 20 rows; baseline 0/20; TIMER 12/20 | same | CONTROLLED / SIMULATED | CONFLICT_UNRESOLVED | LOCAL_RESTRICTED_DO_NOT_PUBLISH | Supplies the material v1/v2 difference. |
| results/phase5/real_world_mining_results.csv | CSV; 13,233 B; 89ed020d726460fd | 100 rows; baseline 50/100; TIMER 100/100 | same | CONTROLLED / SIMULATED | CONFLICT_UNRESOLVED | LOCAL_RESTRICTED_DO_NOT_PUBLISH | Byte-identical to v2 counterpart. |
| results/phase5/results_table.tex | TeX; 831 B; 2e2753b3bb2e0a78 | n=200; 52.5% / 96.0% | hard-negative evaluator | REPORT | AUTHORITATIVE_EXISTING | aggregate-only | Exact isolated Sprint 1.4 reproduction output hash; controlled/simulated interpretation remains required. |
| results/phase5_v2/summary_table.csv | CSV; 652 B; ae9ad80013c8cd5c | 5 rows; n=200; baseline 52.5%; TIMER 90.5% | historical hard-negative evaluator run | CONTROLLED / SIMULATED | SUPERSEDED | aggregate-only | Sprint 1.4 current evaluator plus current v2 input reproduced the exact Phase 5 96.0% output instead; original v2 run provenance remains historical. |
| results/phase5_v2/semantic_collision_results.csv | CSV; 6,214 B; 6aa8156bd52557e2 | 50 rows; baseline 25/50; TIMER 50/50 | same | CONTROLLED / SIMULATED | CONFLICT_UNRESOLVED | LOCAL_RESTRICTED_DO_NOT_PUBLISH | Same as Phase 5 counterpart. |
| results/phase5_v2/negation_recency_results.csv | CSV; 4,483 B; 68bc1d6a49fbc797 | 30 rows; baseline 30/30; TIMER 30/30 | same | CONTROLLED / SIMULATED | CONFLICT_UNRESOLVED | LOCAL_RESTRICTED_DO_NOT_PUBLISH | Same as Phase 5 counterpart. |
| results/phase5_v2/terminology_drift_results.csv | CSV; 2,957 B; 32bc347e7f98727d | 20 rows; baseline 0/20; TIMER 1/20 | same | CONTROLLED / SIMULATED | CONFLICT_UNRESOLVED | LOCAL_RESTRICTED_DO_NOT_PUBLISH | Conflicts with Phase 5 terminology-drift result. |
| results/phase5_v2/real_world_mining_results.csv | CSV; 13,233 B; 89ed020d726460fd | 100 rows; baseline 50/100; TIMER 100/100 | same | CONTROLLED / SIMULATED | CONFLICT_UNRESOLVED | LOCAL_RESTRICTED_DO_NOT_PUBLISH | Same as Phase 5 counterpart. |
| results/phase5_v2/results_table.tex | TeX; 833 B; 0b3e85914e1aa04c | n=200; 52.5% / 90.5% | hard-negative evaluator | REPORT | CONFLICT_UNRESOLVED | aggregate-only | Publication-shaped conflicting headline, not a resolution. |
| results/phase9/end_to_end_results_filtered.csv | CSV; 18,126 B; bcf806f3fb631012 | n=100; semantic-only 22% Accuracy@1 / 69% Recall@5; full TIMER 58% / 80% | current run_end_to_end_eval.py | END_TO_END | AUTHORITATIVE_EXISTING | LOCAL_RESTRICTED_DO_NOT_PUBLISH | Sprint 1.4 reproduced this exact hash using an isolated script copy with only the output path changed; current target-subject filtering was preserved. |
| results/phase9/end_to_end_results.csv | CSV; 19,500 B; 117e7e0eadadaf4c | n=100; semantic-only 1% / 1%; full TIMER 1% / 4% | pre-filter end-to-end script | END_TO_END | SUPERSEDED | LOCAL_RESTRICTED_DO_NOT_PUBLISH | Later tracked change adds filtering and changes output filename. |

## Inputs, sidecar, and indexes

| Path | Type; size; SHA-256 prefix | Safe summary | Category | Authority | Sensitivity | Notes |
|---|---|---|---|---|---|---|
| data/mocks/combined_hard_negatives.json | JSON; 46,081 B; bcf9681fb3d977ab | scenario counts 10 / 5 / 5 / 20; total 40 | CONTROLLED / SIMULATED | NEEDS_REPRODUCTION | LOCAL_RESTRICTED_DO_NOT_PUBLISH | Current evaluator default; cannot alone reproduce either n=200 headline. |
| data/mocks/combined_hard_negatives_v2.json | JSON; 218,536 B; 7f933ca30c3c55c6 | scenario counts 50 / 30 / 20 / 100; total 200 | CONTROLLED / SIMULATED | NEEDS_REPRODUCTION | LOCAL_RESTRICTED_DO_NOT_PUBLISH | Numerically compatible with n=200 outputs; no durable run binding. |
| data/mocks/temporal_sidecar.json | JSON; 11,571 B; fc2845ab63602072 | keys include metadata version, notes, queries, reference date; counts 19 / 20 | SIDECAR | NEEDS_METHOD_CLARIFICATION | LOCAL_RESTRICTED_DO_NOT_PUBLISH | Generator uses random offsets and a placeholder note date. |
| data/evaluation/baseline_dataset.json | JSON; 105,040 B; 015d59ca9a90d9da | 100 records; schema includes original-text and query fields | CONTROLLED | LOCAL_RESTRICTED_DO_NOT_PUBLISH | LOCAL_RESTRICTED_DO_NOT_PUBLISH | Values were not copied. |
| data/evaluation/phase_1_baseline_results.json | JSON; 331 B; c4ffbdc8aeb1e560 | top-level keys results, timestamp | REPORT | UNSUPPORTED for current-paper claims | aggregate-only | Does not establish Phase 5/9 provenance. |
| data/evaluation/phase_2_results.json | JSON; 500 B; 9786798294edb27e | top-level keys results, timestamp | REPORT | UNSUPPORTED for current-paper claims | aggregate-only | Does not establish Phase 5/9 provenance. |
| data/evaluation/phase_3_results.json | JSON; 394 B; c7eb1a79ba8a5638 | top-level keys config, results | REPORT | UNSUPPORTED for current-paper claims | aggregate-only | Does not establish Phase 5/9 provenance. |
| data/vector_store/poc_index.index | FAISS; 3,704,877 B; 82925a757df845d0 | IndexFlatL2; d=768; ntotal=1,206; METRIC_L2 | INDEX | AUTHORITATIVE_EXISTING for inspected metric | derived local artifact | Read by current end-to-end evaluator; contradicts paper's IndexFlatIP claim. |
| data/vector_store/baseline/poc_index.index | FAISS; 8,267,997 B; a471ce2e8be9c9e | IndexFlatL2; d=768; ntotal=2,691; METRIC_L2 | INDEX | AUTHORITATIVE_EXISTING | derived local artifact | Constructor normalizes embeddings then creates IndexFlatL2. |
| data/vector_store/research_v1/poc_index.index | FAISS; 3,714,093 B; 9e9148aa12c81b9e | IndexFlatL2; d=768; ntotal=1,209; METRIC_L2 | INDEX | AUTHORITATIVE_EXISTING | derived local artifact | Constructor normalizes embeddings then creates IndexFlatL2. |

## Sensitive local artifact register

| Path | Status | Reason | Action |
|---|---|---|---|
| reports/novelty/phase5_quantitative_proof.md | LOCAL_RESTRICTED_DO_NOT_PUBLISH | patient-level/query-level content | Keep local only; exclude from public docs/releases; consider a future sanitized derivative. |
| reports/publication/timer_graph_implementation_report.md | LOCAL_RESTRICTED_DO_NOT_PUBLISH | local clinical-example content | Keep local only; exclude from public docs/releases; consider a future sanitized derivative. |
| results/phase5/*.csv, results/phase5_v2/*.csv, results/phase9/*.csv | LOCAL_RESTRICTED_DO_NOT_PUBLISH | schemas include query or retrieval fields | Keep local only; publish aggregate-only derivatives only if approved. |
| data/mocks/*.json, data/evaluation/baseline_dataset.json, data/vector_store/**/*.pkl, data/vector_store/**/*.json | LOCAL_RESTRICTED_DO_NOT_PUBLISH | contains or may contain query, note, chunk, patient, or retrieval metadata | Keep local only; do not release without a separate sanitization review. |

## Authority decision

1. The 96.0% hard-negative result is AUTHORITATIVE_EXISTING for the explicitly controlled/simulated current evaluator plus current v2 input: Sprint 1.4 reproduced every output hash in isolated /tmp output. The 90.5% v2 result is SUPERSEDED as a current result, though its original historical-run cause is not recoverable from ignored artifacts.
2. The filtered Phase 9 artifact is AUTHORITATIVE_EXISTING for the current target-subject-filtered local implementation: Sprint 1.4 reproduced its exact output hash with only the output path isolated. The unfiltered artifact is SUPERSEDED.
3. The inspected index metric is IndexFlatL2 / METRIC_L2, so literal paper claims of IndexFlatIP require NEEDS_PAPER_CORRECTION.
