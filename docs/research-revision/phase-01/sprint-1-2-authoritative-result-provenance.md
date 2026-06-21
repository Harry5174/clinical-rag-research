# Sprint 1.2 — Authoritative Result Provenance and Empirical Claim Lockdown

## Decision summary

1. The hard-negative 96.0% and 90.5% headlines are both supported by existing aggregates, but neither is selected as the paper headline. Their n=200 output sets conflict and lack a durable command/input/configuration manifest. Status: CONFLICT_UNRESOLVED and NEEDS_SUPERVISOR_DECISION.
2. The filtered Phase 9 output supports the paper's 58% Accuracy@1 / 80% Recall@5 and semantic-baseline 22% / 69% values. Current source code writes this file after patient-specific filtering. Status: LIKELY_AUTHORITATIVE_BUT_NEEDS_REPRODUCTION.
3. The unfiltered Phase 9 output is SUPERSEDED.
4. Inspected indexes and index-construction code use IndexFlatL2 / METRIC_L2, not the paper's IndexFlatIP / exact-inner-product description. Status: NEEDS_PAPER_CORRECTION.
5. The sidecar generator creates randomized offsets and a placeholder date. Status: NEEDS_METHOD_CLARIFICATION.

This is a sanitized local-provenance record. It contains no raw clinical notes, patient rows, queries, answers, retrieved content, or note chunks.

## 1. Git state

The audit ran on branch main with remote origin https://github.com/Harry5174/clinical-rag-research.git.

| State | Path | Handling |
|---|---|---|
| modified | .gitignore | Pre-existing; untouched. |
| untracked | docs/ | Pre-existing directory; only the two approved Sprint 1.2 files were added within it. |

No staging, commit, push, reset, checkout, or destructive Git operation occurred.

## 2. Files inspected

- latex_publication/v1/sn-article.tex and latex_publication/v1/sn-bibliography.bib
- results/phase5/, results/phase5_v2/, and results/phase9/
- safe metadata from data/mocks/*.json, data/evaluation/*.json, and data/vector_store/**/*.index
- scripts/run_hard_negative_eval.py, scripts/run_end_to_end_eval.py, and scripts/create_sidecar.py
- src/app/baseline/prepare_index.py, src/app/research/indexing/prepare_header_prop_index.py, and src/app/research/retrieval/base.py
- relevant Git history and tracked-output metadata

Artifact-level size, hash, aggregate, sensitivity, and authority data is in authoritative-results-manifest.md.

## 3. Commands run

Read-only local audit commands included pwd; git status --short; git branch --show-current; git remote -v; git diff --check; targeted paper search; filename inventory; filename-only result search; safe local schema/aggregate/hash snippets; FAISS metadata inspection; script source inspection; sha256sum; git log; and a git diff of the end-to-end filtering change.

No evaluator, generator, indexer, formatter, auto-fixer, networked check, or experiment was run. The evaluator and end-to-end scripts were not executed because they write output artifacts.

## 4. Current paper empirical claims extracted from the TeX source

| Paper location | Claim/value | Type | Supporting file needed | Status |
|---|---|---|---|---|
| Abstract and results | n=200 hard-negative: TIMER 96.0% versus baseline 52.5%; inferential statistics and CI | controlled evaluation | Phase 5 aggregate plus input/config/run manifest | CONFLICT_UNRESOLVED |
| Method | FAISS inner-product / exact-inner-product candidate retrieval | implementation | inspected index metadata and construction code | NEEDS_PAPER_CORRECTION |
| Dataset/method | 165 patients, 1,206 chunks, IndexFlatIP, and a 1,206-entry sidecar | dataset/index/method | safe index metadata and sidecar provenance | mixed: index count supported; IndexFlatIP contradicted; sidecar needs clarification |
| Method | alpha/beta/lambda/tau settings and separate 20-query tuning set | tuning | constants plus tuning-run evidence | constants partly supported; tuning evidence UNSUPPORTED |
| End-to-end ablation | n=100: semantic-only 22% / 69%; full TIMER 58% / 80% | end-to-end evaluation | filtered Phase 9 result and current script | LIKELY_AUTHORITATIVE_BUT_NEEDS_REPRODUCTION |
| Conclusion | repeats empirical headlines and says supplementary files are provided | conclusion/availability | all above plus approved supplementary package | hard-negative unresolved; end-to-end needs reproduction; availability UNSUPPORTED |

## 5. Result inventory and safe aggregate summary

### Hard-negative result sets

| Result set | n | Baseline correct | TIMER correct | Baseline accuracy | TIMER accuracy | Classification |
|---|---:|---:|---:|---:|---:|---|
| results/phase5/ | 200 | 105 | 192 | 52.5% | 96.0% | CONFLICT_UNRESOLVED |
| results/phase5_v2/ | 200 | 105 | 181 | 52.5% | 90.5% | CONFLICT_UNRESOLVED |

The directories are identical for semantic collision, negation recency, and real-world mining. Terminology drift differs: Phase 5 records 12/20 TIMER-correct; Phase 5 v2 records 1/20. That changes the headline from 96.0% to 90.5%.

### Phase 9 end-to-end result sets

| Artifact | Semantic-only Accuracy@1 / Recall@5 | Full TIMER Accuracy@1 / Recall@5 | Classification |
|---|---|---|---|
| results/phase9/end_to_end_results.csv | 1% / 1% | 1% / 4% | SUPERSEDED |
| results/phase9/end_to_end_results_filtered.csv | 22% / 69% | 58% / 80% | LIKELY_AUTHORITATIVE_BUT_NEEDS_REPRODUCTION |

The filtered artifact also records intermediate aggregate ablations: uniform recency 38% / 66%, and intent-without-inversion 47% / 76%. These remain subject to the same reproduction gate.

### Input and sidecar summary

- The current non-v2 combined hard-negative input contains 40 total scenarios: 10 / 5 / 5 / 20.
- The v2 input contains 200 total scenarios: 50 / 30 / 20 / 100.
- The sidecar has 19 notes and 20 queries in its safe count summary.
- The local end-to-end index has 1,206 vectors of dimension 768.

## 6. Hard-negative provenance and conflict classification

run_hard_negative_eval.py produces per-scenario CSVs, a summary CSV, and a LaTeX table. It defaults to the non-v2 40-scenario input and Phase 5 output directory, but accepts dataset and result-directory arguments. A 200-query run is possible, yet no command, input hash, configuration file, or immutable run manifest ties either 200-query directory to a particular input.

The 96.0% table is Git-tracked in the 2026-03-25 Phase 5 commit. The 90.5% v2 table is a separate conflicting artifact with an older local modification time. That temporal evidence does not replace missing run provenance. The headline remains unresolved; do not retain 96.0% or substitute 90.5% until a supervisor decision is supported by a durable run record or future approved controlled reproduction.

## 7. Phase 9/end-to-end provenance and conflict classification

The current end-to-end script reads the local FAISS index and metadata, uses the v2 combined hard-negative input, filters candidates by target subject, and writes results/phase9/end_to_end_results_filtered.csv.

Git history shows the earlier script searched a limited candidate set and wrote end_to_end_results.csv. The tracked later change adds filtering and changes the output filename. The filtered file is the likely current artifact and supports all four paper values, but needs a captured environment, command, input hashes, and index hash before it is publication-authoritative.

## 8. FAISS metric verification

| Evidence | Finding |
|---|---|
| Paper | claims IndexFlatIP / exact inner-product search. |
| data/vector_store/poc_index.index | IndexFlatL2; d=768; ntotal=1,206; METRIC_L2. |
| Other inspected indexes | all IndexFlatL2 / METRIC_L2; d=768. |
| Index construction | both inspected constructors normalize embeddings and construct faiss.IndexFlatL2. |
| Retrieval code | describes returned values as L2 distance, lower is better. |
| End-to-end script | normalizes query embeddings but applies an inner-product-style score conversion. |

For unit-normalized vectors, L2 and cosine/inner-product rankings are monotonically related. That does not make the literal index/metric claim true, and it does not validate the current end-to-end score conversion. The paper's IndexFlatIP and exact-inner-product statements require correction.

## 9. Sidecar / temporal metadata finding

create_sidecar.py assigns randomized offsets by intent category and a placeholder note date. The sidecar must be described as generated/simulated temporal metadata, not observed clinical temporal provenance. Status: NEEDS_METHOD_CLARIFICATION.

## 10. Script-to-result provenance map

| Script / source | Inputs referenced | Outputs referenced | Audit conclusion |
|---|---|---|---|
| scripts/run_hard_negative_eval.py | combined hard-negative JSON; configurable path | Phase 5 scenario CSVs, summary CSV, LaTeX | Writes outputs; not run. Explains schema/statistics, not a specific n=200 run. |
| scripts/run_end_to_end_eval.py | local FAISS index, metadata, v2 combined input | filtered Phase 9 CSV | Writes output; not run. Supports filtered-file provenance. |
| scripts/create_sidecar.py | baseline evaluation JSON | temporal sidecar JSON | Writes output; not run. Establishes random/placeholder behavior. |
| src/app/baseline/prepare_index.py | local raw data | baseline index/metadata | Not run. Uses normalized embeddings plus IndexFlatL2. |
| src/app/research/indexing/prepare_header_prop_index.py | local raw data | research-v1 index/metadata | Not run. Uses normalized embeddings plus IndexFlatL2. |

## 11. Empirical claim status

| Claim | Status | Required next evidence |
|---|---|---|
| 96.0% hard-negative headline | CONFLICT_UNRESOLVED / NEEDS_SUPERVISOR_DECISION | immutable run record or future approved controlled reproduction |
| 90.5% hard-negative alternative | CONFLICT_UNRESOLVED / NEEDS_SUPERVISOR_DECISION | immutable run record or future approved controlled reproduction |
| 52.5% hard-negative baseline | present in both conflicting n=200 aggregates | hard-negative provenance resolution |
| Phase 9 58% / 80% and 22% / 69% | LIKELY_AUTHORITATIVE_BUT_NEEDS_REPRODUCTION | controlled rerun with command, environment, input/index hashes, and sanitized aggregate output |
| Phase 9 unfiltered metrics | SUPERSEDED | retain only as provenance history |
| IndexFlatIP claim | NEEDS_PAPER_CORRECTION | correct paper or verify a different index |
| Real temporal-sidecar claim | NEEDS_METHOD_CLARIFICATION | document simulation or generate validated provenance later |
| Separate 20-query tuning set | UNSUPPORTED | tuning data/configuration and held-out evidence |
| Supplementary-material availability | UNSUPPORTED | reviewed, sanitized, versioned package |

## 12. Claims safe to keep

- Controlled/simulated hard-negative artifacts exist in two n=200 aggregate result sets.
- The filtered Phase 9 artifact contains 58% / 80% and 22% / 69% aggregates, and current code writes it.
- The inspected end-to-end index has 1,206 768-dimensional vectors and uses IndexFlatL2 / METRIC_L2.
- Inspected index-construction code normalizes embeddings before creating an L2 index.
- Sidecar-generation code uses randomized offsets and a placeholder date.

## 13. Claims requiring paper correction

- Replace or qualify literal IndexFlatIP / exact-inner-product statements.
- Do not describe the generated sidecar as authentic observed temporal provenance.
- Remove or qualify the claim that supplementary materials are provided until a reviewed, sanitized package exists.

## 14. Claims requiring reproduction or supervisor decision

- Choose or reproduce the hard-negative headline: 96.0% versus 90.5%.
- Reproduce filtered Phase 9 under a captured environment and provenance manifest.
- Substantiate the claimed separate 20-query tuning set.
- Validate whether L2-distance score handling supports the claimed scoring scale.

## 15. Sensitive local artifact register

| Path | Status | Reason | Action |
|---|---|---|---|
| reports/novelty/phase5_quantitative_proof.md | LOCAL_RESTRICTED_DO_NOT_PUBLISH | patient-level/query-level content | keep local only; exclude from public docs/releases; consider future sanitized derivative |
| reports/publication/timer_graph_implementation_report.md | LOCAL_RESTRICTED_DO_NOT_PUBLISH | local clinical-example content | keep local only; exclude from public docs/releases; consider future sanitized derivative |
| detailed Phase 5, Phase 5 v2, and Phase 9 CSVs | LOCAL_RESTRICTED_DO_NOT_PUBLISH | query/retrieval fields | keep local only; publish aggregate-only derivatives only if approved |
| mock, evaluation, and vector-store content artifacts | LOCAL_RESTRICTED_DO_NOT_PUBLISH | may contain query, note, chunk, patient, or retrieval metadata | keep local only; no release without sanitization review |

## 16. Risks and open questions

1. Why does the 90.5% v2 terminology-drift output differ from the Phase 5 96.0% output while the other scenario outputs are byte-identical?
2. Which exact command, dataset hash, model version, index hash, and code revision produced each hard-negative aggregate table?
3. Does the Phase 9 filtered output use a model/index state consistent with the paper, and is its L2 score handling appropriate?
4. What evidence supports the claimed 20-query tuning set?
5. What sanitized supplementary package, if any, may legitimately be cited as available?

## 17. Scope confirmation

- No paper text, bibliography, code, notebook, data, result, or index was edited.
- No full experiment was rerun.
- No raw clinical-note text, raw MIMIC row, query, answer, or retrieved content is included here or in the manifest.
- No commit or push was made.
- Only this file and authoritative-results-manifest.md were created for Sprint 1.2.

