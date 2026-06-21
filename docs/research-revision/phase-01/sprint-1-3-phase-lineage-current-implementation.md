# Sprint 1.3 — Phase Lineage and Current Implementation Verification

## 1. Purpose and decision summary

This sprint verifies historical versus current result lineage without rerunning experiments. The record is sanitized: no raw clinical notes, patient rows, queries, answers, retrieved content, or note chunks are included.

The key findings are:

1. Phase 5 v2 was the March 3 expansion/statistical-analysis generation. Its tracked LaTeX table reports 90.5% TIMER accuracy.
2. Phase 5 was a later March 25 result publication artifact. Its tracked LaTeX table reports 96.0%.
3. The detailed Phase 5 and Phase 5 v2 result files contain the same 200 query-ID set, scenario by scenario. The 96.0% versus 90.5% difference is therefore not explained by a changed result query set.
4. All 11 changed TIMER outcomes occur in terminology drift; baseline outcomes are unchanged. The hard-negative evaluator source and scoring behavior are unchanged between the relevant committed implementation versions.
5. The dataset/result artifacts are ignored and untracked, so no immutable input-to-output binding establishes why terminology-drift outcomes changed. Hard-negative authority remains CONFLICT_UNRESOLVED and NEEDS_REPRODUCTION.
6. Phase 9 filtered is the current implementation path: the March 26 tracked change added target-subject filtering, changed output from end_to_end_results.csv to end_to_end_results_filtered.csv, and supports the paper's 58% / 80% and 22% / 69% aggregates. It remains NEEDS_REPRODUCTION.

## 2. Git state

- Repository: /home/harry/Desktop/research/poc/clinical-rag-research
- Branch: main
- Remote: origin https://github.com/Harry5174/clinical-rag-research.git
- Sprint 1.3 starting status: clean.
- No staging, commits, pushes, resets, checkouts, history cleanup, or destructive Git commands were performed.

## 3. Files inspected

- Sanitized Phase 01 evidence documents, including Sprint 1.1 and Sprint 1.2 records.
- Result path, size, timestamp, and hash metadata under results/.
- Report path, size, timestamp, and hash metadata under reports/ only. No report content was searched.
- Data path, size, and timestamp metadata under data/ only.
- scripts/run_hard_negative_eval.py, scripts/run_end_to_end_eval.py, scripts/create_sidecar.py, and hard-negative preparation scripts.
- Retrieval, scoring, index-construction, and relevant test files.
- Git history and metadata for implementation and documentation paths.
- Paper source and bibliography only through prior sanitized Sprint 1.2 findings.

## 4. Commands run

Read-only commands included:

- pwd; git status --short; git branch --show-current; git remote -v; git diff --check
- find metadata inventories for results/, reports/, and data/
- sha256sum over result/report metadata paths
- targeted git log, git show --stat, git rev-parse, and historical source metadata inspection
- targeted rg and sed inspections limited to scripts, source, tests, Phase 01 docs, and paper source
- local sanitized CSV/JSON comparison scripts that excluded data/raw, data/processed, and data/vector_store
- final git diff --check and git status --short

No experiment, evaluator, generator, indexer, formatter, auto-fixer, network check, or data-writing script was run.

## 5. Git history findings

| Date | Commit | Lineage evidence |
|---|---|---|
| 2026-01-29 | 9b9473a | Introduced hard-negative mining, preparation, evaluator, and restricted historical reports. |
| 2026-03-03 | 68c6b2c | Expanded hard-negative scenario seed generation and added configurable generation arguments. |
| 2026-03-03 | d84f260 | Added statistical analysis, v2 combining/mining support, updated hard-negative evaluator, and tracked Phase 5 v2 LaTeX result table. |
| 2026-03-25 | 4b3ac68 | Added tracked Phase 5 LaTeX result table and initial Phase 9 evaluator. |
| 2026-03-26 | 1b6ab26 | Added target-subject filtering to the Phase 9 evaluator and changed its output filename to the filtered result. |

The current hard-negative evaluator blob is identical to the d84f260 version. A behavior-only comparison of TIMER scoring constants and methods before and after 4b3ac68 found no change. This rules out a committed evaluator/scoring behavior change as the direct explanation for the hard-negative split.

## 6. Phase 5 lineage findings

Phase 5 is a later March 25 controlled/simulated hard-negative output set. Its aggregate is baseline 105/200 (52.5%) and TIMER 192/200 (96.0%). The tracked results_table.tex aligns with this output.

The file timestamps are later than the v2 result timestamps, but the detailed CSVs and source input are ignored. The current default non-v2 input contains only 40 scenarios, so the present default invocation cannot reproduce this n=200 result without explicit input arguments.

Status: HISTORICAL result artifact with a later publication-facing table; current headline authority remains CONFLICT_UNRESOLVED and NEEDS_REPRODUCTION.

## 7. Phase 5 v2 lineage findings

Phase 5 v2 is the March 3 expansion/statistical-analysis output set. Its aggregate is baseline 105/200 (52.5%) and TIMER 181/200 (90.5%). The tracked v2 results_table.tex aligns with this output.

The current v2 combined input has 200 scenarios and predates both detailed output sets on local timestamps. However, it is ignored and untracked, and no run manifest records its hash, command, code revision, or environment for either output set.

Status: HISTORICAL result artifact; its relationship to the later 96.0% output is CONFLICT_UNRESOLVED.

## 8. Hard-negative query-set comparison

| Scenario | Result rows Phase 5 / v2 | Query-ID overlap | Baseline outcome changes | TIMER outcome changes |
|---|---:|---:|---:|---:|
| semantic collision | 50 / 50 | 50 / 50 | 0 | 0 |
| negation recency | 30 / 30 | 30 / 30 | 0 | 0 |
| terminology drift | 20 / 20 | 20 / 20 | 0 | 11 |
| real-world mining | 100 / 100 | 100 / 100 | 0 | 0 |

The compared result query-ID sets are identical across all 200 rows. The headline difference is an outcome change concentrated in terminology drift, not a result query-set expansion or replacement. This does not identify the cause: untracked input snapshots, query text, note pairs, expected labels, or a local runtime dependency could have differed. Those remain unproven.

The local non-v2 versus v2 combined-input metadata does show a historical dataset expansion from 40 to 200 scenarios. That input expansion is distinct from the Phase 5 versus Phase 5 v2 result-file comparison.

## 9. Phase 9 lineage findings

The initial March 25 Phase 9 script searched 50 candidates and wrote results/phase9/end_to_end_results.csv. Its aggregate result is 1% / 1% for semantic-only and 1% / 4% for full TIMER.

The March 26 current implementation maps expected note IDs to subject IDs, searches the complete index, filters candidates to the target subject, takes the first 50 filtered candidates, and writes results/phase9/end_to_end_results_filtered.csv. Its aggregate result is 22% / 69% for semantic-only and 58% / 80% for full TIMER.

Target-subject filtering is a material protocol change, not merely a filename change. It is appropriate only if the paper accurately states patient-level isolation and the target-subject mapping is justified. The filtered artifact is CURRENT implementation lineage but NEEDS_REPRODUCTION for paper authority. The unfiltered artifact is SUPERSEDED for the current claimed protocol.

## 10. Current implementation path

| Paper-relevant claim | Current implementation path | Current result artifact | Status |
|---|---|---|---|
| controlled hard-negative evaluation | run_hard_negative_eval.py with configurable input/output paths | no uniquely bound current artifact | CONFLICT_UNRESOLVED; NEEDS_REPRODUCTION |
| end-to-end n=100 ablation | run_end_to_end_eval.py using v2 input, local index, and target-subject filtering | results/phase9/end_to_end_results_filtered.csv | CURRENT path; NEEDS_REPRODUCTION |
| TIMER scoring | retrieval/scoring.py and retrieval/timer.py | both evaluation paths | CURRENT constants; controlled-vs-end-to-end interpretation still required |
| index construction | baseline/prepare_index.py and research/indexing/prepare_header_prop_index.py | local FAISS indexes | CURRENT inspected metric is IndexFlatL2 / METRIC_L2; NEEDS_PAPER_CORRECTION |
| sidecar generation | scripts/create_sidecar.py | data/mocks/temporal_sidecar.json | HISTORICAL generated metadata; NEEDS_METHOD_CLARIFICATION |
| paper source | latex_publication/v1/sn-article.tex | claims currently cite 96.0%, IndexFlatIP, and end-to-end values | No paper edit authorized in this sprint |

## 11. Script-to-output map

| Script | Intended role | Inputs | Outputs | Status |
|---|---|---|---|---|
| scripts/generate_hard_negatives.py | generate synthetic scenario seed data | configurable generation parameters | versioned hard-negative JSON | HISTORICAL preparation; writes files |
| scripts/mine_hard_negatives.py | mine local candidates | local data and configurable output | versioned mined-candidate JSON | HISTORICAL preparation; writes files |
| scripts/combine_hard_negatives.py | combine synthetic/mined sets | v2 component inputs | v2 combined JSON | HISTORICAL preparation; writes files |
| scripts/run_hard_negative_eval.py | controlled/simulated evaluator | configurable combined JSON | scenario CSVs, summary CSV, LaTeX | CURRENT evaluator source; result authority unresolved |
| scripts/run_end_to_end_eval.py | target-subject-filtered ablation | v2 input, local index, local metadata | filtered Phase 9 CSV | CURRENT implementation path; needs reproduction |
| scripts/create_sidecar.py | generated temporal metadata | baseline evaluation JSON | temporal-sidecar JSON | HISTORICAL generated artifact; method clarification needed |

## 12. Result artifact classification

| Artifact or path | Classification | Reason |
|---|---|---|
| results/phase5/results_table.tex and aggregate lineage | HISTORICAL; CONFLICT_UNRESOLVED | Later 96.0% output with no immutable input/run binding. |
| results/phase5_v2/results_table.tex and aggregate lineage | HISTORICAL; CONFLICT_UNRESOLVED | Earlier 90.5% output with no immutable input/run binding. |
| results/phase9/end_to_end_results.csv | SUPERSEDED | Pre-filter protocol and output path. |
| results/phase9/end_to_end_results_filtered.csv | CURRENT; NEEDS_REPRODUCTION | Current code produces this filtered protocol output. |
| data/mocks/combined_hard_negatives.json | HISTORICAL | Current default has 40 scenarios; cannot reproduce n=200 outputs alone. |
| data/mocks/combined_hard_negatives_v2.json | CURRENT reproduction candidate; NEEDS_REPRODUCTION | Current end-to-end code references it; ignored/untracked. |
| data/vector_store/poc_index.index | CURRENT inspected index; NEEDS_PAPER_CORRECTION | IndexFlatL2 / METRIC_L2 conflicts with paper's IndexFlatIP wording. |
| data/mocks/temporal_sidecar.json | HISTORICAL; NEEDS_METHOD_CLARIFICATION | Generator uses random offsets and placeholder date. |
| restricted historical reports and detailed local result/data files | LOCAL_RESTRICTED_DO_NOT_PUBLISH | Contain or may contain local patient, query, retrieval, or note-level material. |

## 13. Claims that can be corrected now

The following are lineage or implementation corrections, not new empirical claims:

- The current inspected index is IndexFlatL2 / METRIC_L2, not IndexFlatIP.
- The temporal sidecar is generated with randomized offsets and a placeholder date; it must not be described as directly observed clinical temporal metadata.
- The Phase 9 unfiltered output is not the current implementation result and should not be used for the current patient-filtered protocol.
- Supplementary-material availability remains unsupported until a reviewed, sanitized package exists.

## 14. Claims blocked pending reproduction or supervisor decision

- Whether 96.0% or 90.5% is the authoritative hard-negative headline.
- Any causal explanation for the 11 terminology-drift TIMER outcome changes.
- Publication authority of the filtered Phase 9 58% / 80% and 22% / 69% result.
- The claimed separate tuning-set evidence and held-fixed hyperparameter protocol.
- Whether target-subject filtering is the intended and paper-appropriate evaluation protocol.

## 15. Risks and open questions

1. No manifest binds the v2 input hash, evaluator command, scoring environment, and result hashes.
2. Ignored local result/data artifacts make Git history insufficient for full empirical provenance.
3. The hard-negative evaluator emits sensitive debug output and writes detailed local result fields; future runs need a restricted logging plan.
4. The end-to-end evaluator has a fixed result path and would overwrite the current filtered output.
5. The current end-to-end scoring code treats L2 outputs with an inner-product-style conversion, requiring methodological review.

## 16. Scope confirmation

- No paper, bibliography, code, notebook, result, data, or index artifact was edited.
- No full experiment was run.
- No raw clinical-note text, raw MIMIC row, raw query, answer, or retrieved content is included.
- No commit or push was made.

