# Sprint 2.1 — Pipeline Truth Table and Methodology Gap Report

## 1. Sprint goal

Create a sanitized source-of-truth comparison of the frozen manuscript’s methodology statements against current implementation behavior and Phase 01 aggregate/provenance evidence. This sprint produces specifications for later review only; it does not revise the manuscript or run evaluations.

## 2. Files inspected

- Phase 01 evidence: the Sprint 1.1–1.5 records, authoritative-results manifest, paper-claim correction matrix, and reviewer-concern mapping under `docs/research-revision/phase-01/`.
- Frozen paper source: `latex_publication/v1/sn-article.tex`.
- Current bibliography context was intentionally not edited and was not needed for claim classification.
- Source and scripts: chunking, index construction, base/two-stage/TIMER retrieval, scoring, sidecar generation, hard-negative evaluation, and end-to-end evaluation files under `src/app/` and `scripts/`.
- Tests: `tests/test_scoring.py` and `tests/test_timer_integration.py` as implementation-mechanism context only.

Restricted reports, detailed result rows, raw data, index contents, and raw query/retrieval material were not inspected for this sprint’s documentation and are not reproduced here.

## 3. Commands run

- Required Git baseline: `pwd`, `git status --short`, `git branch --show-current`, `git remote -v`, and `git diff --check`.
- Approved filename inventories and `sed` reads of Phase 01 evidence documents.
- Approved paper section/claim searches; the supplied first combined section-regex form produced an `rg` escape parse error, so equivalent literal section patterns were rerun successfully.
- Approved source/script/test inventory, implementation-evidence search, and `sed` reads of the named implementation files.
- Directory creation limited to `docs/research-revision/phase-02/` and documentation-only file creation in that directory.

No evaluator, generator, indexer, data-access, networked, compilation, formatting, or test command was run.

## 4. Summary of paper methodology claims

The manuscript claims BGE-base-en-v1.5 embedding retrieval, FAISS inner-product top-50 candidate search, an optional reranker, regex-based Current/Historical/Trend routing, a confidence threshold, temporal sidecar metadata, TIMER constants including negative historical beta, a controlled hard-negative evaluation, and a patient-filtered Phase 9 ablation. It also asserts a separate tuning set and supplementary availability.

## 5. Summary of implementation evidence

Current relevant code uses normalized embeddings with `IndexFlatL2`; base retrieval interprets returned values as L2 distances. The current end-to-end evaluator searches the local index, filters candidates to the expected target subject, retains 50 candidates, and evaluates top-1/top-5 after TIMER ablation scoring. The sidecar generator uses generated offsets and a placeholder date. The router and temporal scorer are implemented, but router-performance, tuning, sensitivity, and broader-comparison evidence are absent.

## 6. Key pipeline truth findings

1. The inspected index metric is L2, not inner product.
2. Normalized embeddings are used, but their implementation must not be simplified into unsupported cosine/exact-inner-product statements.
3. Section-aware and header-propagation code exists; exact current-index build lineage requires confirmation.
4. The temporal sidecar fields exist but are generated/simulated rather than observed source provenance.
5. The 96.0% controlled aggregate and 58% / 80% filtered Phase 9 aggregate are authoritative only for their separately reproduced local protocols.
6. Target-subject filtering is a material condition of Phase 9 and must be explicit.
7. Tuning, supplementary availability, router quality, sensitivity, and broader baseline superiority are not supported by the inspected record.

## 7. Key methodology gaps

- `METHOD_CORRECTION_REQUIRED`: `IndexFlatIP` wording and source-derived-sidecar implications.
- `EVALUATION_PROTOCOL_CLARIFICATION_REQUIRED`: Phase 9 target-subject filtering, candidate flow, optional reranking, and index/chunking lineage.
- `TERMINOLOGY_REFRAMING_REQUIRED`: cosine/normalization wording and the narrow definition of “Virtual Graph.”
- `SCOPE_SOFTENING_REQUIRED`: controlled-result interpretation, router, constants, corpus scale, baseline comparison, and clinical/deployment implications.
- `REMOVE_OR_DEFER_CLAIM`: undocumented tuning and supplementary-material availability.

## 8. Risks

- Treating L2 outputs as inner-product scores without a dedicated methodology review could make the paper’s score explanation inaccurate.
- Releasing or claiming availability for local artifacts without a sanitization review could expose restricted material.
- Omitting target-subject filtering or fixed semantic-score conditions would materially overstate evaluation realism.
- An unqualified “Virtual Graph” label could overstate a generated metadata-sidecar mechanism.

## 9. Open questions

1. Which exact chunking/index-build path produced the current 1,206-vector evaluated index?
2. How should L2-distance outputs and the end-to-end score conversion be mathematically described after review?
3. Is target-subject filtering the intended final protocol, and what rationale should govern its manuscript description?
4. Should “Virtual Graph” be retained after narrow reframing?
5. What code, data, aggregate evidence, and documentation may be included in any future sanitized supplementary release?

## 10. Scope confirmation

- No paper or bibliography text was edited.
- No code, tests, notebooks, results, data, indexes, or `.gitignore` files were edited.
- No experiments, reproductions, metric generation, paper compilation, commits, pushes, or external services were used.
- No raw clinical-note text, MIMIC rows, patient-level content, raw queries, answers, retrieved chunks, or raw result rows are included in these deliverables.

## Deliverables

- [Pipeline truth table](pipeline-truth-table.md)
- [Methodology gap report](methodology-gap-report.md)
