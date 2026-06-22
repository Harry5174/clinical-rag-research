# Sprint 2.2 — Retrieval, Embedding, and FAISS Correction Specification

## 1. Sprint goal

Produce a sanitized, evidence-based correction specification for embedding model selection, normalization, FAISS metric, candidate retrieval, top-50/top-5 behavior, and score-treatment wording. This sprint is specification-only and does not edit the paper.

## 2. Files inspected

- Sanitized Phase 01 manifests, correction matrix, and exit-readiness record under `docs/research-revision/phase-01/`.
- Sprint 2.1 truth table, methodology gap report, and Sprint 2.1 report.
- `latex_publication/v1/sn-article.tex` for retrieval/indexing claim mapping only.
- `latex_publication/v1/sn-bibliography.bib` for citation-support context only.
- `src/app/research/indexing/prepare_header_prop_index.py`.
- `src/app/baseline/prepare_index.py`.
- `src/app/research/retrieval/base.py`, `two_stage.py`, `timer.py`, and `scoring.py`.
- `scripts/run_end_to_end_eval.py`, `scripts/run_hard_negative_eval.py`, and approved source/script/test inventories.

No raw data files, raw result files, index contents, or restricted reports were inspected by content.

## 3. Commands run

- Required Git safety baseline: `pwd`, `git status --short`, `git branch --show-current`, `git remote -v`, and `git diff --check`.
- Approved Phase 01/02 documentation inventory and read commands.
- Approved paper retrieval/indexing claim searches and bibliography context search.
- Approved source/script/test inventory and retrieval/indexing evidence searches.
- Read-only line-numbered inspection of the relevant index, retriever, scorer, evaluator, and truth-table files.

No experiments, reproduction runs, metric generation, paper compilation, test execution, data/index reads, network access, staging, commits, or pushes were performed.

## 4. Paper retrieval/indexing claims found

The frozen paper identifies BGE-base-en-v1.5, but also describes FAISS inner-product / `IndexFlatIP` retrieval, cosine-style semantic scores, top-50 candidate selection, an optional cross-encoder, and a 50-to-5 flow. Its Phase 9 wording acknowledges patient-level isolation but does not fully state that filtering occurs before retaining 50 candidates.

## 5. Implementation evidence found

- The inspected model configuration is `BAAI/bge-base-en-v1.5`.
- Inspected document-construction and query-encoding paths request normalized embeddings.
- Inspected constructors use `IndexFlatL2`, and Phase 01 records the current artifact as `IndexFlatL2` / `METRIC_L2`.
- The base retriever treats returned FAISS values as L2 distances, lower being better.
- Phase 9 searches all indexed entries, filters to the target subject, retains 50 filtered candidates, applies TIMER scoring, deduplicates note IDs, and evaluates top-1/top-5.
- The current Phase 9 L2-output transform is increasing in distance and is not a demonstrated conversion to similarity.

## 6. Final retrieval/indexing truth

The current inspected retrieval implementation is normalized-embedding L2 retrieval, not inner-product FAISS retrieval. Under unit normalization of both vectors, L2 order is cosine-equivalent, but this is a conditional mathematical ranking statement only. The Phase 9 evaluator's numeric semantic-score treatment remains unresolved and must not be represented as cosine or inner-product scoring.

## 7. Required corrections

1. Correct `IndexFlatIP` / exact-inner-product text to `IndexFlatL2` / `METRIC_L2`.
2. Describe document and query normalization accurately.
3. If needed, qualify cosine-equivalent language with unit normalization and restrict it to ranking order.
4. State the Phase 9 flow as full-index search, target-subject filter, then top-50 filtered candidates.
5. State that the evaluated Phase 9 path does not invoke the optional cross-encoder.
6. Define top-5 as deduplicated note IDs after TIMER final-score ordering.
7. Defer semantic-score-conversion explanation pending dedicated methodology review.

## 8. Unsafe claims

- `IndexFlatIP`, inner-product FAISS, or exact-inner-product search.
- Raw L2 distance as cosine similarity or an inner-product score.
- A general unfiltered top-50 pipeline for the reported Phase 9 result.
- Cross-encoder participation in the reported Phase 9 evaluation.
- An unconstrained/deployment-like characterization of target-subject-filtered evaluation.
- A validated L2-to-similarity conversion in the current evaluator.

## 9. Future dependencies

- Methodology review of the Phase 9 L2-distance transform, semantic-score direction, and TIMER combination.
- Provenance work that binds the current root index to an exact build invocation.
- Supervisor decision on the intended status and wording of target-subject filtering.
- A later approved paper-editing sprint to apply these instructions.

## 10. Scope confirmation

- No paper or bibliography file was edited.
- No code, notebooks, results, data, indexes, or configuration files were edited.
- No experiment, reproduction, metric generation, compilation, commit, push, or external service call occurred.
- This report and the two associated specifications contain no raw clinical-note, MIMIC-row, patient-level, raw-query, raw-answer, retrieved-chunk, or raw-result-row content.
