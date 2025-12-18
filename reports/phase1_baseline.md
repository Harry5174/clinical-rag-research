# Phase 1: Baseline Benchmarking Report

**Date**: [DATE]
**Dataset**: `discharge.csv` (Limit: 500 rows)
**Evaluation Set**: 100 Query-Document Pairs (`baseline_dataset.json`)

## 1. Executive Summary
This report establishes the baseline performance of the "Naive" RAG pipeline (Fixed-Size Chunking) compared to the Proof-of-Concept (POC) "Hybrid" pipeline.

| Metric | Naive Baseline | POC (Hybrid) | Delta |
| :--- | :--- | :--- | :--- |
| **Recall@1** | [VAL] | [VAL] | [VAL] |
| **Recall@5** | [VAL] | [VAL] | [VAL] |
| **MRR** | [VAL] | [VAL] | [VAL] |
| **Latency** | [VAL]s | [VAL]s | [VAL] |

## 2. Methodology
-   **Naive Strategy**: 250-word fixed windows, 0 overlap.
-   **POC Strategy**: `Hybrid Semantic` (Section-aware + Sentence Splitting).
-   **Embedding Model**: `BAAI/bge-base-en-v1.5` (Normalized L2).
-   **Retrieval**: FAISS FlatL2 Index.

## 3. Failure Mode Analysis (Top 5 Errors)
*Qualitative analysis of queries where POC succeeded but Baseline failed.*

1.  **Query**: "[QUERY_TEXT]"
    -   *Target Note*: `[NOTE_ID]`
    -   *Baseline Result*: `[WRONG_NOTE_ID]` - [REASON]
    -   *POC Result*: Correctly retrieved rank [N].

2.  **Query**: ...

## 4. Conclusion
[To be written after results...]
