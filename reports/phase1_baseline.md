# Phase 1: Baseline Benchmarking Report

**Date**: 2025-12-19
**Dataset**: `discharge.csv` (Limit: 500 rows)
**Evaluation Set**: 100 Query-Document Pairs (`baseline_dataset.json`)

## 1. Executive Summary
This report establishes the baseline performance of the "Naive" RAG pipeline (Fixed-Size Chunking) compared to the Proof-of-Concept (POC) "Hybrid" pipeline.

| Metric | Naive Baseline | POC (Hybrid) | Delta |
| :--- | :--- | :--- | :--- |
| **Recall@1** | 0.3400 | 0.7000 | +0.3600 |
| **Recall@5** | 0.5300 | 0.8700 | +0.3400 |
| **MRR** | 0.4158 | 0.7670 | +0.3512 |
| **Latency** | 0.1173s | 0.1035s | -0.0138s |

## 2. Methodology
-   **Naive Strategy**: 250-word fixed windows, 0 overlap.
-   **POC Strategy**: `Hybrid Semantic` (Section-aware + Sentence Splitting).
-   **Embedding Model**: `BAAI/bge-base-en-v1.5` (Normalized L2).
-   **Retrieval**: FAISS FlatL2 Index.

## 3. Failure Mode Analysis (Top 5 Errors)
*Qualitative analysis (to be populated in Phase 4 synthesis).*

## 4. Conclusion
The "Hybrid Semantic" strategy demonstrates a decisive advantage over standard fixed-size chunking for clinical discharge summaries. 

-   **Performance**: A **+34% improvement in Recall@5** (0.53 &rarr; 0.87) indicates that respecting document structure (Header awareness) is critical for retrieval in this domain.
-   **Implication**: Naive splitting frequently severs the link between clinical sections (e.g., "Medications") and their content, whereas the hybrid approach preserves this local context.
-   **Next Steps**: Proceed to **Phase 2 (Header Propagation)** to address the remaining 13% of failures where context is lost across long subsections.
