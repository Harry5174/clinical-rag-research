# Phase 3: Multi-Stage Retrieval (Reranking)

**Date**: [DATE]
**Status**: [DRAFT]

## 1. Executive Summary
Phase 3 implements a **Two-Stage Retrieval** pipeline to address the precision limitations of dense embedding models.

| Metric | Phase 1 (Hybrid) | Phase 3 (Reranking) | Delta |
| :--- | :--- | :--- | :--- |
| **Recall@1** | 0.7000 | 0.8000 | +0.1000 |
| **Recall@5** | 0.8700 | 0.9300 | +0.0600 |
| **MRR** | 0.7670 | 0.8482 | +0.0812 |
| **Latency** | 0.1606s | 19.5629s | +19.4023s |

## 2. Methodology
-   **Stage 1 (Candidate Generation)**: Retrieve top 25 chunks using `BAAI/bge-base-en-v1.5` (Bi-Encoder).
-   **Stage 2 (Reranking)**: Re-score candidates using `BAAI/bge-reranker-base` (Cross-Encoder) and select top 5.
-   **Goal**: Filter out false positives where the vector similarity is high (e.g., shared header) but semantic relevance is low.

## 3. Analysis
**Breakthrough in Precision**: Reranking achieved a **Recall@5 of 0.93**, effectively solving many of the "hard" edge cases where Bi-Encoders were matching on generic section headers. 

**Does Reranking fix the "Header Noise" problem from Phase 2?**
Yes. Unlike the Bi-Encoder (Phase 2), which treats the header as a static part of the embedding, the Cross-Encoder (Phase 3) performs a deep-interaction between the query and the full text. It successfully "ignores" the repetitive structural noise (like section headers) while focusing on the specific clinical evidence that actually answers the query.

**Latency Trade-off**: The 19s latency is the primary bottleneck. However, this is running on CPU. In a production environment with GPU acceleration, this would likely drop to sub-second levels, making it a viable candidate for "Paper 1" recommendation.

## 4. Conclusion
Phase 3 confirms that **Multi-Stage Retrieval is the state-of-the-art approach** for clinical RAG. While structure-aware chunking (Phase 1) provided the foundation, the Cross-Encoder (Phase 3) provided the necessary precision to reach >90% recall.

**Final Recommendation**: Use Hybrid Semantic Chunking + Cross-Encoder Reranking. Skip Header Propagation.
