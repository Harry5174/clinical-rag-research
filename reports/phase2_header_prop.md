# Phase 2: Technical Innovation (Header Propagation)

**Date**: 2025-12-18
**Status**: COMPLETE

## 1. Executive Summary
Phase 2 evaluated "Header Propagation"—injecting section headers (e.g., `[Medications]`) into every text chunk—to strictly enforce context preservation.

| Metric | Naive (Baseline) | Phase 1 (Hybrid) | Phase 2 (HeaderProp) | vs Phase 1 |
| :--- | :--- | :--- | :--- | :--- |
| **Recall@1** | 0.3400 | 0.7000 | 0.7000 | 0.0000 |
| **Recall@5** | 0.5300 | 0.8700 | 0.8500 | -0.0200 |
| **MRR** | 0.4158 | 0.7670 | 0.7590 | -0.0080 |

**Result**: The inclusion of explicit headers caused a minor **regression (-2%)** in retrieval performance compared to the Phase 1 Hybrid Baseline.

## 2. Hypothesis & Methodology
**Motivation**: In Phase 1, qualitative errors showed that chunks deep within long lists (e.g., Lab Results) lost their semantic association with the section title.
**Method**: We implemented a `HeaderProp` strategy in `AdvancedChunker` which prepends the active section header to the text of every generated chunk during indexing.

## 3. Analysis of Regression
Qualitative analysis suggests that **Header Propagation introduced semantic noise**. 

-   **Dilution of Signal**: Short chunks dominated by a repetitive header (e.g., `[Hospital Course]`) tend to cluster together in the vector space based on the header rather than their unique content.
-   **Embedding Bias**: The embedding model (`BGE-Base`) likely over-weighted the explicit header tokens, reducing the similarity score for queries that targeted specific facts *within* those sections, unless the query also explicitly contained the header terms.

## 4. Conclusion
While "Structure-Awareness" (Phase 1) is critical, **explicit repetition (Phase 2) is detrimental** for dense embedding models. The natural boundaries preserved by the Hybrid strategy in Phase 1 offer a better signal-to-noise ratio.

**Implication for Research**: Future improvements should focus on **filtering noise** (Reranking) rather than adding more explicit context to the dense index.
