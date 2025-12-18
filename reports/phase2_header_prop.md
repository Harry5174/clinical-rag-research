# Phase 2: Technical Innovation (Header Propagation)

**Date**: [DATE]
**Status**: [DRAFT/FINAL]

## 1. Executive Summary
Phase 2 introduces "Header Propagation" to solve the context-loss problem in long clinical sections.

| Metric | Naive (Baseline) | Phase 1 (Hybrid) | Phase 2 (HeaderProp) | vs Phase 1 |
| :--- | :--- | :--- | :--- | :--- |
| **Recall@1** | 0.3400 | 0.7000 | 0.7000 | 0.0000 |
| **Recall@5** | 0.5300 | 0.8700 | 0.8500 | -0.0200 |
| **MRR** | 0.4158 | 0.7670 | 0.7590 | -0.0080 |

## 2. Hypothesis & Methodology
**Problem**: In Phase 1, we observed that chunks deep within a "Medications" or "Lab Results" section lose their semantic label because the header is far away in the document.
**Solution**: We inject the section header into *every* chunk.
-   *Original*: "Aspirin 81mg daily..."
-   *Propagated*: "[Medications] Aspirin 81mg daily..."

## 3. Qualitative Analysis (Success Cases)
*Queries where HeaderProp succeeded but Phase 1 failed.*

1.  **Query**: "[QUERY_TEXT]"
    -   *Phase 1 Result*: Rank [N] (Missed context)
    -   *Phase 2 Result*: Rank [1] (Correct)
    -   *Analysis*: [Explain why propagation helped]

## 4. Conclusion
## 4. Conclusion
**Result: Neutral/Slight Regression.**
Contrary to the hypothesis, blindly injecting section headers into every chunk did *not* improve retrieval performance (Recall@5 dropped from 0.87 to 0.85).

**Possible Reasons**:
1.  **Noise Injection**: Repetitive headers like "[History Of Present Illness]" might be diluting the unique signal in short chunks.
2.  **Model Sensitivity**: BGE-Base might be over-attending to the header rather than the content.
3.  **Sufficient Context**: The Phase 1 Hybrid strategy (which keeps natural section boundaries) might already be capturing enough context, making explicit injection redundant.

**Recommendation**:
Do not adopt Header Propagation as a default. Proceed to Phase 3 (Reranking) to solve the remaining hard cases.
