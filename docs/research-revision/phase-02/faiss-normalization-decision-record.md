# FAISS and Normalization Decision Record — Sprint 2.2

## 1. Decision

The paper must describe the inspected implementation as FAISS `IndexFlatL2` / `METRIC_L2` retrieval over normalized embeddings. It must not claim `IndexFlatIP` unless future implementation evidence changes.

The paper may mention cosine-equivalent **ranking** only with explicit unit-normalization conditions for both query and document vectors. This does not change the implemented FAISS metric and does not validate the evaluator's downstream L2-score transform.

## 2. Evidence

- The research and baseline index constructors call `SentenceTransformer.encode(..., normalize_embeddings=True)` for documents and construct `faiss.IndexFlatL2`.
- The base retriever and Phase 9 evaluator call `encode(..., normalize_embeddings=True)` for queries before FAISS search.
- The base retriever identifies FAISS return values as L2 distances, with lower values preferred.
- The Phase 01 authoritative manifest records the inspected current index as `IndexFlatL2`, `METRIC_L2`, and 768-dimensional.
- No `IndexFlatIP` or `METRIC_INNER_PRODUCT` construction was found in the approved source, script, and test search scope.

The exact build-run provenance of the root vector-store index remains incomplete, but it does not contradict the index metric or normalization evidence above.

## 3. What the paper must not say

- The system uses FAISS `IndexFlatIP`.
- The system performs exact inner-product FAISS retrieval.
- Raw FAISS L2 values are cosine similarities or inner-product scores.
- The current evaluator converts L2 distance to cosine similarity.
- The evaluated Phase 9 score treatment is validated or calibrated.

## 4. What the paper may safely say

- The inspected implementation uses BGE-base-en-v1.5 embeddings with FAISS `IndexFlatL2` / `METRIC_L2` retrieval.
- The inspected document-construction and query-encoding paths request normalized embeddings.
- The base retriever receives L2 distance outputs, with lower distances preferred.
- The reported Phase 9 path derives a downstream evaluator value from L2 output and combines it with temporal decay; its mathematical interpretation requires separate review.

## 5. Mathematical note, if applicable

For unit-normalized vectors `x` and `y`:

`||x - y||² = 2 - 2(x · y)`.

Thus, lower squared L2 distance and higher dot product/cosine similarity induce the same ranking only for unit-normalized query and document vectors. This statement concerns ordering, not index identity, raw score values, calibration, or TIMER score treatment.

## 6. Conditions required for cosine-equivalent wording

All conditions must be stated or verified:

1. Document embeddings are unit normalized before index insertion.
2. Query embeddings are unit normalized before search.
3. The wording concerns ranking/order, not the raw FAISS numeric values.
4. The wording preserves the actual index type: `IndexFlatL2` / `METRIC_L2`.
5. The wording does not characterize the Phase 9 `clip((D + 1) / 2)` value as cosine similarity or a validated conversion.

## 7. Remaining uncertainty

The Phase 9 evaluator receives L2 distances but applies an increasing affine transform and then sorts the composite TIMER score descending. The transform is labeled as inner-product-style in source comments, yet it does not invert distance. Its semantics, ranking effect after temporal combination, and suitability for publication description are unresolved.

The exact build invocation for the current root index is also not durably bound to the inspected builder, although the current index metric and normalization facts are independently recorded in the sanitized Phase 01 evidence.

## 8. Future verification, if needed

Do not rerun or modify anything under this sprint. A separately approved methodology/evaluation task should:

1. establish the exact current-index build lineage;
2. test and document a mathematically valid treatment of L2 output for semantic scoring;
3. distinguish candidate-generation ordering from downstream TIMER ordering; and
4. decide whether optional cross-encoder scores require a separate scale policy.
