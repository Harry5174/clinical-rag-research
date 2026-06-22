# Retrieval and Indexing Correction Specification — Sprint 2.2

## 1. Executive summary

The current inspected implementation uses `BAAI/bge-base-en-v1.5` embeddings, normalized embedding vectors, and FAISS `IndexFlatL2` retrieval. The paper's literal `IndexFlatIP`, exact-inner-product, and unqualified cosine-score descriptions therefore require correction. The reported Phase 9 path searches the complete local index, filters results to the expected target subject, retains the first 50 filtered chunk candidates, applies TIMER scoring, deduplicates note identifiers, and reports top-1 and top-5 membership.

Unit-normalized vectors make squared-L2 ranking mathematically equivalent to dot-product/cosine ranking, but only when both query and document vectors are unit normalized. This is a ranking observation, not evidence that FAISS uses `IndexFlatIP`, and it does not validate the evaluator's current L2-score transform. That transform is a separate `REQUIRES_FUTURE_ANALYSIS` issue.

This is a correction specification only. It does not revise the paper or alter code, data, indexes, results, or evaluation outputs.

## 2. Current paper claims

Sanitized claim mapping for the frozen source:

| Location | Current claim summary | Required disposition |
|---|---|---|
| Methodology, retrieval description (line 223) | BGE embeddings are searched in an inner-product FAISS index and yield 50 candidates. | Correct the metric/index description; qualify the evaluated top-50 flow. |
| Problem formulation, semantic score (lines 241–250) | The semantic term is presented as cosine similarity bounded in a similarity range. | Do not equate returned FAISS L2 distances with a cosine score. Use conditional ranking-equivalence wording only if needed. |
| Candidate-retrieval summary (line 344) | Top-50 retrieval is described as exact inner-product search. | Replace with L2 retrieval over normalized embeddings. |
| Implementation table (lines 442–444) | The vector index is identified as `IndexFlatIP`; 50 to 5 is presented as a general pipeline. | Correct to `IndexFlatL2` / `METRIC_L2`; label 50-to-5 as the target-subject-filtered Phase 9 flow. |
| End-to-end description (lines 533–536) | The setting is characterized as full-index and patient-isolated. | Explicitly state full-index search, target-subject filtering, post-filter top-50 selection, and protocol-local scope. |

## 3. Implementation evidence

| Finding | Evidence |
|---|---|
| Embedding model | `prepare_header_prop_index.py`, `base.py`, `timer.py`, and `run_end_to_end_eval.py` select `BAAI/bge-base-en-v1.5`. |
| Document normalization | The inspected index constructors call `SentenceTransformer.encode(..., normalize_embeddings=True)` before adding vectors. Phase 01 records normalized embeddings for the inspected current index. |
| Query normalization | `base.py` and the reproduced Phase 9 evaluator call `encode(..., normalize_embeddings=True)` before FAISS search. |
| Index type | Inspected constructors instantiate `faiss.IndexFlatL2`; Phase 01 records the current inspected index as `IndexFlatL2`, `METRIC_L2`, dimension 768. |
| Base retrieval output | `base.py` explicitly identifies FAISS return values as L2 distances, where lower is better. |
| Phase 9 candidate flow | `run_end_to_end_eval.py` searches all metadata-indexed vectors, filters to the expected target subject, then retains 50 filtered candidates. |
| Final reporting | The evaluator applies TIMER ablation scoring, sorts descending by final score, deduplicates note IDs, and computes top-1 plus top-5 membership. |

The exact historical build run that produced the current root vector-store index is not durably bound to one constructor script. That lineage uncertainty does not change the inspected current index metric or the Phase 01 normalization evidence; it remains a separate provenance limitation.

## 4. Embedding model truth

The implementation model is `BAAI/bge-base-en-v1.5`. The paper's BGE-base-en-v1.5 model name is consistent with the inspected code and the 768-dimensional index record. A future paper edit may retain this as a local implementation fact, without implying external validation, clinical optimization, or model-comparison evidence.

## 5. Document normalization truth

The inspected baseline and research index-construction paths encode documents with `normalize_embeddings=True` before their vectors are added to `IndexFlatL2`. Phase 01's sanitized manifest additionally records normalized embeddings for the inspected current index. Safe future wording is: “document embeddings were normalized before L2 indexing,” subject to retaining the existing local-index lineage limitation.

## 6. Query normalization truth

The base retriever encodes its query vector with `normalize_embeddings=True`. The reproduced Phase 9 evaluator independently does the same before full-index search. This is the same SentenceTransformer normalization option used in the inspected document-construction paths. No claim should imply that normalization itself converts the FAISS metric from L2 to inner product.

## 7. FAISS index and metric truth

The relevant constructors use `faiss.IndexFlatL2`, and the Phase 01 manifest identifies the inspected current artifact as `IndexFlatL2` with `METRIC_L2`. A source search found no `IndexFlatIP` or `METRIC_INNER_PRODUCT` construction in the approved source, scripts, and tests scope. `IndexFlatIP` appears in the paper, not as an inspected implementation choice.

Future paper edits must describe the actual implementation as exact L2 nearest-neighbor search over the indexed vectors. They must not describe it as `IndexFlatIP`, an inner-product FAISS index, or exact-inner-product FAISS retrieval.

## 8. Cosine / inner-product / L2 interpretation

The actual FAISS metric is L2. The following mathematical ranking relationship is available only under the confirmed normalization condition:

For unit-normalized vectors `x` and `y`:

`||x - y||² = 2 - 2(x · y)`.

Therefore, decreasing squared L2 distance has the same ordering as increasing dot product/cosine similarity only when both the indexed document vector and the query vector are unit normalized.

This permits carefully qualified wording such as: “With unit-normalized query and document embeddings, L2-nearest-neighbor ordering is cosine-equivalent.” It does **not** permit any of the following:

- calling the implemented FAISS index `IndexFlatIP`;
- calling raw FAISS L2 outputs cosine similarities or inner-product scores;
- claiming that the evaluator's downstream numeric score is a validated cosine score; or
- omitting the unit-normalization condition.

## 9. Candidate retrieval flow

Two source-level paths must remain distinct in a future paper edit:

1. The reusable retriever performs BGE query encoding, L2 FAISS search, and can feed an optional cross-encoder reranker through `TwoStageRetriever`.
2. The reproduced Phase 9 evaluator does not invoke that cross-encoder path. It encodes a query, searches the full local index, filters returned entries to the target subject, and only then retains the first 50 filtered candidates.

Accordingly, a paper claim about evaluated Phase 9 behavior must not present top-50 as an unfiltered global first-stage set or assert cross-encoder participation. The target-subject filter is a material local protocol condition, not a generic deployment retrieval procedure.

## 10. Top-50 to final top-5 flow

For the reproduced Phase 9 evaluator:

1. Search all local indexed entries using FAISS L2 search.
2. Preserve the returned FAISS order while filtering entries to the target subject.
3. Retain the first 50 entries remaining after that filter.
4. Attach the evaluator's semantic value and temporal decay to each retained chunk candidate.
5. Compute each ablation's TIMER final score and sort candidates descending by that final score.
6. Deduplicate note identifiers while preserving the final-score order.
7. Use the first deduplicated note for Accuracy@1 and the first five for Recall@5 membership.

The “50 to 5” description is therefore accurate only when explicitly tied to this local target-subject-filtered Phase 9 protocol. It is not evidence of a general unfiltered top-50 production pipeline.

## 11. Score-treatment interpretation

The code has two distinct score conventions:

- Base retrieval labels FAISS output as L2 distance, where lower is better.
- The Phase 9 evaluator applies `clip((D + 1) / 2)` to that L2 output, labels it as an inner-product-style normalization, and passes the result into TIMER scoring. This transform is increasing in `D`; it neither inverts L2 distance nor performs a demonstrated L2-to-cosine conversion.

TIMER then computes a final score as an alpha-weighted semantic term plus an intent-selected temporal-decay term, and sorts that final score descending. Because the semantic transform is incompatible with the stated lower-is-better L2 interpretation, the semantic-score meaning and its interaction with TIMER are `REQUIRES_FUTURE_ANALYSIS`.

Until that analysis is complete, safe wording is limited to: the evaluator consumes a transformed value derived from FAISS L2 output and combines it with temporal decay for its local ablation ranking. It is unsafe to call that value a cosine similarity, inner-product score, correctly normalized semantic similarity, or validated distance-to-similarity conversion.

## 12. Required paper-correction instructions

For a future paper-editing sprint:

1. Replace every literal `IndexFlatIP` and exact-inner-product FAISS claim with `IndexFlatL2` / `METRIC_L2` terminology.
2. State that document and query embeddings are normalized in the inspected implementation, while keeping the current local-index lineage limitation where relevant.
3. If ranking equivalence is useful, state it conditionally: unit-normalized query and document vectors make L2 ordering cosine-equivalent. Do not identify returned distances as cosine scores.
4. Replace unqualified “top-50 semantic candidates” language for Phase 9 with: full local-index search, target-subject filtering, then retention of the first 50 filtered entries.
5. State that the evaluated Phase 9 code applies TIMER scoring directly; do not state or imply cross-encoder reranking for those reported results.
6. Define final top-5 as deduplicated note identifiers after TIMER final-score ordering, with Recall@5 measured by expected-note membership in that set.
7. Do not explain the evaluator's transformed L2 value as a valid inner-product or cosine score until future methodology review resolves it.

## 13. Unsafe claims to avoid

- “FAISS `IndexFlatIP`” or “exact inner-product FAISS search.”
- “FAISS returns cosine similarity” or “the returned L2 distance is an inner-product score.”
- Unqualified “cosine retrieval,” without both unit-normalization conditions and an explicit distinction from the L2 metric.
- “The first 50 global candidates are reranked” for the Phase 9 reported path.
- “The cross-encoder produced the reported Phase 9 results.”
- “Unconstrained” or deployment-like characterization of the target-subject-filtered local protocol.
- Any assertion that the L2-to-`S_sem` transform is validated, mathematically correct, or score-calibrated.

## 14. Open questions

| Question | Status |
|---|---|
| Which exact historical build invocation created the current root vector-store index? | `REQUIRES_FUTURE_ANALYSIS` — source capabilities and manifest facts are known, but a durable build-run binding is incomplete. |
| Is `clip((D + 1) / 2)` appropriate for L2 output and coherent with descending TIMER final-score sorting? | `REQUIRES_FUTURE_ANALYSIS`. |
| What semantic-score scaling is justified when the optional cross-encoder path is used? | `REQUIRES_FUTURE_ANALYSIS`; the scorer itself notes scale assumptions. |
| Is target-subject filtering the intended final evaluation protocol and manuscript framing? | `REQUIRES_SUPERVISOR_DECISION`. |

## 15. Future sprint dependencies

- **Sprint 2.3 / methodology review:** resolve L2-distance-to-semantic-score treatment, its ranking direction, and exact index-build provenance.
- **Future paper-editing sprint:** apply only the correction instructions above after supervisor review; no paper wording is drafted or changed here.
- **Future evaluation-protocol sprint:** decide whether target-subject filtering remains the intended reported protocol and how it should be justified.
- **Future analysis sprint:** assess score scaling, cross-encoder participation, and any claim beyond the current local, protocol-bound evidence.
