# Score-Treatment Decision Record — Sprint 2.3

## 1. Decision

Do not describe the current Phase 9 transformed L2 value as cosine similarity, inner-product similarity, validated semantic similarity, or a correct L2-to-similarity conversion. Its effect requires a supervisor decision and separately approved analysis or code-correction/reproduction work.

## 2. Evidence

- The current index uses `IndexFlatL2` / `METRIC_L2`; lower distance is better.
- Phase 9 computes `clip((D + 1) / 2, 0, 1)` from FAISS output `D`.
- That value becomes the semantic input to `alpha * S + beta * decay`.
- Candidates are sorted by final score descending.
- Controlled hard-negative evaluation uses supplied fixture scores, not this transform.

## 3. Why current Phase 9 score treatment is risky

The transform is nondecreasing in L2 distance. Before clipping, farther candidates receive larger values; after clipping, values at or above the threshold collapse to one. Descending final-score sorting therefore does not preserve lower-is-better L2 semantics for the semantic term.

This does not itself invalidate reproduced aggregates, but it blocks a semantic-similarity interpretation and may affect the ablation ranking.

## 4. What the paper must not say

- Phase 9 converts L2 distance to cosine or inner-product similarity.
- The semantic value is validated, calibrated, or direction-preserving.
- Phase 9 semantic-only is a standard cosine/dense-retrieval baseline.
- Phase 9 differences isolate temporal effects independently of score treatment.

## 5. What the paper may safely say

- Phase 9 is a reproduced local target-subject-filtered ablation protocol.
- The evaluator derives a downstream value from L2 output and combines it with temporal decay.
- Score treatment remains a documented methodological limitation.
- Aggregates may be reported only with protocol and score-treatment caveats.

## 6. Options for future resolution

- **Option A:** Keep code/results and retain the limitation.
- **Option B:** Run an approved analysis sprint to quantify transform/ranking impact.
- **Option C:** If direction is judged incorrect, run an approved code-correction/reproduction sprint.
- **Option D:** Avoid detailed score-scale interpretation and report only protocol-bound aggregates with the limitation.

## 7. Recommended supervisor decision

Adopt Option D for future paper-editing scope and Option B before relying on Phase 9 as evidence about semantic-score behavior or temporal-effect magnitude. Reserve Option C until analysis establishes material need. This sprint implements no option.
