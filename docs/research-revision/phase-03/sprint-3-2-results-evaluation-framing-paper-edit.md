# Sprint 3.2 — Results and Evaluation Framing Paper Edit

## 1. Sprint Goal

Edit only approved Results / Evaluation-facing manuscript content so reported aggregates are framed according to the approved Phase 01, Phase 02, and Phase 03 evidence. The sprint preserves all numeric results, tables, figures, bibliography, code, tests, notebooks, data, indexes, and raw local artifacts.

## 2. Files Inspected

- `docs/research-revision/phase-01/authoritative-results-manifest.md`
- `docs/research-revision/phase-01/reproduction-run-manifest.md`
- `docs/research-revision/phase-01/paper-claim-correction-matrix.md`
- `docs/research-revision/phase-01/reviewer-concern-mapping.md`
- `docs/research-revision/phase-02/paper-methodology-edit-spec.md`
- `docs/research-revision/phase-02/methodology-edit-checklist.md`
- `docs/research-revision/phase-02/evaluation-protocol-correction-spec.md`
- `docs/research-revision/phase-02/timer-scoring-correction-spec.md`
- `docs/research-revision/phase-02/score-treatment-decision-record.md`
- `docs/research-revision/phase-02/retrieval-indexing-correction-spec.md`
- `docs/research-revision/phase-02/phase-02-methodology-exit-gate.md`
- `docs/research-revision/phase-03/sprint-3-1-methods-methodology-paper-edit.md`
- `docs/research-revision/phase-03/methodology-edit-traceability.md`
- `latex_publication/v1/sn-article.tex`

## 3. Files Changed

- `latex_publication/v1/sn-article.tex`
- `docs/research-revision/phase-03/results-evaluation-edit-traceability.md`
- `docs/research-revision/phase-03/sprint-3-2-results-evaluation-framing-paper-edit.md`

## 4. Commands Run

- `pwd`
- `git status --short`
- `git branch --show-current`
- `git remote -v`
- `git diff --check`
- `mkdir -p /tmp/clinical-rag-paper-edit-review/sprint-3-2`
- `cp latex_publication/v1/sn-article.tex /tmp/clinical-rag-paper-edit-review/sprint-3-2/sn-article.before.tex`
- `sed -n '1,360p' docs/research-revision/phase-01/authoritative-results-manifest.md 2>/dev/null || true`
- `sed -n '1,360p' docs/research-revision/phase-01/reproduction-run-manifest.md 2>/dev/null || true`
- `sed -n '1,360p' docs/research-revision/phase-01/paper-claim-correction-matrix.md 2>/dev/null || true`
- `sed -n '1,360p' docs/research-revision/phase-01/reviewer-concern-mapping.md 2>/dev/null || true`
- `sed -n '1,420p' docs/research-revision/phase-02/paper-methodology-edit-spec.md 2>/dev/null || true`
- `sed -n '1,420p' docs/research-revision/phase-02/methodology-edit-checklist.md 2>/dev/null || true`
- `sed -n '1,360p' docs/research-revision/phase-02/evaluation-protocol-correction-spec.md 2>/dev/null || true`
- `sed -n '1,360p' docs/research-revision/phase-02/timer-scoring-correction-spec.md 2>/dev/null || true`
- `sed -n '1,360p' docs/research-revision/phase-02/score-treatment-decision-record.md 2>/dev/null || true`
- `sed -n '1,360p' docs/research-revision/phase-02/retrieval-indexing-correction-spec.md 2>/dev/null || true`
- `sed -n '1,360p' docs/research-revision/phase-02/phase-02-methodology-exit-gate.md 2>/dev/null || true`
- `sed -n '1,360p' docs/research-revision/phase-03/sprint-3-1-methods-methodology-paper-edit.md 2>/dev/null || true`
- `sed -n '1,360p' docs/research-revision/phase-03/methodology-edit-traceability.md 2>/dev/null || true`
- `rg -n "\\section|\\subsection|\\subsubsection|Results|Evaluation|hard-negative|Hard-negative|controlled|simulated|96\\.0|52\\.5|58|80|22|69|Accuracy@1|Recall@5|semantic-only|baseline|Phase 9|target-subject|filtered|score-treatment|L2|cosine|inner product|Semantic Collision|Negation Recency|Terminology Drift|Real-World Mining|STAR-RAG|time-filter|comparator|production|deployment|clinical utility|superior|outperform" latex_publication/v1/sn-article.tex || true`
- `sed -n '488,590p' latex_publication/v1/sn-article.tex`
- `sed -n '590,670p' latex_publication/v1/sn-article.tex`
- `sed -n '410,435p' latex_publication/v1/sn-article.tex`
- `sed -n '478,487p' latex_publication/v1/sn-article.tex`
- `rg -n "controlled/simulated|mechanism-isolation|supplied-score|protocol-bound|Phase 9|target-subject-filtered|score-treatment-limited|transformed L2-derived|deployment-performance|unconstrained retrieval|semantic-only row" latex_publication/v1/sn-article.tex`
- `sed -n '488,548p' latex_publication/v1/sn-article.tex`
- `cp latex_publication/v1/sn-article.tex /tmp/clinical-rag-paper-edit-review/sprint-3-2/sn-article.after.tex`
- `diff -u /tmp/clinical-rag-paper-edit-review/sprint-3-2/sn-article.before.tex /tmp/clinical-rag-paper-edit-review/sprint-3-2/sn-article.after.tex > /tmp/clinical-rag-paper-edit-review/sprint-3-2/sprint-3-2.diff || true`
- `git diff --check`
- `git status --short`
- `diff -u /tmp/clinical-rag-paper-edit-review/sprint-3-2/sn-article.before.tex /tmp/clinical-rag-paper-edit-review/sprint-3-2/sn-article.after.tex | sed -n '1,260p'`
- `git status --short --untracked-files=all`

## 5. Results / Evaluation Sections Edited

- `\section{Results}`
- `\subsection{Overall Performance}`
- Hard Negative Stress Test table caption
- `\subsection{Statistical Significance}`
- `\subsection{End-to-End Retrieval and Ablation Study}`
- Phase 9 end-to-end ablation table caption

## 6. Phase 01 / Phase 02 / Phase 03 Evidence Used

- Phase 01 authoritative result and reproduction manifests for the controlled hard-negative and filtered Phase 9 aggregate authority.
- Phase 01 claim matrix and reviewer mapping for overclaim and comparator-risk disposition.
- Phase 02 evaluation protocol, scoring, score-treatment, retrieval/indexing, and methodology handoff specs.
- Phase 03 Sprint 3.1 report and traceability for prior Methods-only edits and deferred Results / Discussion issues.

## 7. Controlled Hard-Negative Framing Edits

- Labeled the 96.0% TIMER vs 52.5% semantic-only result as controlled/simulated mechanism-isolation evidence.
- Identified the hard-negative semantic-only baseline as supplied-score and protocol-specific.
- Stated that the result is not an unconstrained retrieval or deployment-performance estimate.

## 8. Phase 9 Framing Edits

- Labeled Phase 9 as reproduced local end-to-end ablation evidence.
- Stated the target-subject-filtered flow: local-index search, target-subject filtering, first 50 filtered candidates, ablation scoring, and deduplicated note-level top-1/top-5 evaluation.
- Preserved 58.0% Accuracy@1 / 80.0% Recall@5 vs 22.0% / 69.0% without numeric changes.
- Removed deployment-realistic and unconstrained implications from the edited Results text.

## 9. Score-Treatment Limitation Edits

- Stated that Phase 9 uses a protocol-specific transformed L2-derived score.
- Stated that this score is not validated as cosine, inner-product, or general semantic similarity.
- Limited aggregate interpretation to the reproduced local ablation protocol.

## 10. Semantic-Only Baseline Separation Edits

- Distinguished the supplied-score hard-negative semantic-only baseline from the Phase 9 semantic-only ablation.
- Identified Phase 9 semantic-only as target-subject-filtered and score-treatment-limited.
- Stated that semantic-only comparisons do not substitute for missing external comparator baselines.

## 11. Negation Recency Edits

No direct edit was made because detailed Negation Recency interpretation appears in Discussion, which was out of scope. The Results table numerics were preserved. A later approved Discussion or Limitations sprint should reframe this as no improvement / scenario saturation / non-regression without safety overclaim.

## 12. Terminology Drift Caution Edits

No direct edit was made because detailed Terminology Drift interpretation appears in Discussion, which was out of scope. The Results table numerics and scenario count were preserved. A later approved Discussion or Limitations sprint should caution that Terminology Drift has `n=20` and avoid overconfident scenario-level conclusions.

## 13. Missing Comparator Limitation Edits

The Results end-to-end paragraph now states that semantic-only comparisons do not substitute for empirical STAR-RAG-style or simple time-filter comparator baselines, which were not implemented in the reproduced evaluations.

## 14. Unsupported Claims Removed or Softened

- Softened broad hard-negative improvement language to protocol-bound mechanism-isolation language.
- Softened broad statistical and practical-improvement wording to controlled-protocol wording.
- Replaced realistic/deployment-like Phase 9 framing with reproduced local target-subject-filtered framing.
- Removed Phase 9 wording that implied validated semantic-score correctness.
- Added missing-comparator limitation language.

## 15. Deferred Edits for Discussion / Limitations / Abstract / Conclusion

- Abstract: broad headline and clinical/deployment-adjacent performance framing.
- Introduction: broad empirical-validation and comparator-positioning language.
- Methods / Methodology: any remaining evaluation-setup wording outside Results was not edited.
- Discussion: Negation Recency, Terminology Drift, Real-World Mining, visualization, safety, and clinical implication wording.
- Limitations: tuning/sensitivity and downstream evaluation wording.
- Conclusion: unconstrained Phase 9, virtual-graph, deployment, and broad safety framing.
- Declarations / Supplementary Material: unsupported supplement-availability wording.

## 16. Diff Review Method Used Despite Ignored LaTeX Source

Because `latex_publication/` is ignored by Git, a temporary before/after manuscript snapshot was used:

- Before path: `/tmp/clinical-rag-paper-edit-review/sprint-3-2/sn-article.before.tex`
- After path: `/tmp/clinical-rag-paper-edit-review/sprint-3-2/sn-article.after.tex`
- Diff path: `/tmp/clinical-rag-paper-edit-review/sprint-3-2/sprint-3-2.diff`

The diff was generated after editing and inspected with a bounded `diff -u ... | sed -n '1,260p'` review.

## 17. Open Risks

- Results-facing Discussion text remains out of scope and still needs a later gated edit.
- Phase 9 score treatment remains unresolved.
- Missing STAR-RAG-style and simple time-filter comparators remain future work.
- Negation Recency and Terminology Drift interpretation needs later Discussion / Limitations cleanup.
- The paper source remains ignored by Git, so manuscript edits do not appear in normal Git diffs.

## 18. Scope Confirmation

- Paper source edited only within approved Results / Evaluation-facing scope.
- Bibliography was not edited.
- Code was not edited.
- Tests were not edited.
- Notebooks were not edited.
- Results files were not changed.
- Tables were not numerically changed.
- Figures were not changed.
- Data files were not changed.
- Index files were not changed.
- No experiments were run.
- No reproduction was rerun.
- Paper was not compiled.
- No final rebuttal text was drafted.
- No raw clinical-note text, raw MIMIC rows, raw query/answer text, retrieved chunks, raw sidecar records, or raw result rows were exposed.
- No external APIs/services were used.
- No files were staged, committed, or pushed.
