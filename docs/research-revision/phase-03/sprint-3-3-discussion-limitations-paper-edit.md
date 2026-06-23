# Sprint 3.3 — Discussion and Limitations Paper Edit

## 1. Sprint Goal

Edit only Discussion / Limitations-facing manuscript content so the paper honestly explains what the reported results mean, what they do not establish, and which limitations remain. The sprint preserves all numeric results, tables, figures, bibliography, code, tests, notebooks, data, indexes, and raw local artifacts.

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
- `docs/research-revision/phase-02/intent-router-correction-spec.md`
- `docs/research-revision/phase-02/temporal-sidecar-virtual-graph-correction-spec.md`
- `docs/research-revision/phase-02/phase-02-methodology-exit-gate.md`
- `docs/research-revision/phase-03/sprint-3-1-methods-methodology-paper-edit.md`
- `docs/research-revision/phase-03/methodology-edit-traceability.md`
- `docs/research-revision/phase-03/sprint-3-2-results-evaluation-framing-paper-edit.md`
- `docs/research-revision/phase-03/results-evaluation-edit-traceability.md`
- `latex_publication/v1/sn-article.tex`

## 3. Files Changed

- `latex_publication/v1/sn-article.tex`
- `docs/research-revision/phase-03/discussion-limitations-edit-traceability.md`
- `docs/research-revision/phase-03/sprint-3-3-discussion-limitations-paper-edit.md`

## 4. Commands Run

- `pwd`
- `git status --short`
- `git branch --show-current`
- `git remote -v`
- `git diff --check`
- `mkdir -p /tmp/clinical-rag-paper-edit-review/sprint-3-3`
- `cp latex_publication/v1/sn-article.tex /tmp/clinical-rag-paper-edit-review/sprint-3-3/sn-article.before.tex`
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
- `sed -n '1,360p' docs/research-revision/phase-02/intent-router-correction-spec.md 2>/dev/null || true`
- `sed -n '1,360p' docs/research-revision/phase-02/temporal-sidecar-virtual-graph-correction-spec.md 2>/dev/null || true`
- `sed -n '1,360p' docs/research-revision/phase-02/phase-02-methodology-exit-gate.md 2>/dev/null || true`
- `sed -n '1,360p' docs/research-revision/phase-03/sprint-3-1-methods-methodology-paper-edit.md 2>/dev/null || true`
- `sed -n '1,360p' docs/research-revision/phase-03/methodology-edit-traceability.md 2>/dev/null || true`
- `sed -n '1,360p' docs/research-revision/phase-03/sprint-3-2-results-evaluation-framing-paper-edit.md 2>/dev/null || true`
- `sed -n '1,360p' docs/research-revision/phase-03/results-evaluation-edit-traceability.md 2>/dev/null || true`
- `rg -n "\\section|\\subsection|\\subsubsection|Discussion|Limitations|Failure|Threats|Future Work|Negation Recency|Terminology Drift|n=20|confidence interval|score-treatment|L2|cosine|semantic similarity|Phase 9|target-subject|filtered|hard-negative|controlled|simulated|router|accuracy|calibration|confidence|fallback|near-miss|baseline|STAR-RAG|time-filter|clinical|deployment|production|real-world|generalizable|supplement|supplementary|current-index|lineage|MIMIC|corpus|scale" latex_publication/v1/sn-article.tex || true`
- `sed -n '553,646p' latex_publication/v1/sn-article.tex`
- `sed -n '646,672p' latex_publication/v1/sn-article.tex`
- `rg -n "^\\section|^\\subsection|^\\subsubsection" latex_publication/v1/sn-article.tex`
- `sed -n '553,646p' latex_publication/v1/sn-article.tex`
- `rg -n "no observed gain|scenario saturation|n=20|transformed L2-derived|cosine similarity|inner-product|STAR-RAG-style|simple time-filter|regex/rule-based|zero-match|controlled/simulated|mechanism-isolation|target-subject-filtered|current-index build lineage|generated or randomized offsets|deployment readiness|clinical utility|supplement" latex_publication/v1/sn-article.tex`
- `sed -n '646,672p' latex_publication/v1/sn-article.tex`
- `cp latex_publication/v1/sn-article.tex /tmp/clinical-rag-paper-edit-review/sprint-3-3/sn-article.after.tex`
- `diff -u /tmp/clinical-rag-paper-edit-review/sprint-3-3/sn-article.before.tex /tmp/clinical-rag-paper-edit-review/sprint-3-3/sn-article.after.tex > /tmp/clinical-rag-paper-edit-review/sprint-3-3/sprint-3-3.diff || true`
- `git diff --check`
- `git status --short`
- `diff -u /tmp/clinical-rag-paper-edit-review/sprint-3-3/sn-article.before.tex /tmp/clinical-rag-paper-edit-review/sprint-3-3/sn-article.after.tex | sed -n '1,260p'`

## 5. Discussion / Limitations Sections Edited

All edited sections are under `\section{Discussion}`:

- `\subsection{Per-Scenario Analysis}`
- `\subsection{Scenario-Level Visualization}`
- `\subsection{Per-Query Analysis}`
- `\subsection{Why TIMER-Graph Works}`
- `\subsection{The Role of Negative Beta}`
- `\subsection{Clinical Implications}`
- `\subsection{Limitations}`

Not edited:

- Experimental Setup `\subsection{Limitations of the Controlled Evaluation and Path to Realistic Testing}` because its parent section is not Discussion.
- Conclusion and Supplementary Material because both are forbidden in Sprint 3.3.

## 6. Phase 01 / Phase 02 / Phase 03 Evidence Used

- Phase 01 authoritative results and reproduction manifests.
- Phase 01 claim matrix and reviewer concern mapping.
- Phase 02 methodology, evaluation-protocol, TIMER-scoring, score-treatment, retrieval/indexing, router, sidecar, and exit-gate specs.
- Phase 03 Sprint 3.1 and 3.2 reports and traceability files.

## 7. Negation Recency Correction

Negation Recency is now framed as no observed gain because both systems are already at 100% in the controlled fixture. It is described as saturation and non-regression, not improvement or clinical safety evidence.

## 8. Terminology Drift Caution

Terminology Drift is now framed cautiously because the scenario has `n=20`. The text avoids a strong causal explanation and calls for future error analysis.

## 9. Phase 9 Score-Treatment Limitation

The Limitations section states that Phase 9 is local and target-subject-filtered, and that its semantic input is a protocol-specific transformed L2-derived value not validated as cosine, inner-product, or general semantic similarity.

## 10. Missing Comparator Limitation

The Limitations section states that STAR-RAG-style and simple time-filter baselines were not empirically implemented and that semantic-only baselines do not resolve all comparator concerns.

## 11. Router Evidence-Gap Limitation

The Discussion and Limitations sections now describe the router as fixed regex/rule-based, not learned or calibrated. They identify missing accuracy, calibration, confidence distribution, trigger/fallback, zero-match/tie, and near-miss analysis, including the zero-match Current fallback caveat.

## 12. Controlled / Simulated Hard-Negative Limitation

Discussion and Limitations now state that the hard-negative result is controlled/simulated mechanism-isolation evidence using supplied semantic inputs, not natural-score retrieval, clinical utility, or deployment evidence.

## 13. Target-Subject-Filtered Phase 9 Limitation

The local Phase 9 result is explicitly framed as target-subject-filtered and not unconstrained, deployment-realistic, population-scale, or full-corpus retrieval validation.

## 14. Corpus Scale / Current-Index Lineage Limitation

The Limitations section states that the corpus is a local processed subset, the inspected local index contains 1,206 vectors, and current-index build lineage is not fully proven.

## 15. Temporal Metadata Sidecar Provenance Limitation

The Limitations section states that sidecar temporal fields include generated or randomized offsets and placeholder date provenance, not observed clinical temporal provenance or graph-based retrieval behavior.

## 16. Clinical / Deployment Overclaims Removed or Softened

Discussion clinical-implication language now uses prospective and protocol-bound wording. It states that clinical utility, safety, and deployment readiness require validation beyond current experiments.

## 17. Supplement Availability Handling

No Discussion / Limitations supplement-availability claim was present in the editable section. The unsupported Supplementary Material claim remains in Declarations/Supplementary Material and is deferred because that section is forbidden in Sprint 3.3.

## 18. Reviewer Concern Mapping

- P11-P13: controlled supplied-score hard-negative scope and fixed-score caveat.
- P14: Phase 9 remains prominent but local, target-subject-filtered, and score-treatment-limited.
- P15: Negation Recency no-gain and saturation.
- P16: Terminology Drift `n=20` caution.
- P19-P21: router accuracy, calibration, confidence distribution, fallback, and near-miss gaps.
- P23-P26: fixed alpha/lambda/beta/tau settings and missing sensitivity evidence.
- P28: local processed subset and non-production-scale scope.
- P30-P31: missing STAR-RAG-style and simple time-filter comparators.
- P34-P36: failure-analysis and limitation framing without new analyses.

## 19. Deferred Edits for Abstract / Introduction / Conclusion / Declarations

- Abstract: broad result and virtual-graph wording.
- Introduction: broad validation, virtual-graph, and contribution framing.
- Experimental Setup limitation subsection: parent section is outside Discussion.
- Conclusion: unconstrained Phase 9, virtual-graph, deployment-practicality, and broad safety wording.
- Declarations / Supplementary Material: unsupported supplement-availability wording.

## 20. Diff Review Method Used Despite Ignored LaTeX Source

Because `latex_publication/` is ignored by Git, Sprint 3.3 uses temporary review artifacts:

- Before path: `/tmp/clinical-rag-paper-edit-review/sprint-3-3/sn-article.before.tex`
- After path: `/tmp/clinical-rag-paper-edit-review/sprint-3-3/sn-article.after.tex`
- Diff path: `/tmp/clinical-rag-paper-edit-review/sprint-3-3/sprint-3-3.diff`

The diff was generated after editing and inspected with a bounded `diff -u ... | sed -n '1,260p'` review.

## 21. Open Risks

- Abstract, Introduction, Conclusion, and Supplementary Material still contain out-of-scope overclaims.
- Phase 9 score treatment remains unresolved.
- Router quality and sensitivity analyses remain absent.
- Comparator baselines remain unimplemented.
- Exact current-index build lineage remains incomplete.
- The paper source remains ignored by Git, so manuscript edits do not appear in normal Git diffs.

## 22. Scope Confirmation

- Paper source edited only within approved Discussion / Limitations-facing scope.
- Abstract was not edited.
- Introduction was not edited.
- Methods / Methodology was not edited.
- Results / Evaluation was not edited.
- Conclusion was not edited.
- Declarations / Supplementary Material was not edited.
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
