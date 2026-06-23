# Sprint 3.1 — Methods and Methodology Paper Edit

## 1. Sprint Goal

Edit only Methods / Methodology-related manuscript content so the paper reflects the approved Phase 02 methodology specifications. The sprint corrects unsupported method claims, clarifies implementation behavior, softens overclaims, applies approved terminology, and records traceability without changing results, code, data, bibliography, or out-of-scope paper sections.

## 2. Files Inspected

- `latex_publication/v1/sn-article.tex`
- `docs/research-revision/phase-02/paper-methodology-edit-spec.md`
- `docs/research-revision/phase-02/methodology-edit-checklist.md`
- `docs/research-revision/phase-02/phase-02-methodology-exit-gate.md`
- `docs/research-revision/phase-02/pipeline-truth-table.md`
- `docs/research-revision/phase-02/methodology-gap-report.md`
- `docs/research-revision/phase-02/retrieval-indexing-correction-spec.md`
- `docs/research-revision/phase-02/faiss-normalization-decision-record.md`
- `docs/research-revision/phase-02/timer-scoring-correction-spec.md`
- `docs/research-revision/phase-02/evaluation-protocol-correction-spec.md`
- `docs/research-revision/phase-02/score-treatment-decision-record.md`
- `docs/research-revision/phase-02/temporal-sidecar-virtual-graph-correction-spec.md`
- `docs/research-revision/phase-02/intent-router-correction-spec.md`
- `docs/research-revision/phase-02/virtual-graph-naming-decision-record.md`

## 3. Files Changed

- `latex_publication/v1/sn-article.tex`
- `docs/research-revision/phase-03/methodology-edit-traceability.md`
- `docs/research-revision/phase-03/sprint-3-1-methods-methodology-paper-edit.md`

## 4. Commands Run

- `pwd`
- `git status --short`
- `git branch --show-current`
- `git remote -v`
- `git diff --check`
- `sed -n '1,420p' docs/research-revision/phase-02/paper-methodology-edit-spec.md`
- `sed -n '1,420p' docs/research-revision/phase-02/methodology-edit-checklist.md`
- `sed -n '1,360p' docs/research-revision/phase-02/phase-02-methodology-exit-gate.md`
- `sed -n '1,420p' docs/research-revision/phase-02/pipeline-truth-table.md`
- `sed -n '1,420p' docs/research-revision/phase-02/methodology-gap-report.md`
- `sed -n '1,360p' docs/research-revision/phase-02/retrieval-indexing-correction-spec.md`
- `sed -n '1,360p' docs/research-revision/phase-02/faiss-normalization-decision-record.md`
- `sed -n '1,360p' docs/research-revision/phase-02/timer-scoring-correction-spec.md`
- `sed -n '1,360p' docs/research-revision/phase-02/evaluation-protocol-correction-spec.md`
- `sed -n '1,360p' docs/research-revision/phase-02/score-treatment-decision-record.md`
- `sed -n '1,360p' docs/research-revision/phase-02/temporal-sidecar-virtual-graph-correction-spec.md`
- `sed -n '1,360p' docs/research-revision/phase-02/intent-router-correction-spec.md`
- `sed -n '1,360p' docs/research-revision/phase-02/virtual-graph-naming-decision-record.md`
- `rg -n '\\section|\\subsection|\\subsubsection|\\paragraph|IndexFlatIP|IndexFlatL2|FAISS|cosine|inner product|L2|embedding|BGE|chunk|header|sidecar|Virtual Graph|intent|router|tau|alpha|beta|lambda|TIMER|Ssem|hard-negative|controlled|simulated|Phase 9|target-subject|semantic-only' latex_publication/v1/sn-article.tex`
- `sed -n '217,376p' latex_publication/v1/sn-article.tex`
- `sed -n '377,487p' latex_publication/v1/sn-article.tex`
- `ls docs/research-revision`
- `find docs/research-revision -maxdepth 2 -type f | sort`
- `mkdir -p docs/research-revision/phase-03`
- `rg -n 'IndexFlatIP|exact inner product|exact-inner-product|Virtual Graph|virtual graph|Supplementary Material|supplementary materials|tuned|calibrated|raw dense cosine|source-observed|graph edges|graph traversal|graph database substitute|unconstrained|deployment-ready' latex_publication/v1/sn-article.tex`
- Final checks: `git diff --check`; `git status --short`; `git diff -- latex_publication/v1/sn-article.tex`
- Tracking checks after final diff showed no manuscript diff: `git ls-files latex_publication/v1/sn-article.tex`; `git status --short --untracked-files=all`; `git check-ignore -v latex_publication/v1/sn-article.tex`; `git ls-files -v latex_publication/v1/sn-article.tex`
- Source proof scan: `rg -n 'IndexFlatL2|METRIC|target-subject|zero regex|Temporal Metadata Sidecar|generated local temporal-scoring|transformed value derived from FAISS L2|fixed local implementation/evaluation|separate development-set tuning record|current-index build invocation' latex_publication/v1/sn-article.tex docs/research-revision/phase-03`

Note: `latex_publication/v1/sn-article.tex` is ignored by the repository rule `.gitignore:26:latex_publication/`, so `git status` and `git diff -- latex_publication/v1/sn-article.tex` do not show the paper-source edit even though the local file was edited. The Phase 03 docs are untracked and visible to Git.

## 5. Methods / Methodology Sections Edited

- `\section{Proposed Methodology}`
- `\subsection{Problem Formulation}`
- `\subsection{Intent Classification Router}`
- `\subsection{Temporal Metadata Sidecar}` (renamed from visible Virtual Graph wording)
- `\subsection{TIMER Scoring Function}`
- `\subsubsection{Negative Beta Mechanism}`
- `\subsection{Two-Stage Retrieval Pipeline}`
- `\section{Experimental Setup}` methodology-facing content:
  - `\subsection{Dataset}`
  - `\subsection{Hard Negative Stress Test}` scenario 4 wording
  - `\subsection{Implementation Details}`

The explicit `\subsection{Limitations of the Controlled Evaluation and Path to Realistic Testing}` was not edited because the green gate forbids Limitations editing.

## 6. Phase 02 Specs Used

- Consolidated edit specification and checklist
- Pipeline truth table and methodology gap report
- Retrieval/indexing and FAISS normalization specs
- TIMER scoring and score-treatment specs
- Evaluation protocol spec
- Temporal sidecar / Virtual Graph correction spec and naming decision record
- Intent-router correction spec
- Phase 02 exit gate

## 7. Retrieval / FAISS Corrections Applied

- Replaced methodology `IndexFlatIP` / exact-inner-product wording with `IndexFlatL2` / `METRIC_L2`.
- Added normalized-embedding condition for cosine-equivalent ranking.
- Clarified that raw FAISS outputs remain L2 distances.
- Separated optional cross-encoder architecture from reproduced Phase 9 behavior.
- Described Phase 9 candidate flow as local-index search, target-subject filtering, then first 50 filtered candidates.

## 8. TIMER Scoring Corrections Applied

- Defined the TIMER formula as a local scoring mechanism.
- Clarified that `S_sem` is protocol-specific.
- Stated that controlled evaluation uses supplied fixture scores.
- Stated that Phase 9 uses a transformed L2-derived evaluator value with unresolved semantic interpretation.
- Reframed `alpha`, `beta`, `lambda`, and `tau` as fixed local settings.
- Removed tuning, calibration, and safety implications from methodology-facing text.

## 9. Sidecar / Virtual Graph Corrections Applied

- Renamed visible methodology heading to `Temporal Metadata Sidecar`.
- Reframed the artifact as a JSON metadata sidecar adjacent to the vector index.
- Removed graph nodes, graph edges, graph traversal, and graph-database behavior claims from the edited methodology section.
- Identified offsets as generated local temporal-scoring inputs and `note_date` as a placeholder field.

## 10. Intent-Router Corrections Applied

- Described the router as fixed regex/rule-based logic.
- Kept the Current, Historical, and Trend categories.
- Reframed confidence as a heuristic match-share, not a calibrated probability.
- Described `tau = 0.40` as a fixed local threshold.
- Added the zero-match Current at confidence 0.5 caveat.
- Removed unsupported trained-classifier, accuracy, calibration, and safety implications from edited methodology text.

## 11. Chunking / Corpus / Index-Lineage Edits

- Framed the indexed corpus as a local processed subset.
- Changed temporal offsets to generated temporal offsets.
- Avoided full-MIMIC-scale indexing implication.
- Added a current-index lineage caveat for section-aware/header-propagation chunking.
- Kept BGE-base-en-v1.5 and 1,206-vector facts as local implementation facts.

## 12. Unsupported Claims Removed or Softened

- Removed unsupported supplement availability from methodology-router and implementation-detail text.
- Removed unsupported 20-query tuning-set claim from implementation details.
- Softened graph architecture wording to sidecar metadata lookup.
- Softened negative-beta novelty wording to a local mechanism claim.
- Softened semantic-score wording so Phase 9 is not described as validated cosine/inner-product scoring.
- Softened calibration and safety-threshold language.

## 13. Edits Deferred to Later Sprints

- Abstract correction for virtual graph and broad efficacy wording.
- Introduction correction for Virtual Graph contribution and broad empirical-validation framing.
- Results correction for protocol labels and score-treatment caveats.
- Discussion correction for clinical safety, scenario causality, unconstrained retrieval, and router-quality language.
- Limitations correction for old tuning-development-set language.
- Conclusion correction for unconstrained/patient-filtered, virtual-graph, deployment, and broad safety language.
- Declarations / Supplementary Material correction for unsupported supplement availability.

## 14. Open Risks

- Phase 9 L2-derived score treatment remains unresolved and should not be used for stronger semantic-score claims.
- Router quality, calibration, trigger rates, zero-match/tie rates, and near misses remain unanalyzed.
- Exact current-index build lineage remains incomplete.
- Parameter sensitivity and tuning provenance remain unsupported.
- Supplement release requires a reviewed sanitized release package.
- Out-of-scope sections still contain old claims by instruction and require separate gated sprints.
- Git does not track the manuscript source because `latex_publication/` is ignored; paper-source edits need a supervisor decision if they are expected to become tracked Git changes.

## 15. Scope Confirmation

- Paper source edited only within approved Methods / Methodology scope.
- Bibliography was not edited.
- Code was not edited.
- Tests were not edited or run.
- Notebooks were not edited.
- Results files were not changed.
- Data files were not changed.
- Index files were not changed.
- No experiments were run.
- No reproduction was rerun.
- Paper was not compiled.
- No final rebuttal text was drafted.
- No raw clinical-note text, raw MIMIC rows, raw query/answer text, retrieved chunks, raw sidecar records, or raw result rows were exposed in this report.
- No external APIs/services were used.
- No files were staged, committed, or pushed.
