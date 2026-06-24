# Sprint 3.5 -- Final Manuscript Consistency, Compile / Export Readiness, and Source Tracking

## 1. sprint goal

Perform a final manuscript-wide consistency audit, apply only minimal evidence-backed consistency edits, check compile/export readiness, document source-tracking/export options for ignored LaTeX source, and prepare for a later reviewer-response sprint.

## 2. files inspected

- `latex_publication/v1/sn-article.tex`
- Phase 01 evidence docs:
  - `authoritative-results-manifest.md`
  - `reproduction-run-manifest.md`
  - `paper-claim-correction-matrix.md`
  - `reviewer-concern-mapping.md`
- Phase 02 evidence docs:
  - `paper-methodology-edit-spec.md`
  - `methodology-edit-checklist.md`
  - `evaluation-protocol-correction-spec.md`
  - `timer-scoring-correction-spec.md`
  - `score-treatment-decision-record.md`
  - `retrieval-indexing-correction-spec.md`
  - `intent-router-correction-spec.md`
  - `temporal-sidecar-virtual-graph-correction-spec.md`
  - `phase-02-methodology-exit-gate.md`
- Prior Phase 03 sprint reports and traceability docs from Sprints 3.1, 3.2, 3.3, and 3.4.

## 3. files changed

- `latex_publication/v1/sn-article.tex`
- `docs/research-revision/phase-03/final-manuscript-consistency-audit.md`
- `docs/research-revision/phase-03/submission-readiness-checklist.md`
- `docs/research-revision/phase-03/source-tracking-export-decision-record.md`
- `docs/research-revision/phase-03/final-reviewer-concern-traceability.md`
- `docs/research-revision/phase-03/sprint-3-5-final-manuscript-consistency-compile-export-readiness.md`

## 4. commands run

Command categories run:

- Git safety baseline: `pwd`, `git status --short`, `git branch --show-current`, `git remote -v`, `git diff --check`.
- Temporary snapshot: `mkdir -p /tmp/clinical-rag-paper-edit-review/sprint-3-5`; `cp latex_publication/v1/sn-article.tex /tmp/clinical-rag-paper-edit-review/sprint-3-5/sn-article.before.tex`.
- Exact approved Phase 01, Phase 02, and Phase 03 `sed -n` inspections.
- Manuscript structure search with `rg`.
- High-risk leftover-claim search with `rg`.
- Safe terminology search with `rg`.
- Result-number consistency search with `rg`.
- Compile/export readiness inspection: `ls -la latex_publication/v1`, `find latex_publication/v1 -maxdepth 1 -type f | sort`, `command -v latexmk || true`.
- Local temporary artifact check under `/tmp/clinical-rag-paper-edit-review`.
- Post-edit high-risk and number searches.

Final checks are recorded in the IDE Evidence Report.

## 5. manuscript-wide consistency checks

Checked title, front matter, abstract, introduction, related work, methodology, experimental setup, results, discussion, conclusion, supplementary information, declarations, table captions, and figure captions.

## 6. title review decision

`KEEP_TITLE_WITH_BODY_REFRAMING`.

The title was inspected and not edited. It retains `TIMER-Graph` as the system label, while the body now consistently defines the construct as a prototype retrieval framework using a temporal metadata sidecar rather than graph traversal.

## 7. Related Work review

Related Work contained the only notable final-section consistency issues. Sprint 3.5 softened production-grade, absolute novelty, STAR-RAG comparator, and clinically appropriate phrasing while preserving citations and table values.

## 8. captions / tables review

Captions were checked. Figure 2, Figure 8, and Figure 11 captions were minimally edited for temporal-sidecar and controlled-protocol consistency. Table values and figure files were unchanged.

## 9. declarations / availability review

Supplement availability is not asserted. Data availability points to credentialed PhysioNet access. Code release remains supervisor-decision gated. No public release commitment or raw-data availability claim was added.

## 10. result-number consistency check

The required result-number search confirmed expected values remain present. No numeric results or table values were changed.

## 11. protocol-framing consistency check

Controlled/simulated hard-negative framing, local target-subject-filtered Phase 9 framing, score-treatment limitation, semantic-only protocol separation, missing comparator limitation, fixed local settings, and generated sidecar provenance are consistently present.

## 12. unsupported claims removed or softened

- Related Work production-grade wording softened.
- Absolute gap/novelty wording softened.
- STAR-RAG superiority/comparator wording softened.
- Clinical appropriateness language around negative beta softened to historical-intent local reranking.
- Blanket fixed-0.95 hard-negative limitation corrected.
- Captions strengthened with controlled/local framing.

## 13. compile / export readiness result

`LOCAL_COMPILE_NOT_AVAILABLE`.

`latexmk` was not available in the local environment. No packages were installed, no files were downloaded, no class/style files were modified, and no content changes were made to force compilation.

## 14. source-tracking / export decision record summary

`latex_publication/` is ignored by Git, so normal `git diff` does not verify paper-source edits. Sprint 3.5 used `/tmp` before/after snapshots and generated a temporary diff. A supervisor decision is required before implementing any source-tracking/export path.

## 15. reviewer concern traceability summary

Reviewer concern traceability was created without drafting rebuttal prose. Most concerns are ready for rebuttal in traceability form, while P08 remains partial because the title retains TIMER-Graph and P38 remains unresolved because no manuscript target was confirmed.

## 16. remaining supervisor decisions

- Final source-tracking/export choice.
- Whether the title should remain as-is or receive a later title-specific revision.
- Journal-specific supplement/code availability wording.
- Whether future comparator, router, sensitivity, score-treatment, or failure-analysis sprints should be run.

## 17. open risks

- Local compile could not be performed because `latexmk` was unavailable.
- Ignored LaTeX source remains a source-tracking risk.
- Score-treatment, router, comparator, sensitivity, and exact index-lineage limitations remain unresolved by design.
- Journal-specific availability wording remains supervisor-dependent.

## 18. scope confirmation

- Paper source edited only for final consistency.
- Bibliography not edited.
- Code not edited.
- Tests not edited.
- Notebooks not edited.
- `.gitignore` not edited.
- Results not changed.
- Tables not numerically changed.
- Figures not changed.
- Data not changed.
- Indexes not changed.
- Author list not changed.
- Affiliations not changed.
- No experiments run.
- No reproduction rerun.
- No new metrics generated.
- No citations added or removed.
- No final rebuttal text drafted.
- No raw clinical-note text exposed.
- No raw MIMIC rows printed.
- No raw query/answer/retrieved text printed.
- No external APIs/services used.
- No staging, commits, force-adds, or pushes made.
