# Sprint 4.1 -- Major Comments Rebuttal File

## 1. sprint goal

Create a concise, submission-ready draft response for major comments only, using the revised manuscript and Phase 01-03 traceability evidence. This sprint is a rebuttal-document sprint only.

## 2. files inspected

- `latex_publication/v1/sn-article.tex` for location references only.
- `docs/research-revision/phase-01/reviewer-concern-mapping.md`
- `docs/research-revision/phase-01/paper-claim-correction-matrix.md`
- `docs/research-revision/phase-01/authoritative-results-manifest.md`
- `docs/research-revision/phase-02/paper-methodology-edit-spec.md`
- `docs/research-revision/phase-02/methodology-edit-checklist.md`
- `docs/research-revision/phase-02/phase-02-methodology-exit-gate.md`
- `docs/research-revision/phase-03/final-manuscript-consistency-audit.md`
- `docs/research-revision/phase-03/final-reviewer-concern-traceability.md`
- `docs/research-revision/phase-03/submission-readiness-checklist.md`
- `docs/research-revision/phase-03/source-tracking-export-decision-record.md`

## 3. files created

- `docs/research-revision/phase-04/major-comments-rebuttal-draft.md`
- `docs/research-revision/phase-04/rebuttal-traceability-to-manuscript.md`
- `docs/research-revision/phase-04/open-supervisor-decisions-before-submission.md`
- `docs/research-revision/phase-04/sprint-4-1-major-comments-rebuttal-file.md`

## 4. commands run

- `pwd`
- `git status --short`
- `git branch --show-current`
- `git remote -v`
- `git diff --check`
- `find docs/research-revision/phase-01 -maxdepth 1 -type f | sort`
- `find docs/research-revision/phase-02 -maxdepth 1 -type f | sort`
- `find docs/research-revision/phase-03 -maxdepth 1 -type f | sort`
- Approved `sed -n '1,420p' ... 2>/dev/null || true` inspections for Phase 01, Phase 02, and Phase 03 evidence docs.
- Approved manuscript `rg -n` location-reference search.
- `mkdir -p docs/research-revision/phase-04`
- Final `git diff --check`
- Final `git status --short`

## 5. major comments included

- P08
- P11-P13
- P14
- P15
- P16
- P19-P21
- P23-P26
- P28
- P30-P31
- P34-P36
- P38 as supervisor-decision-required

## 6. comments intentionally excluded and why

Minor comments were excluded because the green gate approved major comments only. No non-major issue was used except where directly necessary to explain a major-comment response.

## 7. rebuttal strategy

The rebuttal uses the required pattern: Comment, Response, Manuscript location, and Status. It relies only on manuscript correction, manuscript clarification, limitation framing, future work, or supervisor decision required. It does not claim new experiments, new metrics, new baselines, router analysis, sensitivity analysis, score-treatment validation, clinical validation, or deployment readiness.

## 8. manuscript locations referenced

- Abstract
- Introduction
- Related Work / Temporal-Aware Retrieval
- Proposed Methodology / Intent Classification Router
- Proposed Methodology / Temporal Metadata Sidecar
- Proposed Methodology / TIMER Scoring Function
- Experimental Setup / Dataset
- Experimental Setup / Hard Negative Stress Test
- Experimental Setup / Implementation Details
- Experimental Setup / Limitations of the Controlled Evaluation and Path to Realistic Testing
- Results / Overall Performance
- Results / End-to-End Retrieval and Ablation Study
- Discussion / Per-Scenario Analysis
- Discussion / Per-Query Analysis
- Discussion / Why TIMER-Graph Works
- Discussion / The Role of Negative Beta
- Discussion / Clinical Implications
- Discussion / Limitations
- Conclusion

## 9. open supervisor decisions

- Keep `TIMER-Graph` title/name as-is, or revise title in a separate sprint.
- Final journal-safe code availability wording.
- Final journal-safe supplementary material wording.
- Source-tracking/export path for ignored LaTeX source.
- P38 handling.
- Whether limitation framing is sufficient or extra analysis is required before final rebuttal.

## 10. risks

- P38 remains untraceable and requires supervisor clarification.
- Final source tracking/export remains unresolved because `latex_publication/` is ignored by Git.
- Code/supplement availability wording remains supervisor-dependent.
- The rebuttal is ready as a draft, not a final submission action.

## 11. scope confirmation

- No manuscript source edited.
- Title not edited.
- Bibliography not edited.
- Code not edited.
- Tests not edited.
- Notebooks not edited.
- `.gitignore` not edited.
- Results not changed.
- Data not changed.
- Indexes not changed.
- No experiments run.
- No reproduction rerun.
- No new metrics generated.
- No citations added or removed.
- No final submission action taken.
- No raw clinical-note text exposed.
- No raw MIMIC rows printed.
- No raw query/answer/retrieved text printed.
- No external APIs/services used.
- No staging, commits, force-adds, or pushes made.
