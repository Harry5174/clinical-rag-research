# Sprint 4.2 -- Final Decision Application and Submission Handoff

## 1. Sprint goal

Apply final working decisions for ASAP submission, update rebuttal and handoff docs, make only approved availability wording changes if needed, check compile/export readiness, and create a safe temporary handoff package outside the repository.

## 2. Files inspected

- `latex_publication/v1/sn-article.tex`
- `latex_publication/v1/sn-bibliography.bib` for handoff/export awareness
- `docs/research-revision/phase-04/major-comments-rebuttal-draft.md`
- `docs/research-revision/phase-04/rebuttal-traceability-to-manuscript.md`
- `docs/research-revision/phase-04/open-supervisor-decisions-before-submission.md`
- Relevant Phase 03 and Phase 04 P38 traceability via approved search

## 3. Files changed

- `latex_publication/v1/sn-article.tex`
- `docs/research-revision/phase-04/major-comments-rebuttal-draft.md`
- `docs/research-revision/phase-04/rebuttal-traceability-to-manuscript.md`
- `docs/research-revision/phase-04/open-supervisor-decisions-before-submission.md`

## 4. Commands run

- `pwd`
- `git status --short --untracked-files=all`
- `git branch --show-current`
- `git remote -v`
- `git diff --check`
- `mkdir -p /tmp/clinical-rag-paper-edit-review/sprint-4-2`
- `cp latex_publication/v1/sn-article.tex /tmp/clinical-rag-paper-edit-review/sprint-4-2/sn-article.before.tex`
- Approved availability wording `rg` inspection.
- Approved P38 target `rg` inspection.
- Approved rebuttal-doc `sed` inspections.
- `command -v latexmk || true`
- `ls -la latex_publication/v1 || true`
- `find latex_publication/v1 -maxdepth 1 -type f | sort || true`
- Temporary handoff package creation and safe file copies.
- Final manuscript after-snapshot and `/tmp` diff generation.
- Final `git diff --check`
- Final `git status --short --untracked-files=all`
- Final handoff package file listing.

## 5. Final decisions applied

- Title/name: keep `TIMER-Graph`.
- Code availability: conservative restricted-data-safe wording.
- Supplementary material: no supplement asserted or invented.
- Source tracking/export: local/Overleaf handoff; no Git tracking change.
- P38: reviewed / no manuscript target identified / no manuscript edit required.
- Extra analysis: not required before ASAP submission.

## 6. Manuscript edits, if any

One approved manuscript edit was made: Code availability wording was updated. No other manuscript content was edited.

## 7. Rebuttal edits

The major-comments rebuttal draft now records final decisions, P38 rebuttal-only handling, no-extra-analysis language, and safe availability posture. Traceability now marks P38 as reviewed / no manuscript target identified. The open-supervisor-decisions file is converted into a final working decision record.

## 8. P38 handling

The approved P38 search did not identify a clear active manuscript target requiring correction. The generic manuscript wording about beta weighting is part of the TIMER scoring mechanism and is not a weighted-scheduling / heterogeneous-workload claim. No P38 manuscript edit was made.

## 9. Availability wording handling

Code availability was supervisor-decision-gated before Sprint 4.2 and was replaced with approved conservative wording. Supplementary information already did not assert supplement availability and was left unchanged. Data availability was left unchanged.

## 10. Compile/export status

`LOCAL_COMPILE_NOT_AVAILABLE` -- `latexmk` was not available locally. Overleaf or an external LaTeX environment is required for compile/export.

## 11. Temporary handoff package

Temporary handoff folder:

```text
/tmp/clinical-rag-submission-handoff/sprint-4-2/
```

Manifest:

```text
/tmp/clinical-rag-submission-handoff/sprint-4-2/handoff-file-manifest.txt
```

Only safe manuscript, bibliography, rebuttal, and traceability files were copied.

## 12. Source-tracking/export path

For ASAP submission, use local/Overleaf export handoff. No `.gitignore` modification, force-add, staging, commit, or push was performed.

## 13. Remaining optional supervisor preferences

- Supervisor may compile/export through Overleaf or another LaTeX environment.
- Supervisor may later request public-release review, supplement packaging, or additional analysis.
- Supervisor may later choose to track LaTeX source in Git through a separately approved closeout step.

## 14. Risks

- Local compile unavailable due missing `latexmk`.
- LaTeX source remains ignored by Git.
- External compile/export remains a supervisor handoff step.

## 15. Scope confirmation

- Title not edited.
- Author list not edited.
- Affiliations not edited.
- Bibliography content not edited.
- Code not edited.
- Tests not edited.
- Notebooks not edited.
- `.gitignore` not edited.
- Data not changed.
- Indexes not changed.
- Result files not changed.
- Numeric results not changed.
- Table values not changed.
- Figure contents not changed.
- No experiments run.
- No reproduction rerun.
- No new metrics generated.
- No citations added or removed.
- No raw clinical-note text exposed.
- No raw MIMIC rows printed.
- No raw query/answer/retrieved text printed.
- No external APIs/services used.
- No staging, commits, force-adds, or pushes made.
- No final submission action taken.
