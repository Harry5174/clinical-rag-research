# Source Tracking / Export Decision Record -- Sprint 3.5

## 1. Current state

`latex_publication/` is ignored by Git. The manuscript source `latex_publication/v1/sn-article.tex` can be edited locally, but normal `git status` and `git diff` do not provide manuscript-source review visibility.

## 2. Risk

Because the paper source is ignored, normal Git review cannot verify manuscript edits, and an unreviewed local manuscript state could diverge from tracked Phase 03 documentation. This is a source-of-truth and export-readiness risk before submission.

## 3. Evidence used so far

Sprint 3.5 used the following local temporary review artifacts:

- `/tmp/clinical-rag-paper-edit-review/sprint-3-5/sn-article.before.tex`
- `/tmp/clinical-rag-paper-edit-review/sprint-3-5/sn-article.after.tex`
- `/tmp/clinical-rag-paper-edit-review/sprint-3-5/sprint-3-5.diff`

The inspected prior Sprint 3.2, 3.3, and 3.4 reports also document temporary before/after diff review methods for ignored LaTeX edits. The current `/tmp` environment did not retain those earlier temporary files at the time of Sprint 3.5 inspection. Sprint 3.1's inspected report records that the LaTeX source was ignored and not visible to normal Git diff, but did not present a retained Sprint 3.1 `/tmp` before/after artifact in the text inspected during this sprint.

## 4. Options

### Option A: Keep LaTeX local/ignored and export final PDF/source manually

Pros: avoids changing repository tracking policy during the revision sprint. Cons: manuscript provenance depends on local snapshots and manual review discipline.

### Option B: Force-add approved final LaTeX files

Pros: allows Git review of final submission source without changing `.gitignore`. Cons: must be separately approved because force-adding ignored files was forbidden in Sprint 3.5.

### Option C: Modify `.gitignore` in a separate approved closeout sprint

Pros: establishes durable repository tracking policy. Cons: changes repository policy and must not be bundled into this manuscript-consistency sprint.

### Option D: Use Overleaf as source of final submission and keep repo docs only

Pros: aligns with external manuscript-preparation workflows if Overleaf is the submission source. Cons: repo remains unable to verify final LaTeX state unless exports are captured separately.

## 5. Recommendation

Recommendation: do not implement a source-tracking decision in Sprint 3.5. For immediate supervisor review, use the Sprint 3.5 `/tmp` before/after diff plus the Phase 03 audit docs. Before submission, choose either Option B or Option C if the repository must preserve final source, or Option D if Overleaf is the authoritative submission source.

## 6. Supervisor decision required

Supervisor must choose the final source-tracking/export path before submission:

- Keep ignored local source and export manually.
- Force-add approved final LaTeX files.
- Modify `.gitignore` in a separate closeout sprint.
- Use Overleaf as final submission source and keep repository docs only.

## 7. What the IDE agent did not do

- Did not stage files.
- Did not commit files.
- Did not push files.
- Did not force-add ignored LaTeX files.
- Did not modify `.gitignore`.
- Did not implement any source-tracking/export decision.
