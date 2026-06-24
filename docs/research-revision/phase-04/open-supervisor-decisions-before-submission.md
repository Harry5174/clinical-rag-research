# Open Supervisor Decisions Before Submission -- Sprint 4.1

## Recommendation for ASAP submission

For ASAP submission, keep the `TIMER-Graph` title/name, keep limitation framing, and do not run new experiments unless the supervisor specifically requests them.

## 1. Keep TIMER-Graph title with body clarification, or revise title?

Recommendation: keep `TIMER-Graph` as the method/system name for this submission cycle. The rebuttal should clarify that the manuscript now describes a temporal metadata sidecar implementation and does not claim graph nodes, graph edges, traversal, or graph database behavior.

Decision needed: confirm title/name retention, or request a separate title-revision sprint.

## 2. Final journal-safe code availability wording

Current manuscript posture: code release requires a separate supervisor decision and approved public-release review.

Recommendation: keep the current guarded wording unless the supervisor approves a public-release package.

Decision needed: final journal-safe code availability statement.

## 3. Final journal-safe supplementary material wording

Current manuscript posture: supplementary file availability is not asserted; any sanitized supplementary package requires separate supervisor review and approval.

Recommendation: keep the current guarded wording unless a reviewed supplementary package is actually prepared.

Decision needed: final journal-safe supplementary material statement.

## 4. Source-tracking / export path for ignored LaTeX files

Current state: `latex_publication/` is ignored by Git, so normal Git status/diff does not verify manuscript-source edits.

Recommendation: for supervisor review, use the documented Phase 03 `/tmp` diff process and Phase 03/04 traceability docs. Before final submission, choose one source-of-truth path: manual export from local/Overleaf source, force-add approved final LaTeX files, or modify `.gitignore` in a separate closeout sprint.

Decision needed: final source-tracking/export path.

## 5. P38 handling

Current state: no confirmed manuscript target was identified for weighted scheduling, heterogeneous workloads, or the DC2 anomaly.

Recommendation: mark P38 as supervisor-decision-required unless the supervisor supplies the intended manuscript location or confirms it can be excluded.

Decision needed: clarify whether P38 needs a response and, if so, identify the exact target.

## 6. Whether extra analysis is required before final rebuttal

Current manuscript posture: router analysis, sensitivity analysis, score-treatment analysis, comparator baselines, and downstream clinical/LLM validation are limitation-framed or future work.

Recommendation: do not run new experiments for ASAP submission unless the supervisor decides limitation framing is insufficient.

Decision needed: confirm whether current limitation framing is sufficient for final rebuttal.
