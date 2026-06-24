# Final Decision Application Record -- Sprint 4.2

## 1. Executive summary

Sprint 4.2 applied the final working decisions for ASAP submission. The title/name is retained, availability wording is made conservative and restricted-data-safe, P38 is handled as rebuttal-only with no manuscript target identified, no extra analysis is required before submission, and local/Overleaf export handoff is documented without Git tracking changes.

## 2. Decisions applied

- Keep `TIMER-Graph` title/name.
- Use conservative Code availability wording.
- Do not assert or invent supplementary material.
- Use local/Overleaf export handoff; do not modify `.gitignore` or force-add ignored files.
- Handle P38 as reviewed / no manuscript target identified / no manuscript edit required.
- Run no extra analysis before ASAP submission.

## 3. Title decision

Decision: keep `TIMER-Graph` title/name.

The title was not edited. The response posture is body clarification: the manuscript describes a temporal metadata sidecar implementation and does not claim graph database, node-edge graph, graph traversal, or observed clinical temporal graph behavior.

## 4. Code availability decision

Decision: use conservative restricted-data-safe wording.

The manuscript Code availability statement was updated to state that implementation code is not publicly released with this submission, that a sanitized version may be made available upon reasonable request after institutional and public-release review, and that restricted clinical data, raw MIMIC-IV-Note text, processed note text, indexes, and restricted-data-derived artifacts are not redistributed.

## 5. Supplementary material decision

Decision: no supplementary material is submitted unless a later approved package is supplied.

The current manuscript did not assert supplementary availability, so no supplementary-material manuscript edit was needed. No supplement was created.

## 6. Source-tracking / export decision

Decision: use local/Overleaf export handoff for ASAP submission.

No `.gitignore` change, force-add, staging, commit, or push was performed. Because `latex_publication/` is ignored, Sprint 4.2 uses temporary before/after manuscript snapshots and a temporary handoff package outside the repository for supervisor review.

## 7. P38 decision

Decision: reviewed / no manuscript target identified / no manuscript edit required.

The approved search did not identify a clear active manuscript claim about weighted scheduling, heterogeneous workloads, DC2 anomaly, queues, or schedulers. The rebuttal is updated as rebuttal-only and does not invent an unsupported experiment or correction.

## 8. Extra-analysis decision

Decision: no extra analysis is required before ASAP submission under the current working decision.

Score-treatment, router, confidence-distribution, regex near-miss, sensitivity, comparator-baseline, failure-analysis, and reproduction work remain limitation-framed or future work.

## 9. Manuscript edits made, if any

One manuscript edit was made in `latex_publication/v1/sn-article.tex`: the Code availability item was replaced with the approved conservative restricted-data-safe wording.

No title, author, affiliation, bibliography, numeric result, table value, figure content, citation, data, index, result file, code, test, or notebook edits were made.

## 10. Rebuttal updates made

- `major-comments-rebuttal-draft.md` now records final working decisions, removes title/P38 supervisor-decision language, and keeps no-extra-analysis wording.
- `rebuttal-traceability-to-manuscript.md` now records P08 as clarified in manuscript and P38 as reviewed / no manuscript target identified.
- `open-supervisor-decisions-before-submission.md` was converted into a final working decision record.

## 11. Remaining optional supervisor preferences

- Supervisor may still choose to compile/export in Overleaf or another LaTeX environment.
- Supervisor may later request public-release packaging, supplementary material, or additional analysis, but none is required under the current ASAP submission decision.
