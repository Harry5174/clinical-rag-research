# Final Working Decisions Before Submission -- Sprint 4.2

## Recommendation for ASAP submission

For ASAP submission, keep the `TIMER-Graph` title/name, keep limitation framing, do not run new experiments, and use the local/Overleaf export handoff path unless the supervisor later requests a different route.

## 1. Title decision

Final working decision: keep `TIMER-Graph` title/name.

Rationale: the manuscript body now clarifies that `TIMER-Graph` is retained as the method/system name while the implementation is described as a temporal metadata sidecar, not a graph database, node-edge graph, graph traversal system, or observed clinical temporal graph.

## 2. Code availability decision

Final working decision: use conservative restricted-data-safe wording.

Applied manuscript posture: the implementation code is not publicly released with this submission. A sanitized version may be made available from the corresponding author upon reasonable request after institutional and public-release review. Restricted clinical data, raw MIMIC-IV-Note text, processed note text, indexes, and restricted-data-derived artifacts are not redistributed.

## 3. Supplementary material decision

Final working decision: no supplementary material is submitted unless the supervisor later supplies an approved package.

Applied manuscript posture: supplementary file availability is not asserted. No supplement is invented in this sprint.

## 4. Source-tracking / export decision

Final working decision: do not modify `.gitignore`, do not force-add ignored LaTeX files, do not stage, commit, or push. Use local/Overleaf export handoff for ASAP submission.

Handoff posture: revised LaTeX source remains in `latex_publication/v1/`; temporary handoff artifacts are copied outside the repository for supervisor review.

## 5. P38 decision

Final working decision: rebuttal-only / no manuscript target identified / no new unsupported claim.

Response posture: we reviewed the manuscript for weighted-scheduling / heterogeneous-workload positioning. No active claim requiring a separate manuscript correction was identified in the revised manuscript, so no manuscript edit is made and no new experiment is introduced.

## 6. Extra-analysis decision

Final working decision: no extra analysis is required before ASAP submission under the current working decision.

Unresolved items remain limitation-framed or future work, including score-treatment analysis, router analysis, confidence distribution, regex near-miss analysis, sensitivity analysis, STAR-RAG-style baseline, simple time-filter baseline, additional failure analysis, and new reproduction.
