# Submission Readiness Checklist -- Sprint 3.5

## Manuscript source

- [x] Full manuscript structure inspected.
- [x] Title inspected.
- [x] Abstract, Introduction, Related Work, Methods, Experimental Setup, Results, Discussion, Conclusion, Supplementary information, and Declarations checked.
- [x] Minimal consistency edits applied only to approved paper source.
- [ ] Final source-tracking/export method requires supervisor decision because `latex_publication/` is ignored by Git.

## Claim safety

- [x] Broad deployment, production-scale, clinical-validation, and comprehensive-superiority claims checked.
- [x] Remaining high-risk terms inspected in context.
- [x] Related Work overbroad wording softened without adding or removing citations.
- [x] No new claim, result, citation, experiment, or baseline was added.

## Result consistency

- [x] Required values checked: 96.0%, 52.5%, 58.0%, 80.0%, 22.0%, 69.0%, 165, 1,206, and Terminology Drift `n=20`.
- [x] Table values unchanged.
- [x] Figure contents unchanged.
- [x] No reported numeric result changed.

## Methodology consistency

- [x] `IndexFlatL2` / `METRIC_L2` wording retained.
- [x] Normalized-vector caveat retained.
- [x] Raw L2 / cosine equivalence avoided.
- [x] Temporal metadata sidecar wording retained.
- [x] Generated/placeholder sidecar provenance retained.
- [x] Fixed regex/rule-based router wording retained.

## Evaluation consistency

- [x] Controlled/simulated hard-negative framing retained.
- [x] Supplied-score hard-negative semantic-only baseline retained.
- [x] Phase 9 local target-subject-filtered framing retained.
- [x] Phase 9 score-treatment limitation retained.
- [x] Missing comparator baselines retained as limitations.

## Discussion / limitations consistency

- [x] Negation Recency no-gain framing retained.
- [x] Terminology Drift `n=20` caution retained.
- [x] Router, sensitivity, score-treatment, corpus-scale, sidecar-provenance, and deployment-validation limitations retained.
- [x] No causal failure analysis was invented.

## Declarations / availability

- [x] Supplementary file availability is not asserted.
- [x] Code release remains supervisor-decision gated.
- [x] Raw MIMIC-IV-Note sharing is not claimed.
- [x] Data availability points to credentialed PhysioNet access.
- [ ] Journal-specific final availability wording remains a supervisor decision.

## Restricted-data safety

- [x] No raw clinical-note text printed.
- [x] No raw MIMIC rows printed.
- [x] No raw query text, answers, retrieved chunks, raw sidecar records, or raw result rows printed.
- [x] No external APIs or services used.

## Compile / export readiness

- [x] Local LaTeX files inspected.
- [x] `latexmk` availability checked.
- [x] No package install, download, class/style edit, bibliography edit, citation removal, or content-for-compile edit performed.
- [ ] Local compile not run because `latexmk` was unavailable.
- [ ] Final export path requires supervisor decision.

## Reviewer-response readiness

- [x] Reviewer concern traceability created.
- [x] Manuscript locations and limitation framing mapped.
- [x] No final rebuttal text drafted.
- [ ] Reviewer-response sprint remains separate and not yet started.

## Supervisor decisions still needed

- [ ] Source tracking/export handling for ignored LaTeX source.
- [ ] Final title decision if supervisor wants a title change beyond body reframing.
- [ ] Journal-specific supplement/code availability wording.
- [ ] Whether to run future comparator, router, sensitivity, score-treatment, or source-tracking closeout sprints.
