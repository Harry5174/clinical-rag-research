# Sprint 3.4 Frontmatter, Conclusion, and Declarations Edit Traceability

## Scope

This traceability record covers Sprint 3.4 edits to approved high-visibility manuscript sections only:

- Abstract
- Introduction
- Conclusion
- Supplementary information statement
- Declarations: Code availability only

The title was inspected but not edited. Methods, Results/Evaluation, Discussion/Limitations, tables, figures, bibliography, code, tests, notebooks, data, indexes, and result files were not edited.

## Traceability Matrix

| Paper location | Old claim summary | New claim summary | Evidence/spec used | Reason for change | Reviewer concern if applicable | Risk reduced | Remaining limitation |
| --- | --- | --- | --- | --- | --- | --- | --- |
| Abstract | Presented TIMER-Graph with broad clinical-decision framing, Virtual Graph wording, and the 96.0% hard-negative result as the primary headline. | Frames TIMER-Graph as a prototype using fixed rule-based intent categories, a local temporal scoring rule, and a temporal metadata sidecar; reports both controlled hard-negative and local Phase 9 results as bounded evidence. | Phase 01 authoritative results manifest; Phase 01 correction matrix; Phase 02 methodology/evaluation/scoring specs; Sprint 3.1-3.3 traceability. | Align high-visibility summary with reproduced and bounded evidence. | Addresses overclaim and evaluation-validity concerns. | Reduces risk of implying deployment readiness, clinical validation, comprehensive comparator coverage, or resolved semantic-score validity. | Abstract still cannot fully detail all limitations; final consistency sprint should check length and journal fit. |
| Introduction opening | Used stronger safety/decision-support motivation that could imply direct clinical-readiness relevance. | Keeps the longitudinal EHR retrieval motivation but narrows it to tracking historical versus current context. | Reviewer concern mapping; Phase 02 methodology exit gate; Sprint 3.3 limitations framing. | Preserve motivation while avoiding unsupported clinical utility claims. | Addresses clinical-deployment overclaim concern. | Reduces risk of overstating current evidence as patient-safety validation. | Broader clinical context remains literature-facing and should be checked in final consistency review. |
| Introduction recency-bias framing | Framed recency bias more broadly and with stronger clinical consequence wording. | Defines the Recency Bias Trap as a retrieval pattern where recent notes are favored even when historical information is needed. | Paper claim correction matrix; evaluation protocol correction spec. | Make the failure mode conceptual and retrieval-specific. | Addresses concern that the paper generalized beyond measured tasks. | Reduces risk of implying broad clinical error measurement. | Figure caption remains unchanged because figure/caption edits were outside the Sprint 3.4 boundary. |
| Introduction contribution list | Described TIMER-Graph as including Virtual Graph architecture and broader empirical validation. | Describes a temporal metadata sidecar, no graph nodes/edges/traversal, controlled/simulated hard-negative protocol, and local target-subject-filtered Phase 9 ablation; names missing comparator, router, sensitivity, and deployment analyses as future work. | Phase 02 temporal-sidecar/virtual-graph correction spec; intent-router correction spec; score-treatment decision record; Sprint 3.1 methodology traceability; Sprint 3.2 results traceability. | Align the contribution summary with implemented and inspected system behavior. | Addresses architecture and comparator concerns. | Reduces risk of implying graph traversal, learned router quality, or comprehensive baseline comparison. | Title still contains TIMER-Graph but was not approved for editing. |
| Conclusion | Drew stronger conclusions about bias reduction and clinical impact from the controlled hard-negative result. | Presents TIMER-Graph as a prototype and reports both hard-negative and Phase 9 results as protocol-bound; explicitly excludes unconstrained retrieval, resolved L2-derived score validity, comprehensive comparator superiority, clinical deployment readiness, and production-scale validation. | Phase 01 authoritative results manifest; Phase 02 scoring and retrieval-indexing correction specs; Sprint 3.2 results framing; Sprint 3.3 limitations framing. | Bring final manuscript takeaway into alignment with corrected Results and Discussion. | Addresses broad-superiority and deployment-readiness concerns. | Reduces risk of unsupported concluding claims. | Future work remains declarative and should be harmonized during final consistency review. |
| Supplementary information | Asserted supplementary files were available. | States supplementary file availability is not asserted in this revision and requires separate supervisor approval. | Phase 01 correction matrix; Sprint 3.4 amendment allowing removal of unsupported supplementary claims. | Avoid unsupported supplementary-material availability claim. | Addresses evidence-availability concern. | Reduces risk of promising unavailable or unreviewed supplementary artifacts. | Final journal submission may require supervisor decision on exact supplementary statement. |
| Declarations: Code availability | Promised code would be made available upon publication at a public repository. | States code release requires separate supervisor decision and approved public-release review. | Phase 01 correction matrix; Sprint 3.4 amendment; project data-handling/public-release policy. | Avoid unapproved public-release commitment. | Addresses artifact-availability and public-release control concerns. | Reduces risk of committing to release unreviewed code or sensitive-adjacent artifacts. | Exact public release policy remains a later supervisor decision. |

## Deferred Items

- Title review is deferred because Sprint 3.4 permitted inspection but not editing of the title.
- Related Work, Methods/Methodology, Results/Evaluation, Discussion/Limitations, table captions, and figure captions were not edited in this sprint.
- Final cross-section consistency should verify that frontmatter, body sections, and declarations use consistent terminology after all Phase 03 edits.

## Diff Evidence

Temporary manuscript comparison files:

- Before snapshot: `/tmp/clinical-rag-paper-edit-review/sprint-3-4/sn-article.before.tex`
- After snapshot: `/tmp/clinical-rag-paper-edit-review/sprint-3-4/sn-article.after.tex`
- Unified diff: `/tmp/clinical-rag-paper-edit-review/sprint-3-4/sprint-3-4.diff`
