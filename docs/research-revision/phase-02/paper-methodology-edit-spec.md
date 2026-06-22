# Paper Methodology Edit Specification — Phase 02 Consolidation

## 1. Executive summary

This is the paper-editing handoff for approved Phase 02 evidence. It prescribes corrections to methodology-related claims without editing the manuscript or supplying replacement prose. Future editing may retain documented implementation and protocol facts only when their stated scope, provenance, and limitations are preserved.

## 2. Scope of future paper edits

The future editing sprint must correct retrieval/indexing, score interpretation, evaluation protocols, sidecar terminology/provenance, router behavior, corpus/index-lineage framing, unsupported tuning/supplement claims, and overbroad clinical/deployment implications. It must not create results, resolve analysis gaps by assertion, or publish restricted materials.

## 3. Section-by-section edit plan

| Paper section | Issue | Required correction | Evidence source | Safe framing | Unsafe framing | Blocked by | Future sprint or phase |
|---|---|---|---|---|---|---|---|
| Abstract | Broad efficacy, graph, and controlled-result framing | Bound the 96.0%/52.5% result to controlled/simulated fixture-score evaluation; remove graph/deployment implications. | Phase 01 claim matrix; Sprints 2.2–2.4 | Protocol-bound mechanism result. | General clinical utility, graph architecture, deployment readiness. | None for scoped revision. | Gated paper editing. |
| Introduction | Contribution/novelty and headline validation overreach | Keep the problem/mechanism motivation; soften broad novelty, safety, and MIMIC-wide validation language; rename Virtual Graph. | Claim matrix C01–C08; Sprint 2.4 decision record | Intent-modulated temporal reranking over a local protocol. | Validated graph retrieval or clinical safety solution. | Strong novelty claims need supervisor evidence decision. | Gated paper editing. |
| Methods / Methodology | Index metric, cosine, router, sidecar, scoring, graph terminology | Replace IndexFlatIP with IndexFlatL2/METRIC_L2; distinguish conditional ranking equivalence from raw-score meaning; specify heuristic router/default behavior; use temporal metadata sidecar; define fixed scoring settings. | Sprints 2.2–2.4 | Implementation facts with stated provenance and protocol limits. | Raw L2 as cosine, trained/calibrated router, observed time, graph traversal. | Phase 9 score interpretation remains limited. | Gated paper editing; future score analysis. |
| Experimental Setup | Controlled fixture framing, corpus/chunking scope, tuning claim | Identify hard-negative inputs as controlled/simulated and scenario-specific for equal 0.95 scores; retain local corpus counts with lineage caveat; remove tuning claim. | Sprints 2.1–2.3; Phase 01 manifest | Controlled mechanism-isolation protocol and local processed subset. | Naturalistic benchmark, fully proven current-index build lineage, tuned settings. | Exact index-build lineage unresolved. | Gated paper editing; future provenance analysis if stronger claim needed. |
| Results | Unqualified aggregates and baseline conflation | Label 96.0%/52.5% controlled/simulated; label 58%/80% versus 22%/69% as local target-subject-filtered and score-treatment-limited; define semantic-only separately by protocol. | Sprints 2.3 and 2.2 | Exact protocol-bound aggregates. | Unconstrained or standard cosine-baseline result. | Phase 9 score treatment; missing comparators. | Gated paper editing; future analysis for stronger inference. |
| Discussion / Limitations | Causal, safety, router, comparator, and scale overclaims | State router evidence gaps, score-treatment limitation, local scale, missing baselines, Negation Recency null result, and Terminology Drift uncertainty; avoid causal attributions. | Phase 01 reviewer mapping; Sprints 2.3–2.4 | Limitations and future-analysis needs. | Validated safety, router reliability, broad superiority, causal explanation. | Router and comparator analyses absent. | Gated paper editing; future analysis. |
| Conclusion | Deployment/unconstrained/graph claims | Restate only protocol-bound mechanism findings and future validation needs; remove deployment and graph-architecture claims. | Claim matrix C20–C22; Sprints 2.2–2.4 | Local controlled and filtered findings. | Clinical deployment readiness or unqualified robustness. | None for scoped revision. | Gated paper editing. |
| Declarations / Supplementary Material | Unsupported availability statement | Remove or defer claims that sidecar, patterns, data, and scripts are provided as a supplement. | Phase 01 manifest; Sprint 2.1 truth table | Availability pending a reviewed sanitized release package. | Existing public/reviewed supplement availability. | Sanitized package and release approval absent. | Future supplement-release sprint. |

## 4. Claims to retain

- `IndexFlatL2` / `METRIC_L2` retrieval over normalized vectors in inspected paths.
- TIMER’s local formula and fixed alpha, beta, lambda, and tau settings.
- Negative historical beta as a penalty on recent candidates relative to equal semantic inputs.
- The controlled/simulated 96.0% versus 52.5% aggregate for its exact fixture/evaluator protocol.
- The local target-subject-filtered Phase 9 58%/80% versus 22%/69% aggregate, with its score-treatment limitation.
- Rule-based Current/Historical/Trend routing as an implementation detail.
- A generated local temporal metadata sidecar as an implementation detail.

## 5. Claims to correct

- IndexFlatIP/exact-inner-product descriptions: correct to `IndexFlatL2` / `METRIC_L2`.
- Raw L2 output described as cosine or inner-product similarity: remove that equation-to-output equivalence.
- Unqualified global top-50 description for Phase 9: specify full local-index search, target-subject filtering, then first 50 filtered candidates.
- Universal neutral handling for unmatched router inputs: zero matches default to Current at confidence 0.5 in the inspected implementation.
- Source-observed sidecar timing: identify generated/randomized offsets and placeholder date.

## 6. Claims to soften

- Empirical validation, clinical utility, safety, and deployment applicability.
- Novelty claims around negative beta and graph-like architecture.
- Interpretation of scenario outcomes, including null and small-sample findings.
- Comparative claims beyond the evaluated semantic-only variants.
- Corpus-scale and chunking-sophistication implications.

## 7. Claims to remove

- Separate 20-query tuning-set / held-fixed provenance.
- Existing supplementary-material availability.
- Graph architecture, nodes/edges/traversal, and graph-retrieval novelty claims.
- Claims of trained, accurate, calibrated, safe, or validated router behavior.
- “Unconstrained,” “deployment-ready,” or equivalent Phase 9 framing.

## 8. Claims to rename/reframe

- Rename “Virtual Graph” to **temporal metadata sidecar**.
- Reframe sidecar temporal inputs as generated/simulated local metadata, not observed clinical chronology.
- Reframe router confidence as a heuristic match-share where matches exist, not a calibrated probability.
- Reframe Phase 9 as a local target-subject-filtered ablation with a documented score-treatment limitation.

## 9. Claims blocked by future analysis

| Claim area | Status | Required evidence before positive claim |
|---|---|---|
| Phase 9 semantic-score validity or temporal-effect magnitude | `REQUIRES_FUTURE_ANALYSIS` / possible code decision | Approved score analysis; code/reproduction decision only if analysis establishes need. |
| Router accuracy, calibration, trigger/fallback rates, and near misses | `REQUIRES_FUTURE_ANALYSIS` | Sanitized standalone router evaluation. |
| Comparator superiority | `REQUIRES_FUTURE_ANALYSIS` | Approved comparator evaluation. |
| Tuning/sensitivity | `REMOVE_OR_DEFER` | Documented selection and sensitivity analysis. |
| Exact current-index build lineage | `REQUIRES_FUTURE_ANALYSIS` | Sanitized provenance record. |
| Public supplement availability | `REMOVE_OR_DEFER` | Reviewed sanitized package and release decision. |

## 10. Safe wording constraints

- State implementation, protocol, and limitation facts separately.
- Keep conditional normalized-vector ranking observations distinct from the downstream Phase 9 score transform.
- Label hard-negative inputs/results controlled/simulated and fixture-score-based.
- Label Phase 9 local, target-subject-filtered, and score-treatment-limited.
- Treat beta/tau as fixed local settings, not tuned or validated choices.
- Describe the sidecar and router only at their supported implementation level.

## 11. Unsafe wording list

- Exact inner-product index; raw L2 equals cosine; validated distance-to-similarity conversion.
- Unconstrained Phase 9 retrieval; deployment-ready clinical decision support.
- Observed temporal provenance; graph nodes, edges, traversal, or graph reasoning.
- Calibrated/accurate/safe router or universally neutral unmatched routing.
- Tuned development-set parameters or available supplement without approved evidence.

## 12. Reviewer concern mapping

| Concerns | Editing direction |
|---|---|
| P08 | Replace Virtual Graph framing and disclose generated sidecar provenance. |
| P11–P13 | Limit fixed-score language to applicable controlled scenarios. |
| P14 | Report the filtered Phase 9 protocol with its limitations. |
| P15–P16 | Report null/small-sample findings without causal or safety claims. |
| P19–P21 | Limit router discussion to heuristic behavior and disclose absent analysis. |
| P23–P26 | Remove tuning claim; mark sensitivity as future work. |
| P28 | Retain corpus counts with local-scale limitation. |
| P30–P31 | Do not claim superiority beyond evaluated semantic-only variants. |

## 13. Required future-analysis caveats

Paper editing may describe the open score-treatment, router, comparator, sensitivity, and lineage issues as limitations. It may not convert them into resolved method claims. A future score-analysis/code-decision sprint is needed before Phase 9 is used as evidence about semantic-score validity or temporal-effect magnitude; router analysis is needed before router-quality claims; supplement work is needed only if availability is to be asserted.

## 14. Paper-editing readiness decision

**Ready for a gated paper-editing phase with hard constraints.** Phase 02 maps the methodology mismatches sufficiently for correction, removal, and limitation framing. Editing is not authorized by this document and must remain blocked from stronger claims listed in Section 9.
