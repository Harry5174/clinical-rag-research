# Final Manuscript Consistency Audit -- Sprint 3.5

## 1. Executive summary

Sprint 3.5 completed a manuscript-wide consistency audit of `latex_publication/v1/sn-article.tex` against Phase 01, Phase 02, and prior Phase 03 evidence. Minimal consistency edits were applied to Related Work, one Experimental Setup limitation paragraph, and selected captions. Reported numeric results, citations, bibliography, tables values, figures, author list, affiliations, code, data, indexes, results, tests, and notebooks were not changed.

Submission-readiness status: ready for Implementation Supervisor review, with supervisor decisions still needed for final source tracking/export and exact journal-specific supplement/code availability policy.

## 2. Full-manuscript sections checked

- Title and author/front matter
- Abstract
- Introduction
- Related Work
- Proposed Methodology
- Experimental Setup
- Results
- Discussion
- Conclusion
- Supplementary information
- Declarations, including data and code availability
- All visible table and figure captions

## 3. High-risk search terms checked

The audit searched for outdated or high-risk wording including `IndexFlatIP`, exact inner product, raw L2/cosine equivalence, validated semantic score, Virtual Graph, graph nodes/edges/traversal, trained or calibrated router claims, deployment/production/clinical-validation claims, broad comparator claims, full-MIMIC/population-scale claims, supplement/code release claims, unsupported tuning claims, and confidence-interval language.

## 4. Remaining old-claim hits

Remaining high-risk hits were inspected and classified as contextual rather than newly unsupported:

- Graph terms appear in negating language such as "does not implement graph nodes, graph edges, graph traversal, or a graph database substitute."
- Comparator terms appear in missing-baseline limitations.
- "Full MIMIC-IV-Note corpus" appears only to distinguish a mining source from the local 1,206-chunk indexed benchmark.
- Confidence intervals are pre-existing manuscript statistics; no intervals were invented or changed in Sprint 3.5.

## 5. Corrections applied

| Paper location | Old claim summary | New claim summary | Evidence/spec used | Reason for change | Reviewer concern addressed | Risk reduced | Remaining limitation |
| --- | --- | --- | --- | --- | --- | --- | --- |
| Related Work opening | Implied a singular absence across existing systems. | Frames the gap as limited attention to query-intent-conditioned temporal modulation. | Phase 02 methodology specs; Sprint 3.4 framing. | Avoid overbroad novelty claim. | P30-P31; P34-P36 | Reduces unsupported novelty/comparator risk. | Literature positioning remains supervisor-reviewable. |
| Related Work / RAG in Healthcare | Called prior systems production-grade and high-quality. | Describes retrieval pipelines and semantic search more neutrally. | Phase 01/02 overclaim guidance. | Avoid deployment-grade implication in final manuscript pass. | P28; P34-P36 | Reduces production/deployment overclaim risk. | No citation set was changed. |
| Related Work / Graph-Based Clinical RAG | Stated none of the systems incorporate temporal metadata into graph structures. | States these examples do not center query-intent-conditioned temporal metadata as the retrieval modulation mechanism. | Phase 02 sidecar/graph correction spec. | Avoid absolute comparative claim. | P08; P30-P31 | Reduces broad graph/comparator overclaim risk. | Empirical comparator baselines remain missing. |
| Related Work / Temporal-Aware Retrieval | Characterized STAR-RAG filtering as rigid and clinically critical outlier loss. | Softens to temporal filtering that may exclude temporally distant records relevant for some longitudinal questions. | Reviewer mapping; missing comparator limitation. | Avoid unsupported superiority claim over STAR-RAG. | P30-P31 | Reduces unsupported comparator claim. | STAR-RAG-style baseline not implemented. |
| Related Work / Positioning | Said negative beta boosts historical documents and prefers older documents when clinically appropriate. | States it can favor older candidates under historical-intent scoring when a query is classified as historical. | TIMER scoring correction spec. | Align mechanism wording with fixed local scoring evidence. | P23-P26; P30-P31 | Reduces clinical-appropriateness and broad mechanism overclaim. | Sensitivity and comparator analyses remain future work. |
| Figure 2 caption | Generic architecture caption. | Names retrieval pipeline with temporal metadata sidecar. | Temporal sidecar correction spec. | Caption consistency with body terminology. | P08 | Reduces Virtual Graph/architecture ambiguity. | Figure file unchanged. |
| Experimental Setup limitation paragraph | Said semantic scores were fixed at 0.95 across hard-negative pairs and validated the mechanism. | States several scenarios use supplied scores, including equal 0.95 fixtures in Semantic Collision and Real-World Mining, and that the design tests the mechanism in isolation. | Evaluation protocol correction spec; TIMER scoring spec. | Correct blanket fixed-score claim. | P11-P13; P15-P16 | Reduces score-treatment and controlled-protocol overclaim. | Controlled protocol remains limited. |
| Figure 8 caption | Generic per-scenario accuracy comparison. | Controlled/simulated per-scenario retrieval accuracy comparison. | Results/evaluation framing traceability. | Caption consistency. | P11-P16 | Reduces broad result framing. | Figure file unchanged. |
| Figure 11 caption | Generic per-query heatmap across all 200 queries. | Per-query heatmap across the controlled 200-query hard-negative protocol. | Results/evaluation framing traceability. | Caption consistency. | P11-P16; P34-P36 | Reduces broad per-query interpretation. | Figure file unchanged. |

## 6. Numbers consistency check

The required number search found the expected manuscript values unchanged: 96.0%, 52.5%, 58.0%, 80.0%, 22.0%, 69.0%, 165 patients, 1,206 chunks/vectors, and Terminology Drift `n=20`. No reported numeric result, table value, figure value, p-value, effect size, or confidence interval was changed.

## 7. Protocol framing consistency check

The manuscript consistently frames the hard-negative evaluation as controlled/simulated and mechanism-isolation oriented. Phase 9 is consistently framed as local, target-subject-filtered, and score-treatment-limited. Semantic-only baselines are separated by protocol, and missing STAR-RAG-style/simple time-filter comparators are stated as limitations.

## 8. Terminology consistency check

The manuscript uses `IndexFlatL2` / `METRIC_L2`, normalized-vector ranking caveats, transformed L2-derived Phase 9 score limitation, temporal metadata sidecar, fixed regex/rule-based router, fixed local settings, and generated/randomized/placeholder sidecar provenance. Remaining graph terminology is used only for related-work context or to deny graph implementation.

## 9. Availability / declarations consistency check

Supplementary file availability is not asserted. Data availability points to credentialed PhysioNet access for MIMIC-IV-Note and MIMIC-IV. Code availability requires a separate supervisor decision and approved public-release review. These statements avoid inventing supplement files, public code release timing, public raw-data release, or raw MIMIC-IV-Note sharing.

## 10. Related Work and caption consistency check

Related Work was edited to soften unsupported novelty/comparator/deployment-adjacent wording. Captions were checked across figures and tables. Minimal caption edits were applied only where they improved final consistency with controlled/simulated or temporal-sidecar framing.

## 11. Reviewer concern coverage audit

- P08: Addressed through temporal metadata sidecar terminology and graph-behavior denial; title retains TIMER-Graph as a label.
- P11-P13: Addressed through controlled/supplied-score framing and corrected fixed-0.95 limitation wording.
- P14: Addressed through prominent local target-subject-filtered Phase 9 reporting.
- P15: Addressed as no observed gain / saturation in Discussion.
- P16: Addressed as `n=20` caution in Discussion and Limitations.
- P19-P21: Addressed as router evidence gaps, fixed regex/rule-based mechanism, and zero-match caveat.
- P23-P26: Addressed as fixed settings with missing tuning/sensitivity/calibration evidence.
- P28: Addressed as local processed subset and non-production-scale limitation.
- P30-P31: Addressed as missing STAR-RAG-style and simple time-filter comparator baselines.
- P34-P36: Addressed by limitation/failure-analysis framing without new analysis.
- P38: No matching manuscript target identified in prior mapping; remains supervisor clarification if still relevant.

## 12. Remaining supervisor decisions

- Final source-tracking/export decision for ignored `latex_publication/`.
- Whether to retain title as-is or request a later title-specific supervisor decision.
- Journal-specific final wording for supplement and code availability.
- Whether a future score-analysis/code-decision sprint is needed before any stronger Phase 9 semantic-score claim.
- Whether future comparator/router/sensitivity analyses will be run before reviewer-response drafting.

## 13. Submission-readiness status

Ready for Implementation Supervisor review, with the open decisions above recorded rather than expanded in scope.
