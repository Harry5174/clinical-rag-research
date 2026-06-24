# Major Comments Rebuttal Draft -- Sprint 4.1

## 1. Rebuttal cover note

This draft addresses major comments only. The response strategy is to explain what was corrected in the manuscript, what was clarified, what was limitation-framed, and what remains future work or requires supervisor decision. The draft does not claim new experiments, new baselines, new router analysis, sensitivity analysis, score-treatment validation, clinical validation, or deployment validation.

## 2. Summary of revision strategy

The manuscript was revised to align major claims with the available evidence. The core strategy was:

- retain `TIMER-Graph` as the method/system name while clarifying that the implementation uses a temporal metadata sidecar, not graph traversal or graph-database behavior;
- present the hard-negative result as controlled/simulated fixture-score mechanism-isolation evidence;
- present Phase 9 as a reproduced local target-subject-filtered ablation with score-treatment limitations;
- convert missing evidence areas into explicit limitations or future work rather than unsupported claims;
- avoid public availability, deployment, clinical-validation, or comparator-superiority claims that are not supported by the current evidence.

## 3. Major comment responses

### P08 -- Virtual Graph / graph overclaim

Comment:
The `Virtual Graph` terminology may overstate the implementation and imply graph nodes, edges, traversal, graph database behavior, or observed clinical temporal graph provenance.

Response:
Thank you for the comment. We have revised the manuscript to clarify the implementation. Specifically, the manuscript now describes the mechanism as a temporal metadata sidecar that supplies generated local temporal-scoring inputs adjacent to the vector index. We retained `TIMER-Graph` as the method/system name, but we do not defend it as a graph database or graph-traversal architecture. We also avoid claiming graph nodes, graph edges, traversal, graph database behavior, or observed clinical temporal graph provenance.

Manuscript location:
Abstract; Introduction; Proposed Methodology / Temporal Metadata Sidecar; Discussion / Limitations; Conclusion.

Status:
Addressed with limitation.

### P11-P13 -- Fixed `Ssem` / controlled-score caveat

Comment:
The hard-negative evaluation uses supplied semantic scores, including equal-score fixtures, so the result should not be framed as live natural-score retrieval evidence.

Response:
Thank you for the comment. We have revised the manuscript to clarify that the hard-negative evaluation is a controlled/simulated fixture-score protocol. Specifically, the manuscript now distinguishes supplied fixture scores from live semantic retrieval and presents the 96.0% result as mechanism-isolation evidence under controlled hard-negative conditions. We avoid claiming that this result establishes deployment-realistic retrieval or natural-score retrieval quality.

Manuscript location:
Experimental Setup / Hard Negative Stress Test; Experimental Setup / Limitations of the Controlled Evaluation and Path to Realistic Testing; Results / Overall Performance; Discussion / Limitations; Conclusion.

Status:
Addressed with limitation.

### P14 -- Phase 9 result prominence and framing

Comment:
The Phase 9 end-to-end result should be more visible and clearly framed, rather than being overshadowed by the controlled 96.0% hard-negative result.

Response:
Thank you for the comment. We have revised the manuscript to make the Phase 9 result more visible and to state its protocol boundary. Specifically, the manuscript now reports the reproduced local target-subject-filtered Phase 9 ablation result, including 58.0% Accuracy@1 and 80.0% Recall@5 for TIMER-Graph Full versus 22.0% and 69.0% for the Phase 9 semantic-only ablation. We also clarify that this is not an unconstrained or deployment-realistic retrieval result.

Manuscript location:
Abstract; Results / End-to-End Retrieval and Ablation Study; Discussion / Limitations; Conclusion.

Status:
Addressed with limitation.

### P15 -- Negation Recency no-gain

Comment:
The Negation Recency scenario shows no gain because both systems are already at 100%, so the manuscript should not claim improvement for this scenario.

Response:
Thank you for the comment. We have corrected the interpretation. Specifically, the manuscript now describes Negation Recency as scenario saturation and no observed gain, with both systems at ceiling performance in the controlled fixture. We avoid claiming improved negation handling or clinical safety from this scenario.

Manuscript location:
Discussion / Per-Scenario Analysis; Discussion / Limitations.

Status:
Addressed.

### P16 -- Terminology Drift `n=20` caution

Comment:
The Terminology Drift result has only `n=20` and should be treated cautiously.

Response:
Thank you for the comment. We have revised the manuscript to add caution around the small scenario size. Specifically, the Terminology Drift result is now framed as preliminary and uncertainty-sensitive rather than a stable scenario-level conclusion. We did not invent new confidence intervals or claim a completed failure analysis.

Manuscript location:
Discussion / Per-Scenario Analysis; Discussion / Limitations.

Status:
Addressed with limitation.

### P19-P21 -- Router accuracy, confidence distribution, and regex near misses

Comment:
The manuscript should not imply validated router accuracy, calibrated confidence, trigger-rate knowledge, or regex near-miss analysis without evidence.

Response:
Thank you for the comment. We have revised the manuscript to clarify that the router is fixed rule-based / regex-based. Specifically, the manuscript describes confidence as a heuristic match-share rather than a calibrated probability, and it identifies router accuracy, calibration, confidence distribution, trigger/fallback rates, zero-match and tie behavior, and regex near misses as limitations or future work. We avoid claiming a trained classifier, validated accuracy, or calibrated confidence.

Manuscript location:
Proposed Methodology / Intent Classification Router; Discussion / Why TIMER-Graph Works; Discussion / Limitations; Conclusion.

Status:
Addressed with limitation.

### P23-P26 -- Beta, lambda, tau, and sensitivity

Comment:
The manuscript should not imply that alpha, beta, lambda, or tau were tuned, calibrated, or sensitivity-tested without evidence.

Response:
Thank you for the comment. We have revised the manuscript to clarify these values as fixed local implementation/evaluation settings. Specifically, the manuscript no longer relies on tuning or calibration implications and identifies sensitivity analysis as future work. We avoid claiming parameter robustness or completed sensitivity analysis.

Manuscript location:
Proposed Methodology / TIMER Scoring Function; Experimental Setup / Implementation Details; Discussion / The Role of Negative Beta; Discussion / Limitations; Conclusion.

Status:
Addressed with limitation.

### P28 -- Corpus scale / production-scale limitation

Comment:
The manuscript should not imply production-scale or full-corpus validation from a local processed subset.

Response:
Thank you for the comment. We have revised the manuscript to clarify the scale and provenance limits. Specifically, the manuscript now describes the evaluated corpus as a local processed subset and records the inspected local index size and current-index lineage caveat. We avoid claiming production-scale, population-scale, full-MIMIC deployment, clinical utility, or deployment readiness.

Manuscript location:
Experimental Setup / Dataset; Results / End-to-End Retrieval and Ablation Study; Discussion / Clinical Implications; Discussion / Limitations; Conclusion.

Status:
Addressed with limitation.

### P30-P31 -- Missing comparator baselines

Comment:
The empirical comparison lacks STAR-RAG-style and simple time-filter comparator baselines, so semantic-only baselines should not be presented as comprehensive comparator coverage.

Response:
Thank you for the comment. We have revised the manuscript to acknowledge the missing comparator baselines. Specifically, the manuscript states that STAR-RAG-style and simple time-filter baselines were not empirically implemented and that semantic-only comparisons do not resolve all comparator concerns. We frame these comparisons as future work and avoid broad comparative-superiority claims.

Manuscript location:
Related Work / Temporal-Aware Retrieval; Results / End-to-End Retrieval and Ablation Study; Discussion / Limitations; Conclusion.

Status:
Addressed with limitation.

### P34-P36 -- Failure analysis and limitation framing

Comment:
The manuscript should reduce unsupported design restatement and include clearer failure-analysis and limitation framing.

Response:
Thank you for the comment. We have revised the Discussion and Limitations to make the interpretation more cautious and explicit. Specifically, the manuscript now describes Negation Recency as no observed gain, Terminology Drift as a small-`n` result requiring caution, and broader router, score-treatment, comparator, sensitivity, corpus-scale, sidecar-provenance, and deployment-validation issues as limitations or future work. We avoid claiming completed causal failure analysis.

Manuscript location:
Discussion / Per-Scenario Analysis; Discussion / Per-Query Analysis; Discussion / Why TIMER-Graph Works; Discussion / The Role of Negative Beta; Discussion / Clinical Implications; Discussion / Limitations.

Status:
Addressed with limitation.

### P38 -- Weighted-scheduling / heterogeneous workload positioning

Comment:
The comment asks about weighted scheduling, heterogeneous workloads, or a DC2 anomaly, but the current traceability records do not confirm a manuscript target.

Response:
Thank you for the comment. We reviewed the manuscript for weighted-scheduling / heterogeneous-workload positioning. No active claim requiring a separate manuscript correction was identified in the revised manuscript. We therefore do not introduce a new unsupported claim or experiment. The current revision focuses on the major retrieval, scoring, evaluation, router, sidecar, baseline, and limitation issues addressed in the manuscript.

Manuscript location:
No confirmed manuscript target.

Status:
Reviewed -- no manuscript target identified / no manuscript edit required.

## 4. Comments addressed by manuscript correction

- P08: graph-overclaim wording corrected to temporal metadata sidecar framing.
- P11-P13: controlled fixture-score wording corrected.
- P14: Phase 9 protocol and result prominence clarified.
- P15: Negation Recency no-gain interpretation corrected.
- P16: Terminology Drift small-`n` caution added.

## 5. Comments addressed by limitation framing

- P19-P21: router accuracy, calibration, confidence distribution, trigger/fallback, zero-match/tie, and near-miss gaps.
- P23-P26: fixed settings and missing sensitivity/calibration/tuning evidence.
- P28: local corpus scale, current-index lineage, and non-production-scale scope.
- P30-P31: missing STAR-RAG-style and simple time-filter comparator baselines.
- P34-P36: failure-analysis boundaries and future-analysis needs.

## 6. Comments deferred as future work

- Natural-score hard-negative behavior beyond supplied fixture scores.
- Router evaluation.
- Parameter sensitivity analysis.
- Comparator baseline implementation.
- Phase 9 score-treatment analysis.
- Downstream clinical or LLM outcome validation.

Under the final Sprint 4.2 working decision, no extra analysis is required before ASAP submission; these unresolved items remain limitation-framed or future work.

## 7. Final working decisions

- Title: keep the `TIMER-Graph` title/name for this submission cycle.
- Code availability: use conservative restricted-data-safe wording and do not promise public code release.
- Supplementary material: do not submit or invent supplementary material unless a later supervisor decision supplies one.
- Source tracking/export: use local/Overleaf export handoff; do not modify `.gitignore`, force-add ignored files, stage, commit, or push.
- P38: handle as reviewed / no manuscript target identified / rebuttal-only.
- Extra analysis: no extra analysis before ASAP submission; unresolved analysis items remain limitation-framed or future work.

## 8. Submission note

For ASAP submission, the recommended path is to keep the title, keep limitation framing, use local/Overleaf export handoff, and avoid new experiments unless the supervisor specifically requests them. This rebuttal draft is ready for Implementation Supervisor review under the final working decisions above.
