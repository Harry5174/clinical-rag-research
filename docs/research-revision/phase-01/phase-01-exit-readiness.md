# Phase 01 Exit Readiness

## Decision

**Recommendation: Phase 01 complete — proceed to detailed revision specs.**

Phase 01 has stabilized the evidence needed to plan a responsible revision. Completion does not mean all reviewer concerns are resolved experimentally. It means the current claims can now be sorted accurately into claims safe to preserve with limits, claims needing correction, and claims that must wait for a separately approved analysis.

## 1. Is empirical result provenance stable enough to plan rewriting?

Yes, for planning a scoped rewrite.

- The 96.0% versus 52.5% result was exactly reproduced for the documented current controlled/simulated hard-negative evaluator and input.
- The 58% / 80% versus 22% / 69% result was exactly reproduced for the documented current local target-subject-filtered Phase 9 protocol.
- The historical 90.5% hard-negative result and unfiltered Phase 9 result are superseded as current-result artifacts.
- Index and sidecar facts are sufficiently established to correct their method descriptions.

These findings do not establish general clinical utility, production readiness, naturalistic end-to-end performance, or empirical comparative superiority.

## 2. Which claims are now safe?

| Claim category | Safe status | Required boundary |
|---|---|---|
| Controlled hard-negative aggregate (96.0% versus 52.5%, n=200) | Safe to retain | Explicitly controlled/simulated local stress-test result. |
| Filtered Phase 9 aggregate (58% / 80% versus 22% / 69%, n=100) | Safe to retain | Explicitly local target-subject-filtered end-to-end protocol result. |
| Index fact | Safe to state after correction | IndexFlatL2 / METRIC_L2, not IndexFlatIP. |
| Sidecar implementation fact | Safe to state after correction | Generated/simulated temporal metadata sidecar; not observed provenance. |
| Corpus descriptive counts | Safe to state | Must accompany limited-scale/generalizability framing. |

## 3. Which claims require paper correction?

- FAISS IndexFlatIP / exact-inner-product wording.
- Sidecar provenance and overly graph-like novelty wording.
- ``Unconstrained'' or deployment-like wording for the filtered Phase 9 protocol.
- Broad clinical or production implications from controlled/local protocols.
- Supplementary-material availability assertion without a reviewed, sanitized package.
- Separate 20-query tuning-set assertion without documented evidence.

## 4. Which concerns require new analysis before rewrite?

The manuscript can be corrected without waiting for these analyses, but no strengthened claims should be written before they are completed:

- Router accuracy, calibration/confidence distribution, trigger/fallback rate, and regex near misses.
- Parameter sensitivity and defensible alpha/beta/lambda/tau rationale or tuning provenance.
- Empirical STAR-RAG-style and simple time-filter baselines.
- Terminology-drift uncertainty and a focused failure analysis.
- Investigation of the no-gain Negation Recency result.
- Any claim that extends protocol-bound results to broader clinical retrieval or deployment.

## 5. Which concerns can be handled by limitation framing?

- Limited corpus scale (165 subjects and 1,206 chunks).
- Controlled fixed-semantic-score hard-negative design and its restricted inference.
- Target-subject filtering as a material local evaluation condition.
- Single-source data and absence of downstream clinician/LLM assessment.
- Missing baselines and sensitivity analysis, provided the manuscript does not imply that they were completed.

## 6. Is Phase 01 complete after Sprint 1.5?

Yes, for the defined Phase 01 objective: analysis and correction groundwork. The evidence, lineage, method corrections, claim inventory, and reviewer-concern map are now available in sanitized planning documents.

Phase 01 does not authorize paper editing, rebuttal drafting, experiments, result changes, or release work.

## 7. What should Phase 02 be?

**Phase 02 — Detailed Revision Specifications and Analysis-Gap Plan.**

Its two coordinated tracks should be:

1. Produce section-by-section revision specifications that implement only the unblocked corrections and scope framing identified in Sprint 1.5.
2. Produce separately gated analysis specifications for router evaluation, sensitivity, missing baselines, uncertainty, and failure analysis. No new empirical claim should be drafted as resolved until its corresponding analysis is approved and completed.

## Open supervisor decision

Before detailed specifications are finalized, obtain clarification for P38: the intended text/location concerning DC2 anomaly, heterogeneous workloads, and weighted-scheduling positioning was not identified in the approved manuscript claim search.

## Scope confirmation

This exit decision is based on existing sanitized evidence and does not edit the paper or bibliography, alter repository implementation/data/results/indexes, expose raw local data, execute any evaluation, or make a commit/push.
