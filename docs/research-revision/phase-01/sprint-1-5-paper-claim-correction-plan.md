# Sprint 1.5 — Paper Claim Correction Plan and Reviewer-Concern Mapping

## Purpose and decision boundary

This document converts the sanitized Phase 01 evidence into a paper-revision plan. It is not a paper rewrite, a rebuttal, or a new evaluation. Claim summaries below are deliberately abbreviated and contain no clinical-note, patient, query, answer, or retrieved-content text.

The evidence base is Sprint 1.1--1.4, the authoritative-results manifest, the reproduction-run manifest, the current manuscript claim inventory, and bibliography-support context. The current manuscript and bibliography are unchanged by this sprint.

## Authority summary

| Evidence item | Sprint 1.5 planning status | Revision consequence |
|---|---|---|
| 200-row hard-negative result: TIMER 96.0% versus semantic-only baseline 52.5% | AUTHORITATIVE_EXISTING only for the reproduced controlled/simulated hard-negative protocol using the documented local evaluator and input | Retain only with controlled-protocol scope, not as general clinical deployment evidence. |
| Historical 90.5% hard-negative artifact | SUPERSEDED as the current result | Do not substitute it for the reproduced 96.0% result. |
| Filtered Phase 9 result: TIMER 58% Accuracy@1 / 80% Recall@5; semantic-only 22% / 69% | AUTHORITATIVE_EXISTING only for the documented local target-subject-filtered protocol | Retain with the filtering and local-protocol conditions stated prominently. |
| Unfiltered Phase 9 result | SUPERSEDED | Remove from current-result presentation. |
| Current inspected FAISS index | IndexFlatL2 / METRIC_L2 | Correct all IndexFlatIP and literal exact-inner-product descriptions. |
| Temporal sidecar | Generated/simulated metadata with randomized offsets and a placeholder date | Reframe it as generated/simulated metadata, not observed clinical temporal provenance or a validated graph. |

## Required claim dispositions

The detailed inventory is in [paper-claim-correction-matrix.md](paper-claim-correction-matrix.md). The required directions are:

1. Keep the 96.0% hard-negative result only as a controlled/simulated local stress-test result; remove any implication that it establishes broad clinical retrieval or deployment performance.
2. Keep the 58% / 80% and 22% / 69% Phase 9 aggregates only as local target-subject-filtered end-to-end protocol results. The protocol must be defined where the metrics appear.
3. Correct the retrieval method to IndexFlatL2 / METRIC_L2. If normalized embeddings are relevant, describe this precisely as L2 search over normalized embeddings; do not assert equivalence beyond evidence.
4. Reframe ``Virtual Graph'' and its sidecar as a lightweight generated/simulated metadata sidecar. Do not claim observed temporal provenance or demonstrated graph novelty without new evidence.
5. Remove or defer the supplementary-material availability claim until a reviewed, sanitized supplementary package exists.
6. Remove or qualify the separate tuning-set claim for tau and other hyperparameters unless documented development-set evidence can be supplied.
7. Do not add router-performance claims until standalone router evaluation, confidence/trigger reporting, and regex near-miss analysis exist.
8. Treat sensitivity, missing baselines, corpus scale, and the lack of downstream clinical evaluation as explicit limitations or future-analysis requirements.

## Section-level correction plan

| Paper section | Main issue | Claims affected | Reviewer comments affected | Required evidence | Correction direction | Rewrite priority | Blocked or unblocked |
|---|---|---|---|---|---|---|---|
| Abstract | Broad empirical and clinical framing gives the controlled result disproportionate weight. | C01--C03 | P14, P28 | Reproduced aggregate manifests; protocol limits | Lead with scoped contribution; distinguish controlled and filtered end-to-end results; avoid deployment language. | High | Unblocked for scoped rewrite. |
| Introduction and positioning | Novelty and comparison language overstates graph/provenance and may imply unavailable empirical baselines. | C04, C07, C12 | P08, P30--P31, P34--P35 | Current implementation facts; bibliography context; new baselines if comparative claims remain | Define the mechanism narrowly; avoid untested empirical superiority claims. | High | Partly blocked by baseline analysis. |
| Methodology: router | Router mechanism is described, but standalone performance, trigger distribution, and near misses are absent. | C06 | P19--P21 | New router evaluation | Describe as heuristic mechanism only, or add analysis before performance claims. | High | Blocked for performance claims. |
| Methodology: sidecar and retrieval | Sidecar provenance and FAISS metric wording conflict with evidence. | C04, C05, C07, C08 | P08 | Current inspected implementation and manifests | State IndexFlatL2 / METRIC_L2; label sidecar generated/simulated; define ``Virtual Graph'' early and narrowly. | Critical | Unblocked for correction. |
| Experimental setup | Corpus scale, fixed semantic score, baselines, and tuning evidence need transparent scope. | C09--C14 | P11--P13, P23--P28, P30--P31 | Existing manifests; new sensitivity/baseline/tuning evidence where claims retained | Separate controlled stress test from end-to-end protocol; remove unsupported tuning claim; add limitations. | Critical | Mixed. |
| Results | Result scope and Phase 9 filtering require clear protocol labels; selected statistical/failure claims require restraint. | C15--C19 | P14--P16, P34--P36 | Existing reproductions; new interval/failure analysis if reported beyond current evidence | Present filtered end-to-end result prominently; report controlled result as controlled; do not overinterpret no-gain or small-n results. | Critical | Mixed. |
| Discussion and limitations | Clinical/deployment implications exceed protocol scope; failure analysis is thin. | C20--C22 | P15--P16, P28, P34--P36 | Existing failure signals; new analysis for quantified claims | Replace broad efficacy/deployment language with limitations and protocol-bound interpretation. | High | Unblocked for scope correction; new analysis needed for expanded failure claims. |
| Conclusion and declarations | Repeats broad claims and asserts supplementary availability. | C02, C18, C22, C23 | P14, P28 | Existing manifests; sanitized package if availability retained | Preserve only scoped findings and remove unsupported supplement statement. | Critical | Unblocked except supplement availability. |
| Scheduling/DC2 terminology | The named concern was not located in the approved manuscript claim search. | C24 | P38 | Supervisor identification of intended text and claim | Do not infer a correction target; add only after clarification. | Medium | Blocked by clarification. |

## Reviewer-concern handling strategy

The complete table is in [reviewer-concern-mapping.md](reviewer-concern-mapping.md). Existing evidence can support immediate paper correction for protocol scope, IndexFlatL2 wording, generated-sidecar framing, corpus limitation, and prominence of the filtered Phase 9 result. It cannot establish router accuracy, tuning validity, parameter sensitivity, missing comparative baselines, or a broader failure analysis.

## Future work queue

### Safe for the detailed revision specification

- Exact method correction from IndexFlatIP to IndexFlatL2 / METRIC_L2.
- Scope correction for the controlled 96.0% result and filtered 58% / 80% result.
- Sidecar/provenance reframe and earlier narrow definition of ``Virtual Graph.''
- Removal or deferral of unsupported supplementary-material and tuning-set assertions.
- Corpus-scale, single-source, protocol, and downstream-evaluation limitations.

### Requires a separately approved analysis plan before stronger paper claims

- Router accuracy, confidence distribution, fallback/trigger rate, and regex near-miss evaluation.
- Sensitivity analysis for alpha, beta, lambda, and tau; documented tuning provenance if tuning is retained.
- Empirical STAR-RAG-style and simple time-filter baselines.
- Terminology-drift uncertainty/failure analysis, including an appropriate interval for the n=20 result.
- Investigation of the zero-gain negation-recency result and a defensible interpretation.
- Review of target-subject filtering as the intended evaluation protocol and any claim about broader retrieval difficulty.

### Requires supervisor clarification

- The exact manuscript location and intended response for P38 (DC2 anomaly, heterogeneous workloads, and weighted-scheduling positioning).
- Whether the planned revision should retain a named ``Virtual Graph'' label after it is narrowly defined as a sidecar-based mechanism.

## Phase transition recommendation

Phase 01 is complete for evidence-grounded revision planning: reproduced result provenance, current implementation facts, and claim dispositions are sufficiently stable to write detailed revision specifications. Phase 02 should be **Detailed Revision Specifications and Analysis-Gap Plan**: specify paper edits without performing them, and separately scope the analyses that are prerequisites for any strengthened router, sensitivity, baseline, or failure claims.

## Scope confirmation

This sprint created planning documentation only. It does not edit the manuscript or bibliography, alter code/data/results/indexes/notebooks, run evaluations, create metrics, inspect restricted report content, expose raw local data, or prepare a final rebuttal.
