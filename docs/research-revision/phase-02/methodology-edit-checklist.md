# Methodology Edit Checklist — Phase 02 Handoff

Use this checklist in a separately approved paper-editing sprint. Check an item only after the manuscript wording has been reviewed against the cited Phase 02 specifications.

## Must replace

- [ ] Replace every `IndexFlatIP` / exact-inner-product index statement with `IndexFlatL2` / `METRIC_L2`.
- [ ] Replace “Virtual Graph” with **temporal metadata sidecar**.
- [ ] Replace source-observed sidecar-time language with generated/simulated metadata provenance.
- [ ] Replace universal ambiguity/neutral-fallback language with the actual below-threshold condition and zero-match Current/0.5 default.

## Must remove

- [ ] Remove the unsupported separate 20-query tuning-set and held-fixed claim.
- [ ] Remove the unsupported existing supplementary-material availability claim.
- [ ] Remove graph nodes/edges/traversal and graph-architecture novelty claims.
- [ ] Remove trained-classifier, router-accuracy, calibration, safety, and validated-threshold claims.
- [ ] Remove unconstrained, deployment-ready, or broad clinical-utility claims not supported by the local protocols.

## Must soften

- [ ] Soften hard-negative outcomes to controlled/simulated mechanism-isolation evidence.
- [ ] Soften Phase 9 claims to local target-subject-filtered ablation evidence.
- [ ] Soften negative-beta novelty, scenario explanations, and clinical implications.
- [ ] Soften corpus-scale, chunking, comparator, and robustness implications.

## Must define

- [ ] Define `IndexFlatL2` / `METRIC_L2` and normalized document/query embeddings accurately.
- [ ] If mentioning cosine-equivalent ranking, state it only conditionally for unit-normalized vectors; do not equate it with raw returned L2 values.
- [ ] Define the TIMER formula as a local mechanism with protocol-dependent semantic input.
- [ ] Define the router as fixed regex/rule-based Current/Historical/Trend logic.
- [ ] Define tau as a fixed local setting, not a tuned/calibrated threshold.
- [ ] Define the sidecar as generated local metadata, including generated/randomized offset and placeholder date provenance.

## Must separate by protocol

- [ ] Separate controlled hard-negative supplied fixture scores from Phase 9 transformed L2-derived values.
- [ ] Separate controlled semantic-only ranking from Phase 9 semantic-only ablation behavior.
- [ ] Restrict equal-0.95 wording to the applicable controlled scenarios.
- [ ] Identify Phase 9’s full-index search, target-subject filtering, post-filter top-50, final-score sorting, note-ID deduplication, and top-1/top-5 evaluation flow.
- [ ] State that the optional cross-encoder is not part of the reproduced Phase 9 path.

## Must mark as limitation

- [ ] Phase 9 score treatment: raw L2-derived value is not validated cosine/inner-product similarity.
- [ ] Missing comparator baselines.
- [ ] Router accuracy, calibration, confidence distribution, trigger/fallback rate, zero-match/tie rate, and near-miss gaps.
- [ ] Negation Recency null finding and Terminology Drift uncertainty.
- [ ] Local corpus size and processed-subset scope.
- [ ] Exact current-index build-lineage / chunking provenance caveat.
- [ ] No downstream clinical/deployment validation.

## Must not invent

- [ ] Do not invent tuning, calibration, sensitivity, graph structure, observed temporal provenance, supplement availability, or router metrics.
- [ ] Do not convert limitations into resolved claims.
- [ ] Do not add new results, confidence intervals, causal explanations, or baseline comparisons.

## Must not overclaim

- [ ] Do not call raw L2 score cosine similarity, inner-product similarity, or a validated conversion.
- [ ] Do not call Phase 9 unconstrained, realistic in a deployment sense, or population-level.
- [ ] Do not call router fallback universally neutral or clinically safe.
- [ ] Do not claim broad comparative superiority or clinical deployment readiness.

## Needs supervisor decision

- [ ] Whether any residual “Virtual Graph” label is retained as a metaphor; Phase 02 recommends no retention.
- [ ] Whether score analysis changes require a code-correction/reproduction sprint before using Phase 9 for stronger semantic-score inference.
- [ ] Whether a sanitized supplementary package will be created; otherwise remove/defer availability language.
- [ ] Whether sufficient provenance exists to make a stronger exact index-build/chunking statement.
