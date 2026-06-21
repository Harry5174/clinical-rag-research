# Draft Future Reproduction Plan

## Purpose and approval boundary

This is a draft only. No command below was executed in Sprint 1.3. Each run requires a separate approval gate because the scripts read restricted local artifacts, write detailed result files, and may emit restricted debug output.

## 1. Hard-negative controlled evaluation

| Item | Draft |
|---|---|
| Purpose | Resolve the 96.0% versus 90.5% conflict using a captured v2 input and isolated output directory. |
| Script | scripts/run_hard_negative_eval.py |
| Intended input | data/mocks/combined_hard_negatives_v2.json |
| Proposed output | /tmp/clinical-rag-reproduction/hard-negative-v2/ |
| Existing-output risk | Script writes scenario CSVs, summary CSV, and LaTeX. Do not point it at results/phase5 or results/phase5_v2. |
| Restricted-output risk | Detailed CSVs and script debug output may include query/retrieval material. Keep logs and detailed outputs local/restricted. |
| Environment capture | record git HEAD, git status, Python version, lockfile hash, package environment, input SHA-256, script SHA-256, scoring-module SHA-256, and seed settings. |
| Approval | Required before execution. |

Draft sequence, for a future approved isolated local run:

    mkdir -p /tmp/clinical-rag-reproduction/hard-negative-v2
    uv run python scripts/run_hard_negative_eval.py --dataset data/mocks/combined_hard_negatives_v2.json --results-dir /tmp/clinical-rag-reproduction/hard-negative-v2 > /tmp/clinical-rag-reproduction/hard-negative-v2/local-restricted.log 2>&1

Before running, the approving sprint should decide whether to patch or otherwise suppress the evaluator's debug query output. Only sanitized aggregate summaries may leave the restricted directory.

## 2. Phase 9 filtered end-to-end evaluation

| Item | Draft |
|---|---|
| Purpose | Reproduce the 58% / 80% and 22% / 69% filtered ablation under captured provenance. |
| Script | scripts/run_end_to_end_eval.py |
| Inputs | data/mocks/combined_hard_negatives_v2.json; data/vector_store/poc_index.index; local metadata pickle; embedding model environment |
| Current output | results/phase9/end_to_end_results_filtered.csv |
| Existing-output risk | Current script has a fixed output path and would overwrite the existing result. |
| Network risk | The embedding model may require a local cache or an explicit approved model-download step. |
| Environment capture | record git HEAD, Python/package versions, model revision/cache identity, input/index/metadata hashes, device, output hash, and all output-path controls. |
| Approval | Required before execution and before any code change that makes the output path configurable. |

No direct reproduction command is safe with the current script because it overwrites the existing filtered output. The future approved sprint should first choose one of these approaches:

1. approve a minimal code change adding an explicit output-path argument; then run it against an isolated directory under /tmp/clinical-rag-reproduction/; or
2. use a separately approved isolated working copy whose local results path is disposable.

The required future record must state whether target-subject filtering is part of the intended paper protocol.

## 3. Index and metric verification

| Item | Draft |
|---|---|
| Purpose | Reconfirm index type, metric, dimension, vector count, and score interpretation alongside reproduction. |
| Inputs | local FAISS index and index-construction source |
| Output | sanitized metadata-only JSON or Markdown in a separately approved location |
| Risk | Low read risk; no index mutation permitted. |
| Approval | Required as part of a reproduction gate. |

## 4. Required reproducibility manifest fields

For each approved run, capture:

- purpose and protocol version;
- exact command;
- start/end timestamps;
- Git commit and clean/dirty status;
- script and relevant source hashes;
- input, index, metadata, and model hashes or immutable identifiers;
- environment and hardware/device information;
- output directory and every output hash;
- random seeds;
- a sanitized aggregate-only result summary;
- restricted-output handling confirmation.

## 5. Acceptance criteria

A future run can resolve the hard-negative headline only if it produces a sanitized aggregate table and a manifest tying the exact input hash, command, source revision, environment, and output hashes together. A Phase 9 run can support paper rewriting only if it additionally documents the target-subject-filtering protocol and avoids overwriting the existing historical result.

