# Sprint 1.4 — Safe Reproduction Harness and Output Isolation

## Outcome

Sprint 1.4 established a private, permission-restricted harness under /tmp/clinical-rag-reproduction/sprint-1-4-20260621T190150Z. Both approved reproductions completed with outputs and logs outside the repository. No repository result, data, index, or code artifact was written.

All documentation in this sprint is sanitized. Detailed CSV outputs and logs remain LOCAL_RESTRICTED_DO_NOT_PUBLISH under the private /tmp run directory.

## Git state and scope

- Repository branch: main
- Commit at capture and execution: e8d926c0b3cd8b8f6c9f0d7b0f0fbb02522d207c
- Starting pre-existing untracked files: Sprint 1.3 lineage/map/reproduction-plan documents.
- No staging, commit, push, reset, checkout, or destructive Git operation occurred.

## Files inspected

- Sanitized Phase 01 Sprint 1.1 through Sprint 1.3 documentation.
- scripts/run_hard_negative_eval.py
- scripts/run_end_to_end_eval.py
- current scoring, retrieval, input, index, and result metadata identified by the prior manifest.
- Private reproduction outputs, hash manifests, and restricted logs only through aggregate-safe tooling.

Raw historical reports were not inspected by content.

## Commands and safety strategy

Hard-negative evaluator: Strategy A. The script accepts explicit dataset and results-directory arguments. Its output directory was redirected under the private run root and stdout/stderr were redirected to a private log.

Phase 9 evaluator: Strategy C. The source script has a fixed repository results path. A copy was created under the private run root and patched at exactly one line: RESULTS_DIR. The patch changed only the absolute output directory to the private run root; it did not alter inputs, retrieval, filtering, scoring, labels, ranking, or metrics. A source-link in the private workspace pointed read-only to the repository src tree.

Both runs used .venv/bin/python and local-only dependency/model resources. Phase 9 used HF_HUB_OFFLINE=1, TRANSFORMERS_OFFLINE=1, and HF_HUB_DISABLE_TELEMETRY=1.

## Isolated workspace

| Item | Value |
|---|---|
| Run root | /tmp/clinical-rag-reproduction/sprint-1-4-20260621T190150Z |
| Permissions | 0700 |
| Logs | run-root/logs/ |
| Outputs | run-root/outputs/ |
| Manifests | run-root/manifests/ |
| Temporary Phase 9 copy | run-root/snapshots/phase9-workspace/scripts/run_end_to_end_eval.py |
| Patch verification | one differing source line; only expected output-path change = true |

## Environment capture

- Baseline capture: system Python 3.12.4 and uv 0.5.21.
- Execution interpreter: .venv/bin/python, Python 3.13.1.
- Runtime imports: FAISS 1.13.1, Torch 2.9.1+cpu, CUDA unavailable.
- Local BAAI BGE model-cache directory was present; Phase 9 was forced offline.
- Environment-manifest hash: b8e2ddc1ae19aa9567aef8657875568e8078951267a94aeb6cbfacaac55972ea.
- Input-hash-manifest hash: 1764b3e7ec1fe8df7239a2ff325fe51f2c78606b944af31fcf62367eff7401d8.
- Local runtime-check hash: 25bd59d862a9dc6c44cc666be4c2ea295967015c75019b60c9cd64bdabeab18a.
- Execution-runtime-manifest hash: b2838e3019694b82db17453aa10e19ae847149fd1469c568d90872ddb75ca9b5.

## Input, script, and index provenance

| Artifact | SHA-256 |
|---|---|
| hard-negative evaluator | 0690f454e229797e9a8547bce9557bc0d5dbd769abdfcd8f566a8705a1999b41 |
| Phase 9 evaluator source | e6f58943a6a313b726c21ec880b41ecfd5750fd0cdbbe448c05945d0ab26a848 |
| v2 combined hard-negative input | 7f933ca30c3c55c68635a9aea61ac1fda4a2117fcb611efe8226c48ffd339f41 |
| current end-to-end index | 82925a757df845d07cf25dc91ad79220a0dd68fa9fd89385d67ff545f9be4b7f |
| reproduced Phase 5 summary | c5e33a5c41a740994017c84e4e7a55f89ea732d730f146daf4f6be72cd5d3743 |
| reproduced Phase 9 filtered output | bcf806f3fb631012766b242f57e7ec6484f8048de9aac388acb88bd0afcc9ed3 |

The full sanitized path/hash inventory is retained in the private input-hash manifest.

## Hard-negative reproduction

Command shape used:

    .venv/bin/python scripts/run_hard_negative_eval.py --dataset data/mocks/combined_hard_negatives_v2.json --results-dir RUN_ROOT/outputs/hard-negative-v2 > RUN_ROOT/logs/hard-negative.local-restricted.log 2>&1

Result:

| Metric | Aggregate |
|---|---:|
| rows | 200 |
| baseline correct | 105 |
| baseline accuracy | 52.5% |
| TIMER correct | 192 |
| TIMER accuracy | 96.0% |
| terminology-drift TIMER correct | 12 / 20 |

The reproduced output hashes exactly match every corresponding Phase 5 artifact, including summary CSV and LaTeX table. The restricted log hash is 244c7c9217463f03eb1669f1d7e40f0083315701d73ae0c2e20872726f58fdc0.

## Terminology-drift non-text trace

- Reproduced versus Phase 5: 20/20 ID overlap; 20/20 matching baseline outcomes; 20/20 matching TIMER outcomes.
- Reproduced versus Phase 5 v2: 20/20 ID overlap; 20/20 matching baseline outcomes; 9/20 matching TIMER outcomes.
- The reproduced run aligns with Phase 5 96.0%, not Phase 5 v2 90.5%.

No query, note, answer, retrieval, or patient-level values were inspected or included.

## Phase 9 reproduction

The private script copy was executed from the repository working directory with the output-path-only patch and offline model settings.

| Metric | Aggregate |
|---|---:|
| rows | 100 |
| semantic-only Accuracy@1 / Recall@5 | 22% / 69% |
| full TIMER Accuracy@1 / Recall@5 | 58% / 80% |

All four configurations have 100/100 matching correctness booleans against the historical filtered artifact, and the reproduced output file hash is exactly identical. The private output hash is bcf806f3fb631012766b242f57e7ec6484f8048de9aac388acb88bd0afcc9ed3. The restricted log hash is 5331c479dfc12f13042175c08851f126f44e205085653943667d380c9a2f490b.

## Authority and remaining limits

The Phase 5 96.0% result is now AUTHORITATIVE_EXISTING for the explicit current controlled/simulated evaluator plus current v2 input. The historical 90.5% Phase 5 v2 output is SUPERSEDED as a current result.

The filtered Phase 9 result is now AUTHORITATIVE_EXISTING for the explicit current local target-subject-filtered implementation. This does not establish general clinical utility, resolve the L2 versus inner-product paper correction, validate sidecar provenance, or justify unqualified end-to-end deployment claims.

## Required future fixes

- Add an approved explicit output-path argument to the Phase 9 evaluator; do not rely on copied-script patching for normal reproduction.
- Remove or gate hard-negative debug printing before any non-restricted execution.
- Retain a sanitized, durable run manifest with every future publication-facing evaluation.
- Correct paper claims about IndexFlatIP and real temporal sidecar provenance in a separately approved paper-rewrite sprint.

## Scope confirmation

- No paper, bibliography, repository code, notebook, result, data, index, or .gitignore file was edited.
- No existing repository result was overwritten.
- All run outputs, logs, snapshots, and manifests stayed under the private /tmp run root.
- No raw clinical-note text, MIMIC row, query, answer, or retrieved content is included.
- No network access, external API/service, commit, or push was used.

