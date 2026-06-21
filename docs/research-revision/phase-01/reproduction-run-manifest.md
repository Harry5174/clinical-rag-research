# Sprint 1.4 Reproduction Run Manifest

## Run identity

| Field | Value |
|---|---|
| Run ID | sprint-1-4-20260621T190150Z |
| Private run root | /tmp/clinical-rag-reproduction/sprint-1-4-20260621T190150Z |
| Start time | 2026-06-21T19:02:04Z |
| Repository commit | e8d926c0b3cd8b8f6c9f0d7b0f0fbb02522d207c |
| Repository branch | main |
| Execution interpreter | .venv/bin/python, Python 3.13.1 |
| Runtime | FAISS 1.13.1; Torch 2.9.1+cpu; CUDA unavailable |
| Network policy | local-only; Phase 9 forced HF and Transformers offline |
| Sensitivity | all logs and detailed result outputs are LOCAL_RESTRICTED_DO_NOT_PUBLISH |

## Input and source hashes

| Artifact | SHA-256 |
|---|---|
| scripts/run_hard_negative_eval.py | 0690f454e229797e9a8547bce9557bc0d5dbd769abdfcd8f566a8705a1999b41 |
| scripts/run_end_to_end_eval.py | e6f58943a6a313b726c21ec880b41ecfd5750fd0cdbbe448c05945d0ab26a848 |
| data/mocks/combined_hard_negatives_v2.json | 7f933ca30c3c55c68635a9aea61ac1fda4a2117fcb611efe8226c48ffd339f41 |
| data/vector_store/poc_index.index | 82925a757df845d07cf25dc91ad79220a0dd68fa9fd89385d67ff545f9be4b7f |
| input hash manifest | 1764b3e7ec1fe8df7239a2ff325fe51f2c78606b944af31fcf62367eff7401d8 |
| environment manifest | b8e2ddc1ae19aa9567aef8657875568e8078951267a94aeb6cbfacaac55972ea |

## Run 1: controlled hard-negative reproduction

| Field | Value |
|---|---|
| Strategy | A: explicit isolated output argument |
| Inputs | current v2 combined hard-negative input and current evaluator |
| Command shape | .venv/bin/python evaluator with explicit v2 dataset and RUN_ROOT output directory |
| Output directory | private run-root/outputs/hard-negative-v2 |
| Logging | stdout/stderr redirected to private restricted log |
| Exit status | 0 |
| Aggregate | baseline 105/200 (52.5%); TIMER 192/200 (96.0%) |
| Terminology drift | baseline 0/20; TIMER 12/20 |
| Summary output hash | c5e33a5c41a740994017c84e4e7a55f89ea732d730f146daf4f6be72cd5d3743 |
| TeX output hash | 2e2753b3bb2e0a786d251e5564062b3c2feb4d41ebffa8dab8e2a26e07c5fd73 |
| Restricted log hash | 244c7c9217463f03eb1669f1d7e40f0083315701d73ae0c2e20872726f58fdc0 |
| Historical match | exact byte-for-byte match to Phase 5 outputs |

## Run 2: filtered Phase 9 reproduction

| Field | Value |
|---|---|
| Strategy | C: private copy with output-path-only patch |
| Source-copy hash before patch | e6f58943a6a313b726c21ec880b41ecfd5750fd0cdbbe448c05945d0ab26a848 |
| Source-copy hash after patch | 7aea766e72ba326e1f74257ce18f711e27a7e63519f662b250c022d6fb974635 |
| Patch verification | exactly one differing line; only expected output path change = true |
| Inputs | current v2 input, current local index and metadata, current source-linked retrieval code |
| Output directory | private run-root/outputs/phase9-filtered |
| Logging | stdout/stderr redirected to private restricted log |
| Offline controls | HF_HUB_OFFLINE=1; TRANSFORMERS_OFFLINE=1; HF_HUB_DISABLE_TELEMETRY=1 |
| Exit status | 0 |
| Aggregate | semantic-only 22% Accuracy@1 / 69% Recall@5; full TIMER 58% / 80% |
| Output hash | bcf806f3fb631012766b242f57e7ec6484f8048de9aac388acb88bd0afcc9ed3 |
| Restricted log hash | 5331c479dfc12f13042175c08851f126f44e205085653943667d380c9a2f490b |
| Historical match | exact byte-for-byte match to filtered Phase 9 artifact |

## Sanitized trace

The reproduced terminology-drift result has complete ID overlap with both historical result sets. Its baseline correctness matches both sets, its TIMER correctness matches Phase 5 20/20, and matches Phase 5 v2 9/20. Only non-text counts and hashes were used.

## Authority conclusion

- Controlled hard-negative: 96.0% is AUTHORITATIVE_EXISTING for the current evaluator and current v2 input, with exact isolated reproduction.
- Filtered Phase 9: 58% / 80% and 22% / 69% are AUTHORITATIVE_EXISTING for the current target-subject-filtered local implementation, with exact isolated reproduction.
- Phase 5 v2 90.5% is SUPERSEDED as a current result.
- The results retain controlled/simulated and local-protocol limitations documented in earlier sprints.

