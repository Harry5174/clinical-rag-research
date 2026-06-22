# Temporal Sidecar / Virtual Graph Correction Specification — Sprint 2.4

## 1. Executive summary

The inspected artifact is a generated JSON metadata sidecar, not observed clinical temporal provenance and not an implemented graph. Offsets are randomized from a category-dependent range, the note date is a placeholder, and the section-derived intent label is heuristic. The recommended paper-facing term is **temporal metadata sidecar**. The current “Virtual Graph” label should be removed rather than retained as a novelty or architecture claim.

## 2. Current paper sidecar / Virtual Graph claims

The paper describes a sidecar that supplies candidate offsets, calls it a lightweight virtual graph, and presents it as avoiding graph-database deployment. Its wording risks three unsupported inferences: source-observed dates, graph structure/behavior, and a graph-architecture novelty claim.

## 3. Implementation evidence

- `scripts/create_sidecar.py:18-22` generates offsets with pseudo-random integer ranges selected from a section-derived intent.
- `scripts/create_sidecar.py:52-57` writes per-note fields including a literal placeholder date.
- `scripts/create_sidecar.py:68-73` writes query metadata, including a restricted query-text field. Its contents were not inspected or reproduced.
- `src/app/research/retrieval/timer.py:31-35` loads only the `notes` mapping and reference date at runtime.
- `src/app/research/retrieval/timer.py:55-88` consumes `offset_days` and `section` for scoring/output; it does not consume note date or the generator's intent-ground-truth field.
- A source-only search across `src`, `scripts`, and `tests` found no graph library, graph data structure, node/edge construction, or traversal implementation.

## 4. Sidecar fields

| Scope | Fields established by generator | Runtime use |
|---|---|---|
| Top level | metadata version, fixed reference date, notes mapping, queries mapping | notes and reference date are loaded by `TIMERRetriever` |
| Per note | `offset_days`, `note_date`, `section`, `intent_ground_truth` | `offset_days` and `section` are read; the latter two are not used in the inspected runtime path |
| Per query | restricted query-text field, `intent_ground_truth`, valid temporal window, expected-note identifiers | not read by the inspected `TIMERRetriever` |

The per-query text field is restricted local content. This specification records only its schema role, never its contents.

## 5. Sidecar provenance

| Field or signal | Provenance classification | Basis |
|---|---|---|
| `offset_days` | generated and randomized | Generated from a category-dependent random range; not read from a source timestamp. |
| `note_date` | placeholder | The generator assigns one fixed literal date. |
| `section` | copied processed metadata | Taken from the local baseline input; exact source provenance was not independently validated in this sprint. |
| `intent_ground_truth` | generated heuristic label | Assigned from section membership with a default. |
| reference date | fixed local configuration | A fixed constant in the generator. |

The sidecar is therefore mixed metadata, but its temporal fields are generated/simulated. Any description of those fields as observed clinical chronology is `CONTRADICTED_BY_PROVENANCE`.

## 6. What the sidecar is

It is a JSON lookup mapping note identifiers to generated temporal-scoring inputs and related metadata. In the reusable retriever, it supplies an offset and section to a temporal reranker after candidate retrieval.

## 7. What the sidecar is not

- Not a source-observed temporal-provenance record.
- Not a clinical event timeline or validated longitudinal representation.
- Not a graph database, graph data structure, node/edge model, or traversal system.
- Not evidence that a graph-based retrieval method was evaluated.

## 8. Virtual Graph naming assessment

Classification: `RENAME_TO_TEMPORAL_SIDECAR`.

There is no implementation evidence for graph objects, explicit relationships, adjacency, graph search, or traversal. “Virtual Graph” is therefore an `UNSUPPORTED_NOVELTY_CLAIM` when used to imply graph-like behavior or architecture.

## 9. Recommended terminology

Use **temporal metadata sidecar** as the primary term. It accurately states that the artifact is metadata adjacent to the vector index while avoiding graph and observed-provenance implications.

## 10. Safe paper wording constraints

- Describe the artifact as a generated local metadata sidecar used to provide a temporal offset to the scorer.
- Identify offsets as generated/simulated when their provenance is relevant.
- Say only that the current reusable retriever looks up sidecar metadata by note identifier.
- Keep any mechanism statement separate from efficacy, clinical provenance, or graph novelty claims.

## 11. Unsafe paper wording

- “Observed clinical timestamps” or equivalent source-derived temporal language.
- “Graph nodes,” “edges,” “graph traversal,” “graph reasoning,” or graph-database equivalence.
- Claims that the sidecar alone establishes clinical chronology, temporal validity, or graph-based novelty.

## 12. Reviewer P08 mapping

P08 is resolved in direction by a paper correction: rename the construct to temporal metadata sidecar; disclose generated temporal provenance; remove graph-behavior and graph-novelty implications. Retaining any “Virtual Graph” label would require a supervisor decision and a narrow non-technical definition, but is not recommended.

## 13. Future dependencies

- `REQUIRES_FUTURE_ANALYSIS`: establish a sanitized, source-derived temporal-provenance pipeline before any observed-time claim.
- `REQUIRES_SUPERVISOR_DECISION`: only if retaining “Virtual Graph” as a metaphor despite the recommendation.
- `REMOVE_OR_DEFER`: graph novelty, graph-retrieval, and source-observed temporal claims until supporting implementation/evidence exists.
