# Virtual Graph Naming Decision Record — Sprint 2.4

## 1. Decision

**Recommend Option B: rename “Virtual Graph” to “temporal metadata sidecar.”**

Decision class: `RENAME_TO_TEMPORAL_SIDECAR`.

## 2. Evidence

The generator writes a JSON sidecar with generated offsets, a placeholder date, section information, and heuristic intent labels. The reusable retriever loads a note-keyed mapping and uses an offset and section during reranking. Source-only inspection found no graph data structure, explicit nodes or edges, graph traversal, graph query, or graph database integration.

## 3. Why the original name is risky

“Virtual Graph” naturally implies graph representation or graph operations. Neither is evidenced. It also magnifies the risk that generated offsets and placeholder dates will be understood as observed clinical temporal relationships. The name is therefore an `UNSUPPORTED_NOVELTY_CLAIM` if presented as technical architecture.

## 4. Recommended terminology

Use **temporal metadata sidecar** consistently. Where precision is needed, qualify it as a generated local sidecar supplying a temporal offset to the current scorer.

## 5. What the paper must not claim

- An implemented graph, graph traversal, nodes/edges, graph reasoning, or graph database substitute.
- Graph-based retrieval novelty or comparison with graph architectures.
- Observed clinical temporal provenance for generated offsets or placeholder dates.

## 6. What the paper may safely claim

- The current implementation has a JSON metadata sidecar that is looked up by note identifier during temporal reranking.
- The sidecar supplies generated local temporal-scoring inputs in the inspected mechanism.
- The implementation does not require a graph database for this metadata lookup.

## 7. Reviewer P08 response direction

Apply the rename in a future approved paper-editing sprint, narrow the mechanism description, and disclose generated temporal provenance. Do not respond by defending graph-like behavior without implementation evidence.

## 8. Remaining supervisor decision, if any

No decision is required to use the recommended term. A supervisor decision is required only if retaining “Virtual Graph” as a non-technical metaphor; that retention is not recommended.
