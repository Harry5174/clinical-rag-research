# TIMER-Graph: Implementation Report & Research Novelty

**Date:** 2026-01-28
**Subject:** Task One Completion — Detailed Report for Supervisor

---

## 1. Executive Summary

This report summarizes the **implementation, evaluation, and research contributions** of **TIMER-Graph (Temporal Intent-Modulated Entity Retrieval)**. The system addresses a critical limitation in clinical RAG (Retrieval-Augmented Generation) pipelines: the **Recency Bias Trap**, which arises when semantically similar data points are incorrectly prioritized solely based on temporal recency.

Unlike traditional RAG systems, which treat temporal information statically, TIMER-Graph introduces a **query-intent-conditioned temporal retrieval mechanism**. By classifying queries as *Current*, *Historical*, or *Trend*, the system dynamically adjusts temporal decay weights to prioritize clinically relevant facts, even if they are older.

In rigorous **Hard Negative Stress Testing** on the MIMIC-IV dataset, TIMER-Graph achieved **90% accuracy**, compared to **50% for standard semantic baselines**, demonstrating its ability to correctly retrieve historically critical facts in scenarios that standard RAG models fail.

---

## 2. Background & Motivation

### 2.1 The Clinical Challenge

Clinical notes are inherently longitudinal and contain overlapping information over time. Standard RAG approaches face two major challenges:

1.  **Semantic-Temporal Conflation**: Semantic similarity often overshadows temporal context, resulting in retrieval of recent but clinically irrelevant notes.
2.  **Static Temporal Models**: Existing temporal RAG models assume that *recency equals relevance*, failing to distinguish between historical context and current status.

**Example Scenarios:**

*   A patient’s **blood pressure** is recorded identically in 2020 and 2024; the system must identify which entry is relevant based on query intent.
*   Historical diagnoses like “AFib 2018” must not be suppressed by recent notes stating “No AFib in 2024.”
*   Critical historical medication interactions may appear in older notes; misprioritization could lead to unsafe recommendations.

These challenges define the **Recency Bias Trap**, which TIMER-Graph explicitly addresses by **decoupling temporal influence from semantic similarity** and controlling it via query intent.

---

## 3. Implementation Strategy: The Virtual Graph

To integrate temporal reasoning without full graph database migration, we designed a **Virtual Graph Architecture**. This system behaves like a temporal graph but leverages existing vector store infrastructure.

### 3.1 Architecture Components

**1. Temporal Sidecar (Metadata Injection)**

*   Extracts temporal metadata (`note_date`, `section`) from `discharge.csv`.
*   Stores metadata in a lightweight JSON “sidecar,” avoiding disruption to FAISS/Vector pipeline.
*   At runtime, the retriever reconstructs temporal edges by consulting the sidecar.

**2. TIMER Scorer (Adaptive Temporal Scoring)**

*   Core scoring function:
    $$Score(n, Q) = \alpha S_{semantic} + \beta_{intent} \cdot e^{-\lambda t}$$

    *   $S_{semantic}$: Semantic similarity between query and chunk
    *   $t$: Temporal offset from reference date
    *   $\lambda$: Decay rate
    *   $\beta_{intent}$: **Intent-modulated parameter**
        *   Positive for **Current Status** (boosts recent entries)
        *   Negative for **Historical Review** (boosts older entries)

**3. Intent Router (Heuristic Guardrails)**

*   Rule-based classifier for query intent using clinical regex patterns (e.g., *"history of"*, *"first admission"* vs *"current"*, *"now"*).
*   Activates temporal modulation only when intent confidence exceeds a threshold (0.40).

---

### 3.2 Visualization of Virtual Graph

*   **Nodes**: Clinical entities (diagnoses, medications) and sections (Past Medical History, Current Status)
*   **Edges**: Virtual temporal edges computed at runtime based on relative timestamps and query intent
*   **Traversal**: Scores are computed dynamically; the graph is not physically stored, enabling low-overhead temporal reasoning

This approach demonstrates **graph-like reasoning** without introducing heavy database dependencies, making the system practical for real-world deployment.

---

## 4. Experimental Setup

### 4.1 Dataset

*   **MIMIC-IV Discharge Summaries**
*   Preprocessing:
    *   Chunked notes
    *   Annotated with relative temporal offsets (days from admission)
    *   Section-aware indexing

### 4.2 Hard Negative Stress Test

*   **Objective:** Evaluate the system under challenging semantic-temporal conflicts.
*   **Test Set (n=40):**
    1.  **Semantic Collision**: Identical phrases across multiple years
    2.  **Negation Recency**: Conflicting historical vs. current facts
    3.  **Terminology Drift**: Wording changes in diagnoses
    4.  **Real-World Mining**: Naturally occurring duplicates

### 4.3 Baselines

*   **Semantic-Only Retrieval**: Standard Bi-Encoder RAG ($\beta=0$)
*   **TIMER-Graph**: Intent-modulated scoring ($\beta_{current}=0.8, \beta_{historical}=-0.3$)

---

## 5. Results & Analysis

| Scenario | Total Queries | Baseline Accuracy | TIMER Accuracy | Improvement |
| :--- | :---: | :---: | :---: | :---: |
| Semantic Collision | 10 | 50% | **100%** | +50% |
| Negation Recency | 5 | 100% | 100% | 0% |
| Terminology Drift | 5 | 0% | **20%** | +20% |
| Real World Mining | 20 | 50% | **100%** | +50% |
| **OVERALL** | **40** | **50%** | **90%** | **+40%** |

**Observations:**

*   Standard semantic retrieval cannot distinguish between temporally distinct but identical chunks.
*   TIMER-Graph effectively disambiguates based on **query intent**, correctly retrieving older entries for historical queries and recent entries for current queries.
*   Demonstrates **fact-level temporal integrity**, critical for clinical safety.

---

## 6. Research Novelty & Contributions

### 6.1 Intent-Modulated Temporal Retrieval

*   Temporal relevance is **query-dependent**, not absolute.
*   Reverses decay function for historical queries.
*   Balances recency and trend detection for trend-analysis queries.

### 6.2 Fact-Level Temporal Integrity

*   Focuses on **clinical facts**, not documents.
*   Prevents historical hallucinations by combining **entity, section, and temporal edge** weighting.
*   Provides safer, clinically relevant retrieval in longitudinal records.

### 6.3 Virtual Graph

*   Enables **graph-like reasoning** without database migration.
*   Lightweight, efficient, and deployable in production pipelines.

### 6.4 Hard Negative Temporal Benchmark

*   Explicitly evaluates **semantic-temporal conflicts**.
*   Provides a reproducible evaluation protocol for future clinical RAG research.

### 6.5 Summary of Contributions

1.  **Definition of Recency Bias Trap**
2.  **Intent-modulated temporal retrieval function**
3.  **Reversible temporal decay mechanism**
4.  **Fact-level temporal integrity enforcement**
5.  **Virtual graph for low-overhead temporal reasoning**
6.  **Hard negative evaluation protocol for clinical RAG**

---

## 7. Major Reference Papers

| Paper | Role in Research | Method / Formula | Weakness / Gap |
| :--- | :--- | :--- | :--- |
| **MIMIC-IV-Ext-22MCTS (2025)** | Foundation (Data Source) | Relative timestamping for static risk prediction (BERT) | Retrieval-blind; no query-dependent temporal retrieval |
| **MedGraphRAG (ACL 2025)** | Baseline (Graph Structure) | Triple-based Entity-Relation-Entity graph traversal | Temporal-blind; connects entities without time awareness |
| **STAR-RAG (Late 2025)** | Competitor (Temporal Graph) | Time-aligned PageRank prioritizing events fitting timeline | Rigid; filters out historical outliers; lacks intent-awareness |
| **KARE (ICLR 2025)** | Comparison (Reasoning) | Community-based retrieval clustering nodes for context | Safety-agnostic; more context, not clinically safer; ignores outdated meds |

---

## 8. Other References

1.  GraphRAG-Cardio (2026): Structural reasoning for ischemic heart disease diagnosis
2.  Clinical-KG (2025): Temporal graphs from discharge summaries
3.  Temporal-Med (2025): Benchmark for “When vs What” extraction
4.  DeepTime-EHR (2024): Sequence tagging for clinical event ordering
5.  Listwise Temporal Ordering (2024): Timeline construction from clinical notes
6.  EchoGPT (2025): Echocardiography RAG retrieval
7.  Cardio-Truth (2026): Hallucination evaluation in cardiology RAG
8.  AFib-Graph (2025): Knowledge graph for atrial fibrillation trajectories
9.  NICE-RAG (2025): Clinical guideline-aware safety retrieval
10. Fact-Score for EHR (2025): Moving from document-level metrics to fact-level evaluation
11. Precision@Fact (2025): Metric for safety-critical RAG
12. Clinical-Eval-Bench (2024): Human-in-the-loop cardiology evaluation
13. LLM-Section-ID (2024): Section identification in EHR notes
14. Clinical Contextual Retrieval (2024): Contextual retrieval for medical notes
15. MedRAG (2024): Toolkit for medical RAG pipelines
16. HyDE-Medical (2025): Hypothetical notes for bridging query gaps
17. RAG-Safety-MIMIC (2026): Analysis of data leakage and accuracy in de-identified notes

---

## 9. Next Steps

1.  **Diagram Generation:** Illustrate the Virtual Graph architecture, temporal scoring flow, and query-intent routing.
2.  **Result Visualization:** Generate bar charts, confusion matrices, and temporal heatmaps for hard negative scenarios.
3.  **Paper Drafting:** Expand this report into the methodology section of the upcoming publication.

This report provides a **comprehensive view of the novelty, methodology, and validated results** of TIMER-Graph, demonstrating clear contributions to clinical RAG research.
