# Strategic Roadmap: Phase 5 (Hard Negative Dataset)

**Objective**: Quantitative proof of TIMER-Graph superiority over Semantic Baseline.
**Target**: Demonstrate >20% improvement in retrieval accuracy on "Hard Negative" scenarios.

---

## 1. The Strategy: Isolate Temporal Logic
We need to move beyond "Parity" safety checks. We will create a dataset where **Semantic Scores are identical**, forcing the retriever to rely on **Temporal Intent**.

### Scenario Definition
| Scenario | Description | TIMER Advantage |
|---|---|---|
| **Semantic Collision** | Identical phrases (e.g., "BP 120/80") at different times. | Uses `Intent` to pick Old vs New. |
| **Negation Recency** | Recent "No X" vs Old "Diagnosed with X". | Finds diagnosis for historical queries. |
| **Terminology Drift** | "IDDM" (Old) vs "Type 1 DM" (New). | Inverts decay to find older terms. |

---

## 2. Execution Plan (1 Week)

### Task 1: Synthetic Generation (Days 1-2)
- [ ] Create `data/mocks/hard_negatives.json`.
- [ ] Generate 10 examples with:
    - Identical semantic text.
    - `offset_days` = 0 (New) vs `offset_days` > 365 (Old).
    - Ground Truth: Historical Intent → Old Note.

### Task 2: Real-World Mining (Days 2-3)
- [ ] Scan `discharge.csv` for repeated clinically significant phrases (vitals, diagnoses).
- [ ] Filter for temporal separation > 1 year.
- [ ] Annotate 10 "wild" hard negatives.

### Task 3: Evaluation & Visualization (Days 4-5)
- [ ] Run Ablation Study on Hard Negatives.
- [ ] Generate "Ranking Delta" plot (Baseline Rank vs TIMER Rank).
- [ ] **Success Criteria**: TIMER Accuracy = 100%, Baseline Accuracy ~50%.

---

## 3. Deployment Strategy
**Decision**: Delay broad staging deployment until Phase 5 results are confirmed.
**Rationale**: "Deployment with Proof" is stronger than "Deployment with Parity".

---

## 4. Immediate Action Items
1.  Initialize `data/mocks/hard_negatives.json`.
2.  Script the synthetic data generation.
3.  Configure `run_timer_eval.py` to support the new dataset schema.
