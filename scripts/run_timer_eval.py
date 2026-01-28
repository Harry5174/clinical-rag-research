import json
import pandas as pd
from pathlib import Path
from tqdm import tqdm
import time
import copy

# Import application modules
from app.research.retrieval.timer import TIMERRetriever
from app.research.retrieval.scoring import TIMERScorer
from app.evaluation.metrics import compute_recall_at_k, compute_TRA

# Setup Paths
BASE_DIR = Path(".")
DATA_DIR = BASE_DIR / "data"
SIDECAR_PATH = DATA_DIR / "mocks/temporal_sidecar.json"
INDEX_DIR = DATA_DIR / "vector_store"

def run_evaluation():
    print(">>> Starting TIMER-Graph Evaluation (Phase 4)")
    
    # 1. Load Data
    with open(SIDECAR_PATH, "r") as f:
        sidecar = json.load(f)
    
    queries = sidecar["queries"]
    notes_meta = sidecar["notes"]
    
    print(f"Loaded {len(queries)} queries and {len(notes_meta)} annotated notes.")

    # 2. Define Conditions
    conditions = [
        "Baseline-1-Semantic",
        "Baseline-2-Static", 
        "Baseline-3-NoNegBeta",
        "Baseline-4-NoConf",
        "TIMER-Full"
    ]
    
    results_log = []

    # 3. Initialize Retriever
    retriever = TIMERRetriever(INDEX_DIR, SIDECAR_PATH)

    # Reorder Queries to prioritize Historical ones for faster feedback
    prio_ids = ["q_018", "q_019"]
    query_items = list(queries.items())
    prio_items = [item for item in query_items if item[0] in prio_ids]
    rest_items = [item for item in query_items if item[0] not in prio_ids]
    sorted_queries = prio_items + rest_items
    
    start_time = time.time()

    for query_id, q_data in tqdm(sorted_queries, desc="Eval Queries"):
        query_text = q_data["text"]
        target_note_id = q_data["expected_notes"][0]
        valid_window = q_data["valid_temporal_window"]
        intent_gt = q_data["intent_ground_truth"]
        intent_gt = q_data["intent_ground_truth"]

        # --- Base Search (Once per Query) ---
        candidates = retriever.get_candidates(query_text, fetch_k=50)

        for cond in conditions:
            # --- Configure Scorer for Condition ---
            # Reset defaults
            TIMERScorer.BETA_CURRENT = 0.8
            TIMERScorer.BETA_HISTORICAL = -0.3
            TIMERScorer.BETA_TREND = 0.0
            TIMERScorer.CONFIDENCE_THRESHOLD = 0.70
            
            # Apply patches
            if cond == "Baseline-1-Semantic":
                TIMERScorer.BETA_CURRENT = 0.0
                TIMERScorer.BETA_HISTORICAL = 0.0
                TIMERScorer.BETA_TREND = 0.0
            
            elif cond == "Baseline-2-Static":
                # Static Decay for everything
                TIMERScorer.BETA_CURRENT = 0.5
                TIMERScorer.BETA_HISTORICAL = 0.5
                TIMERScorer.BETA_TREND = 0.5
            
            elif cond == "Baseline-3-NoNegBeta":
                # No Negative Beta (Historical treats old and new same? or just 0?)
                # Plan says: "Prove inversion is necessary". So use 0.0 or positive small?
                # Usually "Semantic Search" is Beta=0. 
                # If we want to show "Standard Temporal RAG" fails on history, we assume it uses Beta > 0.
                # So maybe this is same as Static?
                # Let's say Beta_Hist = 0.0 (Neutral) vs -0.3 (Pro-Old).
                TIMERScorer.BETA_HISTORICAL = 0.0
                
            elif cond == "Baseline-4-NoConf":
                # Always trust the router, or Never trust? 
                # "No Confidence Threshold" -> means we DON'T fallback to 0.0 if confidence is low.
                # The implementation falls back if conf < THRESHOLD.
                # So settng THRESHOLD = 0.0 means we ALWAYS use the predicted intent.
                TIMERScorer.CONFIDENCE_THRESHOLD = 0.0
            
            # TIMER-Full uses defaults

            # --- Run Search (Optimized) ---
            # We already have candidates, just need to re-score
            # Note: We need a DEEP COPY of candidates because apply_scoring modifies them in-place (adding keys)
            # Actually, apply_scoring creates a new list 'final_results', but modifies the dicts inside 'candidates'
            # to add 'timer_score'.
            # If we reuse 'candidates' objects, subsequent iterations will see 'timer_score' from previous iter.
            # This shouldn't break anything provided we overwrite 'timer_score' every time.
            # BUT: side effects on 'doc' dict might start to accumulate rubbish? No, just keys.
            # However, to be safe, let's copy the list of dicts.
            # However, to be safe, let's copy the list of dicts.
            candidates_copy = copy.deepcopy(candidates)
            
            ranked_results = retriever.apply_scoring(candidates_copy, query_text)
            retrieved = ranked_results[:5]
            
            # --- Compute Metrics ---
            recall_5 = compute_recall_at_k(retrieved, target_note_id, k=5)
            tra = compute_TRA(retrieved, notes_meta, valid_window)
            
            # Log
            results_log.append({
                "condition": cond,
                "query_id": query_id,
                "intent": intent_gt,
                "recall_5": recall_5,
                "tra": tra
            })

        # Save Incremental
        pd.DataFrame(results_log).to_csv("reports/timer_eval_results_partial.csv", index=False)

    # 4. Aggregate Results
    df = pd.DataFrame(results_log)
    
    print("\n>>> Results Summary (Mean by Condition & Intent) <<<")
    summary = df.groupby(["condition", "intent"])[["recall_5", "tra"]].mean()
    print(summary)
    
    # Also Overall Mean
    print("\n>>> Overall Mean by Condition <<<")
    overall = df.groupby(["condition"])[["recall_5", "tra"]].mean()
    print(overall)
    
    # Save
    df.to_csv("reports/timer_eval_results_raw.csv", index=False)
    summary.to_csv("reports/timer_eval_summary.csv")
    
    print(f"\nEvaluation Complete in {time.time() - start_time:.2f}s.")

if __name__ == "__main__":
    run_evaluation()
