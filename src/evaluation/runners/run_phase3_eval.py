import json
import sys
import time
from pathlib import Path
import numpy as np

# Setup paths
BASE_DIR = Path(__file__).resolve().parent.parent.parent.parent
sys.path.append(str(BASE_DIR))

# Import retrievers
from src.research.retrieval.base import Retriever
from src.research.retrieval.two_stage import TwoStageRetriever

# CONFIG
DATA_DIR = BASE_DIR / "data"
BASELINE_DATASET = DATA_DIR / "evaluation" / "baseline_dataset.json"
VECTOR_STORE = DATA_DIR / "vector_store" 
# vector_store/poc_index.index is the Hybrid index (Phase 1 winner)

TOP_K = 5
FETCH_K = 25 # Candidates to rerank

def calculate_metrics(results, target_note_id):
    """Calculate Recall and MRR for a single query"""
    recall_1 = 0
    recall_5 = 0
    mrr = 0
    
    for i, res in enumerate(results):
        if res["note_id"] == target_note_id:
            if i == 0: recall_1 = 1
            recall_5 = 1
            mrr = 1 / (i + 1)
            break
            
    return recall_1, recall_5, mrr

def run_benchmark(name, retriever, dataset):
    print(f"\n>>> Running Benchmark: {name}")
    metrics = {
        "recall_1": [],
        "recall_5": [],
        "mrr": [],
        "latency": []
    }
    
    for i, item in enumerate(dataset):
        query = item["query"]
        target_note_id = item["target_note_id"]
        
        start_time = time.time()
        # Handle TwoStage specific signature if needed, but search(query, k) is standard
        # For TwoStage, we hardcode fetch_k inside the class or pass it if flexible
        if isinstance(retriever, TwoStageRetriever):
            results = retriever.search(query, k=TOP_K, fetch_k=FETCH_K)
        else:
            results = retriever.search(query, k=TOP_K)
            
        latency = time.time() - start_time
        
        r1, r5, mrr = calculate_metrics(results, target_note_id)
        
        metrics["recall_1"].append(r1)
        metrics["recall_5"].append(r5)
        metrics["mrr"].append(mrr)
        metrics["latency"].append(latency)
        
        if (i+1) % 50 == 0:
            print(f"  Processed {i+1}/{len(dataset)}...")

    return {
        "Recall@1": np.mean(metrics["recall_1"]),
        "Recall@5": np.mean(metrics["recall_5"]),
        "MRR": np.mean(metrics["mrr"]),
        "Avg Latency": np.mean(metrics["latency"])
    }

def main():
    if not BASELINE_DATASET.exists():
        print(f"Error: Baseline dataset not found at {BASELINE_DATASET}")
        return

    print(f"Loading baseline dataset...")
    with open(BASELINE_DATASET, "r") as f:
        dataset = json.load(f)
    print(f"Loaded {len(dataset)} evaluation pairs.")

    results_map = {}

    # 1. Phase 1 (Hybrid) - The Winner so far
    print("\nLoading Phase 1 Retriever (Base Bi-Encoder)...")
    try:
        # Note: We use the root vector_store which has the Hybrid index
        r_phase1 = Retriever(VECTOR_STORE)
        results_map["Phase 1 (Hybrid)"] = run_benchmark("Phase 1 (Hybrid)", r_phase1, dataset)
    except Exception as e:
        print(f"Failed to load Phase 1: {e}")

    # 2. Phase 3 (Two-Stage Reranking)
    print("\nLoading Phase 3 Retriever (Two-Stage)...")
    try:
        # We use the SAME Hybrid Index, but wrap it with Reranking
        r_phase3 = TwoStageRetriever(VECTOR_STORE)
        results_map["Phase 3 (Reranking)"] = run_benchmark(f"Phase 3 (Reranking @ {FETCH_K})", r_phase3, dataset)
    except Exception as e:
        print(f"Failed to load Phase 3: {e}")
        import traceback
        traceback.print_exc()

    # Print Comparison
    print("\n" + "="*80)
    print(f"{'Metric':<15} | {'Phase 1':<15} | {'Phase 3':<15} | {'Delta':<8}")
    print("-" * 80)
    
    metrics_list = ["Recall@1", "Recall@5", "MRR", "Avg Latency"]
    
    for metric in metrics_list:
        v_p1 = results_map.get("Phase 1 (Hybrid)", {}).get(metric, 0.0)
        v_p3 = results_map.get("Phase 3 (Reranking)", {}).get(metric, 0.0)
        delta = v_p3 - v_p1
        
        print(f"{metric:<15} | {v_p1:<15.4f} | {v_p3:<15.4f} | {delta:+.4f}")
    
    print("="*80)

    # Save results
    report_data = {
        "results": results_map,
        "config": {"fetch_k": FETCH_K, "model": "BAAI/bge-reranker-base"}
    }
    report_path = DATA_DIR / "evaluation" / "phase_3_results.json"
    with open(report_path, "w") as f:
        json.dump(report_data, f, indent=2)
    print(f"\nRaw results saved to {report_path}")

if __name__ == "__main__":
    main()
