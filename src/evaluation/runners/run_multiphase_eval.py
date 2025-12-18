import json
import sys
import time
from pathlib import Path
import numpy as np

# Setup paths
BASE_DIR = Path(__file__).resolve().parent.parent.parent.parent
sys.path.append(str(BASE_DIR))

# Import from valid research paths
from src.research.retrieval.base import Retriever

# CONFIG
DATA_DIR = BASE_DIR / "data"
BASELINE_DATASET = DATA_DIR / "evaluation" / "baseline_dataset.json"
VECTOR_STORE = DATA_DIR / "vector_store"
TOP_K = 5

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
    
    # Check if retriever loaded correctly
    try:
        if not retriever.index:
            print("Retriever index not loaded.")
            return None
    except:
        pass
    
    for i, item in enumerate(dataset):
        query = item["query"]
        target_note_id = item["target_note_id"]
        
        start_time = time.time()
        results = retriever.search(query, k=TOP_K)
        latency = time.time() - start_time
        
        r1, r5, mrr = calculate_metrics(results, target_note_id)
        
        metrics["recall_1"].append(r1)
        metrics["recall_5"].append(r5)
        metrics["mrr"].append(mrr)
        metrics["latency"].append(latency)
        
        # if (i+1) % 20 == 0:
        #     print(f"  Processed {i+1}/{len(dataset)}...")

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

    # Initialize retrievers
    print("Initializing retrievers...")
    
    retrievers = {}
    
    # 1. Naive Baseline
    try:
        path = VECTOR_STORE / "baseline"
        if path.exists():
            retrievers["Naive (Baseline)"] = Retriever(path)
    except Exception as e:
        print(f"Failed to load Baseline: {e}")

    # 2. Phase 1 POC (Hybrid)
    try:
        path = VECTOR_STORE 
        # Note: POC index is at root of vector_store currently based on previous phases
        # But wait, looking at file structure: data/vector_store/poc_index.index
        # Yes.
        if (path / "poc_index.index").exists():
             retrievers["Phase 1 (Hybrid)"] = Retriever(path)
    except Exception as e:
        print(f"Failed to load Phase 1: {e}")

    # 3. Phase 2 HeaderProp
    try:
        path = VECTOR_STORE / "research_v1"
        if path.exists():
            retrievers["Phase 2 (HeaderProp)"] = Retriever(path)
    except Exception as e:
        print(f"Failed to load Phase 2: {e}")

    results_map = {}

    # Run benchmarks
    for name, retriever in retrievers.items():
        results_map[name] = run_benchmark(name, retriever, dataset)

    # Print Comparison Table
    print("\n" + "="*80)
    print(f"{'Metric':<15} | {'Naive':<12} | {'Phase 1':<12} | {'Phase 2':<12} | {'Vs Ph1':<8}")
    print("-" * 80)
    
    metrics_list = ["Recall@1", "Recall@5", "MRR", "Avg Latency"]
    
    for metric in metrics_list:
        v_naive = results_map.get("Naive (Baseline)", {}).get(metric, 0.0)
        v_ph1 = results_map.get("Phase 1 (Hybrid)", {}).get(metric, 0.0)
        v_ph2 = results_map.get("Phase 2 (HeaderProp)", {}).get(metric, 0.0)
        
        delta = v_ph2 - v_ph1
        
        print(f"{metric:<15} | {v_naive:<12.4f} | {v_ph1:<12.4f} | {v_ph2:<12.4f} | {delta:+.4f}")
    
    print("="*80)

    # Save results
    report_data = {
        "results": results_map,
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S")
    }
    report_path = DATA_DIR / "evaluation" / "phase_2_results.json"
    with open(report_path, "w") as f:
        json.dump(report_data, f, indent=2)
    print(f"\nRaw results saved to {report_path}")

if __name__ == "__main__":
    main()
