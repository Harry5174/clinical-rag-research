import json
import sys
import time
from pathlib import Path
import numpy as np

# Setup paths
BASE_DIR = Path(__file__).resolve().parent.parent.parent.parent.parent
sys.path.append(str(BASE_DIR / "src"))

# Import from valid research paths
from app.research.retrieval.base import Retriever

# CONFIG
DATA_DIR = BASE_DIR / "data"
BASELINE_DATASET = DATA_DIR / "evaluation" / "baseline_dataset.json"
POC_VECTOR_DIR = DATA_DIR / "vector_store"
BASELINE_VECTOR_DIR = DATA_DIR / "vector_store" / "baseline"
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
        
        if (i+1) % 20 == 0:
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

    # Initialize retrievers
    print("Initializing retrievers...")
    poc_retriever = None
    baseline_retriever = None

    try:
        print(f"Loading POC Index from {POC_VECTOR_DIR}")
        poc_retriever = Retriever(POC_VECTOR_DIR)
    except Exception as e:
        print(f"Warning: POC Index load failed: {e}")

    try:
        print(f"Loading Baseline Index from {BASELINE_VECTOR_DIR}")
        baseline_retriever = Retriever(BASELINE_VECTOR_DIR)
    except Exception as e:
        print(f"Warning: Baseline Index load failed: {e}")
        
    results_map = {}

    # Run benchmarks
    if baseline_retriever:
        baseline_results = run_benchmark("NAIVE (Fixed-Size)", baseline_retriever, dataset)
        results_map["naive"] = baseline_results
    
    if poc_retriever:
        poc_results = run_benchmark("POC (Hybrid Semantic)", poc_retriever, dataset)
        results_map["poc"] = poc_results

    # Print Comparison Table
    print("\n" + "="*60)
    print(f"{'Metric':<20} | {'Naive Baseline':<15} | {'POC (Hybrid)':<15} | {'Delta':<10}")
    print("-" * 60)
    
    metrics_list = ["Recall@1", "Recall@5", "MRR", "Avg Latency"]
    
    for metric in metrics_list:
        v_base = results_map.get("naive", {}).get(metric, 0.0)
        v_poc = results_map.get("poc", {}).get(metric, 0.0)
        delta = v_poc - v_base
        print(f"{metric:<20} | {v_base:<15.4f} | {v_poc:<15.4f} | {delta:+.4f}")
    
    print("="*60)

    # Save results
    report_data = {
        "results": results_map,
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S")
    }
    report_path = DATA_DIR / "evaluation" / "phase_1_baseline_results.json"
    with open(report_path, "w") as f:
        json.dump(report_data, f, indent=2)
    print(f"\nRaw results saved to {report_path}")

if __name__ == "__main__":
    main()
