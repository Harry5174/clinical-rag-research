import json
import pickle
import faiss
import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime
import sys
import torch
from sentence_transformers import SentenceTransformer

# Add src to path
sys.path.append(str(Path(__file__).parent.parent / "src"))
from app.research.retrieval.scoring import TIMERScorer

def apply_ablation(alpha, beta_current, beta_hist, beta_trend, S_sem, decay, intent):
    if intent == 'current':
        beta = beta_current
    elif intent == 'historical':
        beta = beta_hist
    else:  # trend
        beta = beta_trend
    
    return (alpha * S_sem) + (beta * decay)

def main():
    FAISS_PATH = "data/vector_store/poc_index.index"
    META_PATH = "data/vector_store/poc_metadata.pkl"
    MOCK_PATH = "data/mocks/combined_hard_negatives_v2.json"
    RESULTS_DIR = Path("results/phase9")
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    
    # 1. Load Data
    print("Loading index...")
    index = faiss.read_index(FAISS_PATH)
    print("Loading metadata...")
    with open(META_PATH, 'rb') as f:
        metadata = pickle.load(f)
    print("Loading mock dataset...")
    with open(MOCK_PATH, 'r') as f:
        mocks = json.load(f)
        
    queries = mocks['scenarios']['real_world_mining']
    
    # 2. Setup Embedder & Scorer
    print("Loading embedder BAAI/bge-base-en-v1.5...")
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = SentenceTransformer('BAAI/bge-base-en-v1.5', device=device)
    timer = TIMERScorer(lambda_decay=0.005)
    
    # Ablation configurations
    configs = {
        "Semantic-Only":      {"alpha": 1.0, "b_c": 0.0, "b_h": 0.0,  "b_t": 0.0},
        "Uniform Recency":    {"alpha": 0.6, "b_c": 0.8, "b_h": 0.8,  "b_t": 0.8},
        "Intent w/o Invert":  {"alpha": 0.6, "b_c": 0.8, "b_h": 0.0,  "b_t": 0.0},
        "TIMER-Graph Full":   {"alpha": 0.6, "b_c": 0.8, "b_h": -0.3, "b_t": 0.0}
    }
    
    results = {name: {"acc_1": 0, "recall_5": 0} for name in configs.keys()}
    total_queries = len(queries)
    
    print(f"\nExecuting end-to-end evaluation on {total_queries} queries...")
    export_data = []

    for i, q in enumerate(queries):
        query_text = q['text']
        expected_note_id = q['expected_retrieval']
        intent, _ = timer.classify_intent(query_text)
        
        # Find anchor date from mock (relative simulation date)
        anchor_date_str = None
        for n in q['notes']:
            if n.get('offset_days') == 0:
                anchor_date_str = n['note_date']
                break
        if not anchor_date_str:
            anchor_date_str = "2180-01-01" # fallback
        anchor_date = datetime.strptime(anchor_date_str.split(' ')[0], "%Y-%m-%d")
        
        # Embed Query
        q_emb = model.encode([query_text], normalize_embeddings=True)
        
        # Search FAISS
        D, I = index.search(q_emb, 50)
        D = D[0]
        I = I[0]
        
        # Score Normalization: mapping Inner Product to [0,1]
        S_norm = np.clip((D + 1.0) / 2.0, 0.0, 1.0)
        
        candidates = []
        for rank, (faiss_id, s_sem) in enumerate(zip(I, S_norm)):
            if faiss_id < 0 or faiss_id >= len(metadata):
                continue
                
            meta = metadata[faiss_id]
            note_date_str = str(meta['charttime']).split(' ')[0]
            try:
                note_date = datetime.strptime(note_date_str, "%Y-%m-%d")
                offset_days = (anchor_date - note_date).days
                # Prevent negative offsets if document is technically "after" the simulation date
                offset_days = max(0, offset_days) 
            except:
                offset_days = 0
                
            decay = timer.compute_temporal_decay(offset_days)
            candidates.append({
                "note_id": meta['note_id'],
                "s_sem": s_sem,
                "decay": decay
            })
            
        # Deduplicate note_ids preserving rank for accurate recall
        # (Since RAG returns the doc/chunk, and expected retrieval is a doc ID)
        
        row_data = {"query_id": q['id'], "query_text": query_text, "intent": intent, "expected": expected_note_id}
        
        for c_name, c_params in configs.items():
            # calculate final scores
            scored_candidates = []
            for cand in candidates:
                final_score = apply_ablation(
                    c_params["alpha"], 
                    c_params["b_c"], 
                    c_params["b_h"], 
                    c_params["b_t"], 
                    cand["s_sem"], 
                    cand["decay"], 
                    intent
                )
                scored_candidates.append((cand["note_id"], final_score))
                
            # sort & deduplicate maintaining order
            scored_candidates.sort(key=lambda x: x[1], reverse=True)
            seen = set()
            dedup_candidates = []
            for n_id, score in scored_candidates:
                if n_id not in seen:
                    dedup_candidates.append(n_id)
                    seen.add(n_id)
            
            top_1 = dedup_candidates[0] if dedup_candidates else None
            top_5 = dedup_candidates[:5]
            
            is_acc_1 = expected_note_id == top_1
            is_rec_5 = expected_note_id in top_5
            
            if is_acc_1: results[c_name]["acc_1"] += 1
            if is_rec_5: results[c_name]["recall_5"] += 1
            
            row_data[f"{c_name}_top1"] = top_1
            row_data[f"{c_name}_acc1"] = is_acc_1
            row_data[f"{c_name}_rec5"] = is_rec_5
            
        export_data.append(row_data)
        
    # Print results
    print("\n" + "="*50)
    print("END-TO-END EVALUATION RESULTS (n=100)")
    print("="*50)
    print(f"{'Configuration':<20} | {'Acc@1':<8} | {'Rec@5':<8}")
    print("-" * 50)
    for c_name in configs.keys():
        acc1 = results[c_name]["acc_1"] / total_queries
        rec5 = results[c_name]["recall_5"] / total_queries
        print(f"{c_name:<20} | {acc1:>.1%}    | {rec5:>.1%}")
        
    df = pd.DataFrame(export_data)
    out_path = RESULTS_DIR / "end_to_end_results.csv"
    df.to_csv(out_path, index=False)
    print(f"\nSaved detailed results to {out_path}")

if __name__ == "__main__":
    main()
