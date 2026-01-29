"""
Merges synthetic hard negatives with mined real-world candidates.
Formats mined candidates into the evaluation schema, forcing identical semantic scores
to test the temporal mechanism in isolation.
"""

import json
from pathlib import Path
from datetime import datetime

def merge_datasets():
    # Paths
    synthetic_path = Path("data/mocks/hard_negatives.json")
    mined_path = Path("data/mocks/mined_candidates.json")
    output_path = Path("data/mocks/combined_hard_negatives.json")
    
    # Load Synthetic
    if synthetic_path.exists():
        with open(synthetic_path, 'r') as f:
            synthetic_data = json.load(f)
    else:
        print("⚠️ Synthetic dataset not found.")
        return

    # Load Mined
    mined_queries = []
    if mined_path.exists():
        with open(mined_path, 'r') as f:
            mined_data = json.load(f)
            
        print(f"Loaded {len(mined_data.get('candidates', []))} mined candidates.")
        
        for cand in mined_data.get('candidates', []):
            # We treat mined candidates as "Real World Semantic Collisions"
            # We assign identical semantic scores to force the retriever to decide on time.
            
            # Common Notes List
            # Ensure fields match Evaluator expectations
            old_note = cand['old_note']
            new_note = cand['new_note']
            
            # Enrich with mock semantic scores
            old_note['semantic_score'] = 0.95
            old_note['section'] = 'Clinical Note' # Default
            old_note['note_date'] = old_note['date']
            
            new_note['semantic_score'] = 0.95 
            new_note['section'] = 'Clinical Note'
            new_note['note_date'] = new_note['date']
            
            notes = [new_note, old_note] # Order doesn't matter for logic, but let's mix
            
            # 1. Historical Query
            q_hist = {
                "id": f"{cand['id']}_hist",
                "text": cand['suggested_queries']['historical'],
                "intent": "historical",
                "intent_confidence": 0.90, # Assume router works
                "notes": notes,
                "expected_retrieval": old_note['id'],
                "expected_rank": 1,
                "failure_mode": "real_world_collision",
                "source": "mimic_iv_mining"
            }
            
            # 2. Current Query
            q_curr = {
                "id": f"{cand['id']}_curr",
                "text": cand['suggested_queries']['current'],
                "intent": "current",
                "intent_confidence": 0.90,
                "notes": notes,
                "expected_retrieval": new_note['id'],
                "expected_rank": 1,
                "failure_mode": "real_world_collision",
                "source": "mimic_iv_mining"
            }
            
            mined_queries.append(q_hist)
            mined_queries.append(q_curr)
            
    # Merge
    combined_data = synthetic_data.copy()
    combined_data['scenarios']['real_world_mining'] = mined_queries
    
    # Update Metadata
    combined_data['metadata']['updated'] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    combined_data['metadata']['total_queries'] += len(mined_queries)
    combined_data['metadata']['description'] += " + Real World Mined Examples"
    
    # Save
    with open(output_path, 'w') as f:
        json.dump(combined_data, f, indent=2)
        
    print(f"✅ Merged Dataset Created: {output_path}")
    print(f"   Synthetic Queries: {synthetic_data['metadata']['total_queries']}")
    print(f"   Real-World Queries: {len(mined_queries)}")
    print(f"   Total Eval Queries: {combined_data['metadata']['total_queries']}")

if __name__ == "__main__":
    merge_datasets()
