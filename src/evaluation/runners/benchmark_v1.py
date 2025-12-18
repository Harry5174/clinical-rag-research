import sys
import json
import random
import time
import re
import numpy as np
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent.parent))
from app.retrieval.search import Retriever

# CONFIG
BASE_DIR = Path(__file__).resolve().parent.parent.parent
PROCESSED_JSON = BASE_DIR / "data" / "processed" / "discharge.json"
VECTOR_DIR = BASE_DIR / "data" / "vector_store"
TEST_SAMPLE_SIZE = 50 
TOP_K = 5 

# SKIP QUERIES CONTAINING THESE BOILERPLATE PHRASES
BOILERPLATE_TRIGGERS = [
    "Admission Date:", "Discharge Date:", "Date of Birth:", 
    "Service: MEDICINE", "No Known Allergies", "Social History:",
    "Family History:", "Discharge Diagnosis:", "Attending:", "Chief Complaint:"
]

def load_data(json_path):
    with open(json_path, "r") as f:
        data = json.load(f)
    return data

def generate_clean_query(chunk_text):
    # 1. Reject if chunk looks like a header/footer
    if any(phrase in chunk_text for phrase in BOILERPLATE_TRIGGERS):
        return None

    # 2. Split into sentences
    sentences = re.split(r'(?<!\w\.\w.)(?<![A-Z][a-z]\.)(?<=\.|\?)\s', chunk_text)
    valid_sentences = [s.strip() for s in sentences if len(s.split()) > 8] # increased to 8 words
    
    if not valid_sentences:
        return None
    
    # 3. Pick the longest non-boilerplate sentence
    query = max(valid_sentences, key=len)
    
    # 4. Final safety check on the QUERY itself
    if len(query) < 40 or any(phrase in query for phrase in BOILERPLATE_TRIGGERS):
        return None
        
    return query

def run_evaluation():
    print(f"--- STARTING FINAL EVALUATION (N={TEST_SAMPLE_SIZE}) ---")
    
    retriever = Retriever(VECTOR_DIR)
    all_chunks = load_data(PROCESSED_JSON)
    
    # Pre-filter dataset to only "content-rich" chunks
    candidates = []
    for c in all_chunks:
        if len(c['chunk_text']) > 150: # Only check substantial chunks
             candidates.append(c)
    
    test_set = random.sample(candidates, min(len(candidates), TEST_SAMPLE_SIZE * 3)) # Sample more, filter later
    
    metrics = {
        "strict_matches": 0,  # Exact Chunk ID match
        "note_matches": 0,    # Correct Document ID match (The real RAG metric)
        "latencies": [],
        "valid_queries": 0
    }
    
    print(f"\nFiltering for high-quality queries...")
    
    for target_chunk in test_set:
        if metrics["valid_queries"] >= TEST_SAMPLE_SIZE:
            break
            
        query = generate_clean_query(target_chunk['chunk_text'])
        if not query:
            continue
            
        metrics["valid_queries"] += 1
        
        # --- EXECUTE SEARCH ---
        start = time.time()
        results = retriever.search(query, k=TOP_K)
        metrics["latencies"].append(time.time() - start)

        # --- CHECK RESULTS ---
        found_strict = False
        found_note = False
        
        for res in results:
            # Check 1: Is it the EXACT same chunk?
            if res['note_id'] == target_chunk['note_id'] and \
               res['text'][:30] == target_chunk['chunk_text'][:30]:
                found_strict = True
            
            # Check 2: Is it the SAME NOTE (Parent Document)?
            if res['note_id'] == target_chunk['note_id']:
                found_note = True
        
        if found_strict: metrics["strict_matches"] += 1
        if found_note: metrics["note_matches"] += 1
        
        sys.stdout.write(f"\rProgress: {metrics['valid_queries']}/{TEST_SAMPLE_SIZE}")
        sys.stdout.flush()

    total = metrics["valid_queries"]
    if total == 0: return print("\nCould not generate valid queries.")
    
    print("\n\n" + "="*50)
    print("       FINAL POC VALIDATION REPORT       ")
    print("="*50)
    print(f"Total Quality Queries: {total}")
    print(f"Avg Latency:           {sum(metrics['latencies'])/total:.4f}s")
    print("-" * 50)
    print(f"STRICT Chunk Recall@5: {(metrics['strict_matches']/total)*100:.1f}%")
    print("   (Exact text window match)")
    print("-" * 50)
    print(f"PARENT Note Recall@5:  {(metrics['note_matches']/total)*100:.1f}%")
    print("   (Correct Patient Document retrieved)")
    print("="*50)
    
    if (metrics['note_matches']/total) > 0.8:
        print("✅ PASSED: High Fidelity for Document Retrieval.")
    else:
        print("⚠️ WARNING: Review embedding model selection.")

if __name__ == "__main__":
    run_evaluation()