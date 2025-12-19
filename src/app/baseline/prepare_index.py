import pandas as pd
import json
import sys
from pathlib import Path
import faiss
import pickle
import numpy as np
from sentence_transformers import SentenceTransformer

# Setup paths
BASE_DIR = Path(__file__).resolve().parent.parent.parent.parent
sys.path.append(str(BASE_DIR / "src"))

# Import the chunking logic (now in research/chunking/hybrid.py)
from app.research.chunking.hybrid import AdvancedChunker, ChunkConfig, ChunkingStrategy

# CONFIG
INPUT_CSV = BASE_DIR / "data" / "raw" / "discharge.csv"
OUTPUT_DIR = BASE_DIR / "data" / "vector_store" / "baseline"
MODEL_NAME = "BAAI/bge-base-en-v1.5"
LIMIT = 500 # Match POC limit for fair comparison

def main():
    print(f"--- PREPARING BASELINE INDEX (Strategy: FIXED_SIZE) ---")
    
    # 1. Load and Filter Data
    print(f"Loading data from {INPUT_CSV}...")
    df = pd.read_csv(INPUT_CSV, nrows=LIMIT)
    cardiac_keywords = ["cardiac", "heart", "myocardial", "coronary", "ECG", "heart failure", "atrial"]
    df["text"] = df["text"].fillna("")
    df_filtered = df[df["text"].str.contains("|".join(cardiac_keywords), case=False, na=False)].copy()
    print(f"Filtered to {len(df_filtered)} cardiac-related records.")

    # 2. Chunking (Naive Fixed Size)
    config = ChunkConfig(
        strategy=ChunkingStrategy.FIXED_SIZE,
        chunk_size=250, # Match POC size for fair comparison
        min_chunk_size=50
    )
    chunker = AdvancedChunker(config)
    
    all_chunks = []
    print("Chunking documents...")
    for idx, row in df_filtered.iterrows():
        base_meta = {
            "note_id": str(row.get("note_id", "")),
            "subject_id": str(row.get("subject_id", "")),
            "hadm_id": str(row.get("hadm_id", "")),
            "note_type": str(row.get("note_type", "discharge")),
            "charttime": str(row.get("charttime", ""))
        }
        chunks = chunker.chunk_text(row["text"], base_meta)
        all_chunks.extend(chunks)
    
    print(f"Generated {len(all_chunks)} baseline chunks.")

    # 3. Embedding
    print(f"Loading model {MODEL_NAME}...")
    model = SentenceTransformer(MODEL_NAME)
    
    texts = [c["chunk_text"] for c in all_chunks]
    print(f"Embedding {len(texts)} chunks...")
    embeddings = model.encode(texts, normalize_embeddings=True, show_progress_bar=True)

    # 4. Save Index & Metadata
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    
    dimension = embeddings.shape[1]
    index = faiss.IndexFlatL2(dimension)
    index.add(np.array(embeddings).astype('float32'))
    
    index_path = OUTPUT_DIR / "poc_index.index"
    meta_path = OUTPUT_DIR / "poc_metadata.pkl"
    chunk_json_path = OUTPUT_DIR / "baseline_chunks.json"

    faiss.write_index(index, str(index_path))
    with open(meta_path, "wb") as f:
        pickle.dump(all_chunks, f)
    with open(chunk_json_path, "w") as f:
        json.dump(all_chunks, f, indent=2)

    print(f"Baseline index saved to {OUTPUT_DIR}")

if __name__ == "__main__":
    main()
