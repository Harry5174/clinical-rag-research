import faiss
import pickle
import json
import numpy as np
from pathlib import Path
from sentence_transformers import SentenceTransformer
from typing import List

def build_index(json_paths: List[Path], output_dir: Path, model_name: str = "BAAI/bge-base-en-v1.5"):
    print(f"--- INDEXING STARTED: {model_name} ---")
    
    # 1. Load Model
    model = SentenceTransformer(model_name)
    
    all_texts = []
    all_metadata = []

    # 2. Load Data
    for p in json_paths:
        if not p.exists():
            print(f"Skipping missing file: {p}")
            continue
            
        with open(p, "r") as f:
            data = json.load(f)
            print(f"Loaded {len(data)} chunks from {p.name}")
            
            for item in data:
                text = item.get("chunk_text", "").strip()
                if text:
                    # BGE-1.5 specific instruction for indexing side? 
                    # Usually BGE instruction is for queries, but let's keep raw text in index
                    # and add instruction at query time.
                    all_texts.append(text) 
                    all_metadata.append(item)

    if not all_texts:
        print("No text to index.")
        return

    # 3. Generate Embeddings
    # BGE instruction: For retrieval, we often embed documents directly 
    # and prepend instruction only to the QUERY.
    print(f"Embedding {len(all_texts)} documents...")
    embeddings = model.encode(all_texts, normalize_embeddings=True, show_progress_bar=True)

    # 4. Build FAISS Index
    dimension = embeddings.shape[1]
    index = faiss.IndexFlatL2(dimension)
    index.add(np.array(embeddings).astype('float32'))

    # 5. Save
    output_dir.mkdir(parents=True, exist_ok=True)
    index_path = output_dir / "poc_index.index"
    meta_path = output_dir / "poc_metadata.pkl"

    faiss.write_index(index, str(index_path))
    with open(meta_path, "wb") as f:
        pickle.dump(all_metadata, f)

    print(f"--- INDEXING COMPLETE ---")
    print(f"Index: {index_path}")
    print(f"Metadata: {meta_path}")