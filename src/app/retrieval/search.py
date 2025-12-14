import faiss
import pickle
import numpy as np
from pathlib import Path
from sentence_transformers import SentenceTransformer

class Retriever:
    def __init__(self, index_dir: Path, model_name: str = "BAAI/bge-base-en-v1.5"):
        self.index_path = index_dir / "poc_index.index"
        self.meta_path = index_dir / "poc_metadata.pkl"
        self.model = SentenceTransformer(model_name)
        
        self._load_index()

    def _load_index(self):
        if not self.index_path.exists():
            raise FileNotFoundError(f"Index not found at {self.index_path}")
        
        self.index = faiss.read_index(str(self.index_path))
        with open(self.meta_path, "rb") as f:
            self.metadata = pickle.load(f)
        print("Retriever loaded successfully.")

    def search(self, query: str, k: int = 3):
        # BGE-1.5 Instruction for Queries
        query_text = f"Represent this sentence for retrieval: {query}"
        
        query_embedding = self.model.encode([query_text], normalize_embeddings=True)
        distances, indices = self.index.search(np.array(query_embedding).astype('float32'), k)
        
        results = []
        for i, idx in enumerate(indices[0]):
            if idx == -1: continue
            meta = self.metadata[idx]
            results.append({
                "score": float(distances[0][i]), # L2 distance (lower is better)
                "text": meta.get("chunk_text"),
                "note_id": meta.get("note_id")
            })
        return results