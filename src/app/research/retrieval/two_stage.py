import torch
from pathlib import Path
from typing import List, Dict
from sentence_transformers import CrossEncoder
import numpy as np

from app.research.retrieval.base import Retriever

class TwoStageRetriever(Retriever):
    """
    Two-Stage Retrieval Pipeline:
    1. Candidate Generation: Retrieve Top-N using dense embeddings (Bi-Encoder).
    2. Reranking: Re-score candidates using a Cross-Encoder.
    """
    
    def __init__(self, index_dir: Path, 
                 bi_encoder_name: str = "BAAI/bge-base-en-v1.5",
                 cross_encoder_name: str = "BAAI/bge-reranker-base",
                 use_gpu: bool = False):
        
        # Initialize Base Retriever (Bi-Encoder)
        super().__init__(index_dir, bi_encoder_name)
        
        # Initialize Reranker (Cross-Encoder)
        print(f"Loading Reranker: {cross_encoder_name}...")
        device = "cuda" if use_gpu and torch.cuda.is_available() else "cpu"
        self.reranker = CrossEncoder(cross_encoder_name, device=device)
        print(f"Reranker loaded on {device}.")

    def search(self, query: str, k: int = 3, fetch_k: int = 20) -> List[Dict]:
        """
        Search with Reranking.
        
        Args:
            query: The user query.
            k: Number of final results to return.
            fetch_k: Number of candidates to fetch for reranking (usually 5x-10x k).
        """
        # Step 1: Candidate Generation (Dense Retrieval)
        # We fetch more candidates than needed (fetch_k)
        candidates = super().search(query, k=fetch_k)
        
        if not candidates:
            return []

        # Step 2: Prepare Pairs for Reranking
        # CrossEncoder expects list of [query, doc_text] pairs
        pairs = [[query, doc['text']] for doc in candidates]
        
        # Step 3: Rerank
        # scores is a list of floats
        scores = self.reranker.predict(pairs)
        
        # Step 4: Re-order Candidates
        # Combine candidate with its new score
        ranked_candidates = []
        for i, doc in enumerate(candidates):
            doc_with_score = doc.copy()
            doc_with_score['rerank_score'] = float(scores[i])
            doc_with_score['initial_score'] = doc['score'] # Keep original dist
            ranked_candidates.append(doc_with_score)
            
        # Sort by rerank_score (Descending - higher is better for CrossEncoder)
        ranked_candidates.sort(key=lambda x: x['rerank_score'], reverse=True)
        
        # Return Top-K
        return ranked_candidates[:k]
