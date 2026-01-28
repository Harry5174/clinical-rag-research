import json
from pathlib import Path
from typing import List, Dict, Optional

from app.research.retrieval.two_stage import TwoStageRetriever
from app.research.retrieval.scoring import TIMERScorer

class TIMERRetriever(TwoStageRetriever):
    """
    TIMER-Graph Retriever: Extends Two-Stage Reranking with Temporal Intent Modulation.
    """
    
    def __init__(self, index_dir: Path, 
                 sidecar_path: Path,
                 bi_encoder_name: str = "BAAI/bge-base-en-v1.5",
                 cross_encoder_name: str = "BAAI/bge-reranker-base",
                 use_gpu: bool = False):
        
        super().__init__(index_dir, bi_encoder_name, cross_encoder_name, use_gpu)
        
        self.scorer = TIMERScorer()
        self.sidecar_path = sidecar_path
        self._load_sidecar()

    def _load_sidecar(self):
        if not self.sidecar_path.exists():
            print(f"Warning: Sidecar not found at {self.sidecar_path}. Temporal scoring will be disabled.")
            self.sidecar_data = {}
            return
            
        with open(self.sidecar_path, "r") as f:
            data = json.load(f)
            self.sidecar_data = data.get("notes", {})
            self.reference_date = data.get("reference_date")
            print(f"Loaded Temporal Sidecar: {len(self.sidecar_data)} notes.")

    def get_candidates(self, query: str, fetch_k: int) -> List[Dict]:
        """Fetch candidates using the underlying TwoStageRetriever."""
        # INCREASE candidate pool for TIMER to allow re-ranking to work effectively
        timer_fetch_k = max(fetch_k, 20) # Ensure minimal pool
        return super().search(query, k=timer_fetch_k, fetch_k=timer_fetch_k)

    def apply_scoring(self, candidates: List[Dict], query: str) -> List[Dict]:
        """Apply TIMER scoring to a list of candidates."""
        if not candidates:
            return []

        # 1. Identify Intent
        intent, confidence = self.scorer.classify_intent(query)
        beta = self.scorer.get_beta_intent(intent, confidence)
        
        # 2. Apply TIMER Scoring
        final_results = []
        
        for doc in candidates:
            # Merge Sidecar Metadata
            note_id = doc.get("note_id")
            offset_days = 0.0 # Default to Recent/New if unknown
            section = "Unknown"
            
            # Check Sidecar first (Virtual Graph)
            if note_id in self.sidecar_data:
                note_meta = self.sidecar_data[note_id]
                offset_days = float(note_meta.get("offset_days", 0))
                section = note_meta.get("section", section)
            # Fallback to Index Metadata (if we had it)
            elif "_metadata" in doc:
                meta = doc["_metadata"]
                offset_days = float(meta.get("offset_days", 0))
                section = meta.get("section", section)
            
            # Semantic Score from Reranker
            sem_score = doc.get("rerank_score", 0.0)
            
            # Compute Final Score
            timer_score = self.scorer.score_node(sem_score, offset_days, beta)
            
            # Enhance Doc
            doc["timer_score"] = timer_score
            doc["timer_intent"] = intent
            doc["timer_beta"] = beta
            doc["offset_days"] = offset_days
            doc["section"] = section
            
            final_results.append(doc)
            
        # 3. Re-sort by TIMER Score
        final_results.sort(key=lambda x: x["timer_score"], reverse=True)
        return final_results

    def search(self, query: str, k: int = 5, fetch_k: int = 20) -> List[Dict]:
        """Original Search Method (Wrapper)"""
        candidates = self.get_candidates(query, k * 10) # fetch_k logic from before
        ranked = self.apply_scoring(candidates, query)
        return ranked[:k]
