import math
import re
from typing import Tuple, Dict, Any

class TIMERScorer:
    """
    TIMER-Graph Scorer: Implements Temporal Intent-Modulated Entity Retrieval.
    
    Core Novelty:
    - Intent Classification (Current vs Historical vs Trend)
    - Adaptive Weighting (Beta) based on Intent
    - Negative Beta for Historical Queries to inverse the temporal decay penalty
    """
    
    # Hyperparameters from Implementation Plan
    BETA_CURRENT = 0.8
    BETA_TREND = 0.0
    BETA_HISTORICAL = -0.3
    
    ALPHA_SEMANTIC = 0.6
    
    # Default Lambda for temporal decay (approx half-life of 6 months = 180 days -> ln(2)/180 ~ 0.0038)
    # Plan suggested 0.005
    LAMBDA_DECAY = 0.005 

    # Confidence Threshold for Intent Router
    CONFIDENCE_THRESHOLD = 0.70

    def __init__(self, lambda_decay: float = LAMBDA_DECAY):
        self.lambda_decay = lambda_decay

    def classify_intent(self, query: str) -> Tuple[str, float]:
        """
        Classify query intent using regex heuristics.
        Returns (intent_label, confidence_score)
        """
        query_lower = query.lower()
        
        # 1. Historical Intent Patterns
        historical_patterns = [
            r"history of", r"previous", r"past", r"long[- ]term", r"chronic",
            r"diagnosed in", r"recurrence", r"family history", r"mother", r"father",
            r"brother", r"sister", r"uncle", r"aunt", r"grandfather", r"grandmother"
        ]
        
        # 2. Current Intent Patterns (Default/Acute)
        current_patterns = [
            r"current", r"now", r"today", r"latest", r"recent", r"discharge",
            r"admission", r"presenting", r"complaint", r"plan"
        ]
        
        # 3. Trend Intent Patterns
        trend_patterns = [
            r"change in", r"progression", r"worsening", r"improving", 
            r"over time", r"trend", r"fluctuation"
        ]

        # Scoring Logic (Simple Keyword Match Count)
        h_score = sum(1 for p in historical_patterns if re.search(p, query_lower))
        c_score = sum(1 for p in current_patterns if re.search(p, query_lower))
        t_score = sum(1 for p in trend_patterns if re.search(p, query_lower))
        
        total = h_score + c_score + t_score
        
        if total == 0:
            return "current", 0.5 # Default low confidence
            
        # Softmax-ish normalization
        scores = {"historical": h_score, "current": c_score, "trend": t_score}
        best_intent = max(scores, key=scores.get)
        confidence = scores[best_intent] / total
        
        return best_intent, confidence

    def get_beta_intent(self, intent: str, confidence: float) -> float:
        """
        Get Beta value based on intent and confidence.
        Falls back to Static (Beta=0) if confidence is low.
        """
        # If confidence is too low, we shouldn't bias the retrieval strongly
        # However, plan says "Static Temporal (Beta=0.5)" is a baseline.
        # Let's say fallback is 0.0 (Pure Semantic) or maybe a safe small positive decay (0.1)?
        # For this implementation, let's treat low confidence as "Neutral/Trend" -> Beta = 0.0
        
        if confidence < self.CONFIDENCE_THRESHOLD:
            return 0.0
            
        if intent == "current":
            return self.BETA_CURRENT
        elif intent == "historical":
            return self.BETA_HISTORICAL
        elif intent == "trend":
            return self.BETA_TREND
        
        return 0.0

    def compute_temporal_decay(self, offset_days: float) -> float:
        """
        Exponential Decay Function: exp(-lambda * t)
        Range: [0, 1]
        t=0 -> 1.0 (Recent)
        t=inf -> 0.0 (Old)
        """
        # Ensure non-negative time
        t = max(0.0, float(offset_days))
        return math.exp(-self.lambda_decay * t)

    def score_node(self, semantic_score: float, offset_days: float, beta: float) -> float:
        """
        Compute Final TIMER Score.
        Formula: Alpha * Semantic + Beta * Decay(t)
        
        Note: The Semantic Score from TwoStageRetriever/CrossEncoder is usually logits.
        We might need to sigmoid it to [0,1] if it's not already, OR just treat it as raw score.
        However, blending logits with decay [0,1] requires scaling.
        A CrossEncoder usually outputs sigmoid-like score [0,1] if trained with BCE, 
        or raw logits. BAAI/bge-reranker outputs logits (can be negative).
        
        To correspond to the plan "Alpha * Semantic", we assume comparable scales.
        If Semantic is [0,1] and Decay is [0,1]:
        Range with Beta=0.8: [0, 1.4]
        Range with Beta=-0.3: [-0.3, 0.7] (approx)
        
        This seems fine for ranking.
        """
        
        decay_score = self.compute_temporal_decay(offset_days)
        
        # We assume semantic_score is the primary driver.
        # If semantic_score is unbounded (logits), the decay impact might be negligible.
        # BGE Reranker scores are often in range [-10, 10].
        # Decay is [0, 1].
        # Beta=0.8 means max boost is +0.8.
        # If SemScore difference is typically 0.5-2.0, this is significant.
        # If SemScore difference is 0.01, this dominates.
        
        final_score = (self.ALPHA_SEMANTIC * semantic_score) + (beta * decay_score)
        
        return final_score
