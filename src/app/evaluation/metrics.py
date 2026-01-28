from typing import List, Dict, Tuple, Any

def compute_recall_at_k(retrieved: List[Dict], target_note_id: str, k: int) -> int:
    """
    Compute Recall@k.
    Returns 1 if target_note_id is in top-k results, else 0.
    """
    top_k = retrieved[:k]
    for res in top_k:
        if res.get("note_id") == target_note_id:
            return 1
    return 0

def compute_TRA(
    retrieved: List[Dict],
    note_metadata: Dict[str, dict], # Sidecar Note Data (or from index if available)
    valid_window: Tuple[int, int]   # [min_days, max_days]
) -> float:
    """
    Temporal Relevance Alignment (TRA).
    Formula: |Retrieved ∩ Valid_Window| / |Retrieved|
    
    Args:
        retrieved: List of result dicts (must contain 'note_id')
        note_metadata: Dict mapping note_id to metadata (containing 'offset_days')
        valid_window: (min_offset, max_offset) inclusive
    """
    if not retrieved:
        return 0.0
        
    valid_count = 0
    min_days, max_days = valid_window
    
    for res in retrieved:
        note_id = res.get("note_id")
        # Get offset from sidecar or result itself
        if note_id in note_metadata:
            offset = note_metadata[note_id].get("offset_days")
        else:
            offset = res.get("offset_days")
            
        if offset is not None:
            if min_days <= float(offset) <= max_days:
                valid_count += 1
        # If offset is missing, we assume INVALID for safety? Or ignore?
        # For this experiment, all relevant notes should be in sidecar.
        
    return valid_count / len(retrieved)

def compute_temporal_leakage(
    retrieved: List[Dict],
    note_metadata: Dict[str, dict], 
    valid_window: Tuple[int, int]
) -> float:
    """
    Temporal Leakage Rate.
    Formula: 1.0 - TRA
    """
    return 1.0 - compute_TRA(retrieved, note_metadata, valid_window)
