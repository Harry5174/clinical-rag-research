import pytest
from unittest.mock import MagicMock, patch
import sys

# Mock torch and sentence_transformers BEFORE importing app modules
sys.modules["torch"] = MagicMock()
sys.modules["sentence_transformers"] = MagicMock()
sys.modules["faiss"] = MagicMock()
sys.modules["numpy"] = MagicMock()

from pathlib import Path
from app.research.retrieval.timer import TIMERRetriever

@pytest.fixture
def mock_sidecar(tmp_path):
    sidecar_path = tmp_path / "sidecar.json"
    import json
    data = {
        "metadata_version": "1.0",
        "reference_date": "2024-01-15",
        "notes": {
            "note_recent": {"offset_days": 0, "section": "Plan"},
            "note_old": {"offset_days": 1000, "section": "History"}
        }
    }
    with open(sidecar_path, "w") as f:
        json.dump(data, f)
    return sidecar_path

@pytest.fixture
def retriever(mock_sidecar):
    # Mocking init to avoid loading actual FAISS/Models
    with patch("app.research.retrieval.two_stage.TwoStageRetriever.__init__", return_value=None):
        retriever = TIMERRetriever(Path("."), mock_sidecar)
        # Manually init scorer since we mocked super init but TIMER init calls super then inits scorer
        # Wait, TIMER init calls super().__init__, then self.scorer = ...
        # If we mock super().__init__, the rest of TIMER.__init__ still runs! 
        # So self.scorer IS initialized.
        return retriever

def test_timer_reordering_historical(retriever):
    # Mock Candidates (Equal Semantic Score)
    candidates = [
        {"note_id": "note_recent", "text": "Recent stuff", "rerank_score": 1.0, "score": 0.5},
        {"note_id": "note_old", "text": "Old stuff", "rerank_score": 1.0, "score": 0.5}
    ]
    
    # Patch super().search to return these
    with patch("app.research.retrieval.two_stage.TwoStageRetriever.search", return_value=candidates):
        # Query: Historical -> Beta = -0.3
        results = retriever.search("History of past illness", k=2)
        
        # Expect Old > Recent
        # Old Score: 0.6*1.0 + (-0.3)*0 = 0.6
        # Recent Score: 0.6*1.0 + (-0.3)*1 = 0.3
        
        assert results[0]["note_id"] == "note_old"
        assert results[1]["note_id"] == "note_recent"
        assert results[0]["timer_beta"] == -0.3

def test_timer_reordering_current(retriever):
    # Mock Candidates (Equal Semantic Score)
    candidates = [
        {"note_id": "note_recent", "text": "Recent stuff", "rerank_score": 1.0, "score": 0.5},
        {"note_id": "note_old", "text": "Old stuff", "rerank_score": 1.0, "score": 0.5}
    ]
    
    with patch("app.research.retrieval.two_stage.TwoStageRetriever.search", return_value=candidates):
        # Query: Current -> Beta = 0.8
        results = retriever.search("Current discharge plan", k=2)
        
        # Expect Recent > Old
        # Recent Score: 0.6*1.0 + 0.8*1 = 1.4
        # Old Score: 0.6*1.0 + 0.8*0 = 0.6
        
        assert results[0]["note_id"] == "note_recent"
        assert results[1]["note_id"] == "note_old"
        assert results[0]["timer_beta"] == 0.8

def test_missing_sidecar_fallback(retriever):
    # Candidate NOT in sidecar -> offset=0 (Recent)
    candidates = [
        {"note_id": "note_unknown", "text": "Unknown", "rerank_score": 1.0, "score": 0.5},
        {"note_id": "note_old", "text": "Old stuff", "rerank_score": 1.0, "score": 0.5}
    ]
    
    with patch("app.research.retrieval.two_stage.TwoStageRetriever.search", return_value=candidates):
        # Query: Current -> Beta = 0.8
        # Unk (Offset=0) -> Score 1.4
        # Old (Offset=1000) -> Score 0.6
        results = retriever.search("Current plan", k=2)
        
        assert results[0]["note_id"] == "note_unknown"
        assert results[0]["offset_days"] == 0.0
