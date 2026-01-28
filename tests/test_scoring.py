import pytest
import math
from app.research.retrieval.scoring import TIMERScorer

@pytest.fixture
def scorer():
    return TIMERScorer()

def test_classify_intent(scorer):
    # Test Historical
    query = "What is the patient's family history of cancer?"
    intent, conf = scorer.classify_intent(query)
    assert intent == "historical"
    assert conf > 0.3

    # Test Current
    query = "What is the discharge plan?"
    intent, conf = scorer.classify_intent(query)
    assert intent == "current"
    assert conf > 0.3

    # Test Trend
    query = "Is the blood pressure worsening over time?"
    intent, conf = scorer.classify_intent(query)
    assert intent == "trend"

    # Test Ambiguous (Default)
    query = "the patient"
    intent, conf = scorer.classify_intent(query)
    assert intent == "current"

def test_get_beta_intent(scorer):
    # High confidence
    assert scorer.get_beta_intent("current", 0.9) == 0.8
    assert scorer.get_beta_intent("historical", 0.9) == -0.3
    assert scorer.get_beta_intent("trend", 0.9) == 0.0

    # Low confidence -> Fallback (0.0)
    assert scorer.get_beta_intent("current", 0.1) == 0.0

def test_compute_temporal_decay(scorer):
    # t=0 -> 1.0
    assert scorer.compute_temporal_decay(0) == 1.0
    
    # t -> inf -> 0.0
    val = scorer.compute_temporal_decay(10000)
    assert val < 0.001
    assert val >= 0.0

def test_score_node_current(scorer):
    # Intent = Current (Beta = 0.8)
    # Recent (t=0, decay=1) vs Old (t=big, decay=0)
    # Recent should win given same semantic score
    
    sem = 1.0
    beta = 0.8
    
    score_recent = scorer.score_node(sem, 0, beta)      # 0.6*1 + 0.8*1 = 1.4
    score_old = scorer.score_node(sem, 1000, beta)      # 0.6*1 + 0.8*0 = 0.6
    
    assert score_recent > score_old

def test_score_node_historical(scorer):
    # Intent = Historical (Beta = -0.3)
    # Recent (t=0, decay=1) vs Old (t=big, decay=0)
    # Old should win (or simply be penalized LESS than Recent)
    
    sem = 1.0
    beta = -0.3
    
    score_recent = scorer.score_node(sem, 0, beta)      # 0.6*1 + (-0.3)*1 = 0.3
    score_old = scorer.score_node(sem, 1000, beta)      # 0.6*1 + (-0.3)*0 = 0.6
    
    assert score_old > score_recent
    print(f"\nHistorical Check: Old ({score_old}) > Recent ({score_recent})")

def test_mixed_semantic_and_temporal(scorer):
    # Verify that a SUPER relevant recent note might still beat a mildly relevant old note 
    # even in historical mode (we don't want to completely bury recent facts if they match well)
    
    beta = -0.3
    
    # Recent: Very relevant (Sem=2.0) -> Score = 1.2 + (-0.3) = 0.9
    # Old: Less relevant (Sem=1.0)    -> Score = 0.6 + 0      = 0.6
    
    score_recent_relevant = scorer.score_node(2.0, 0, beta)
    score_old_irrelevant = scorer.score_node(1.0, 1000, beta)
    
    assert score_recent_relevant > score_old_irrelevant
