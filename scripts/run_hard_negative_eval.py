"""
Evaluates TIMER-Graph on hard negative scenarios.
Target: Demonstrate >20% improvement over semantic baseline.
"""

import json
import pandas as pd
from pathlib import Path
import sys

# Add src to path
sys.path.append(str(Path(__file__).parent.parent / "src"))

from app.research.retrieval.scoring import TIMERScorer
# from app.research.retrieval.timer import TIMERRetriever # Not strictly needed if we simulate
from app.evaluation.metrics import compute_TRA, compute_recall_at_k
from typing import Dict, List, Any

class HardNegativeEvaluator:
    """
    Evaluates retrieval performance on hard negative scenarios.
    """
    
    def __init__(
        self,
        hard_negatives_path: str = "data/mocks/combined_hard_negatives.json",
        results_dir: str = "results/phase5"
    ):
        self.hard_negatives_path = Path(hard_negatives_path)
        if not self.hard_negatives_path.exists():
            raise FileNotFoundError(f"Dataset not found at {hard_negatives_path}")
            
        with open(hard_negatives_path, 'r') as f:
            self.data = json.load(f)
        
        self.results_dir = Path(results_dir)
        self.results_dir.mkdir(parents=True, exist_ok=True)
        
        self.scorer = TIMERScorer(lambda_decay=0.005)
    
    def simulate_semantic_baseline(self, query_data: Dict) -> Dict:
        """
        Simulates semantic-only retrieval (β=0 for all intents).
        Uses provided semantic_scores to rank.
        """
        notes = query_data['notes']
        
        # Sort by semantic score only
        # If scores are identical, this sort is unstable/arbitrary in Python (stability implies preserving original order)
        # But we want to simulate RANDOMNESS or at least lack of temporal preference if scores match.
        # Python sort is stable. So it will preserve order in list.
        # To simulate "random selection when scores tie", we could shuffle?
        # But let's just stick to the list order, which is "random" in dataset construction perspective.
        # Actually HardNegativeGenerator puts New then Old or something?
        # Let's trust the semantic score.
        
        ranked = sorted(notes, key=lambda x: x['semantic_score'], reverse=True)
        
        return {
            "top_retrieval": ranked[0]['id'],
            "ranking": [n['id'] for n in ranked],
            "scores": {n['id']: n['semantic_score'] for n in ranked}
        }
    
    def simulate_timer_retrieval(self, query_data: Dict) -> Dict:
        """
        Simulates TIMER retrieval with intent-modulated scoring.
        """
        notes = query_data['notes']
        query_text = query_data['text']
        
        # Classify intent
        intent, confidence = self.scorer.classify_intent(query_text)
        beta = self.scorer.get_beta_intent(intent, confidence)
        
        # Compute TIMER scores
        timer_scores = {}
        for note in notes:
            score = self.scorer.score_node(
                semantic_score=note['semantic_score'],
                offset_days=note['offset_days'],
                beta=beta
            )
            timer_scores[note['id']] = score
        
        # Rank by TIMER score
        ranked = sorted(notes, key=lambda x: timer_scores[x['id']], reverse=True)
        
        return {
            "top_retrieval": ranked[0]['id'],
            "ranking": [n['id'] for n in ranked],
            "scores": timer_scores,
            "intent": intent,
            "confidence": confidence
        }
    
    def evaluate_scenario(self, scenario_name: str, queries: List[Dict]) -> pd.DataFrame:
        """
        Evaluates all queries in a scenario.
        """
        results = []
        
        for i, query in enumerate(queries):
            # Baseline
            baseline_result = self.simulate_semantic_baseline(query)
            baseline_correct = (baseline_result['top_retrieval'] == query['expected_retrieval'])
            
            # TIMER
            timer_result = self.simulate_timer_retrieval(query)
            timer_correct = (timer_result['top_retrieval'] == query['expected_retrieval'])
            
            if i == 0:
                print(f"DEBUG: Query: {query['text']}")
                print(f"DEBUG: Intent Detected: {timer_result['intent']}")
                print(f"DEBUG: Timer Scores: {timer_result['scores']}")
                print(f"DEBUG: Expected: {query['expected_retrieval']}, Got: {timer_result['top_retrieval']}")
            
            results.append({
                "query_id": query['id'],
                "query_text": query['text'],
                "intent": query['intent'],
                "expected_note": query['expected_retrieval'],
                "baseline_retrieval": baseline_result['top_retrieval'],
                "baseline_correct": baseline_correct,
                "timer_retrieval": timer_result['top_retrieval'],
                "timer_correct": timer_correct,
                "timer_intent_detected": timer_result['intent'],
                "improvement": timer_correct and not baseline_correct
            })
        
        return pd.DataFrame(results)
    
    def run_full_evaluation(self) -> Dict:
        """
        Runs evaluation on all scenarios and generates report.
        """
        all_results = {}
        
        for scenario_name, queries in self.data['scenarios'].items():
            print(f"\n{'='*60}")
            print(f"Evaluating: {scenario_name}")
            print(f"{'='*60}")
            
            df = self.evaluate_scenario(scenario_name, queries)
            all_results[scenario_name] = df
            
            # Print summary
            baseline_acc = df['baseline_correct'].mean()
            timer_acc = df['timer_correct'].mean()
            improvement = timer_acc - baseline_acc
            
            print(f"\n📊 Results:")
            print(f"   Baseline Accuracy: {baseline_acc:.2%}")
            print(f"   TIMER Accuracy:    {timer_acc:.2%}")
            print(f"   Improvement:       {improvement:+.2%}")
            
            # Save scenario results
            output_path = self.results_dir / f"{scenario_name}_results.csv"
            df.to_csv(output_path, index=False)
            print(f"   💾 Saved to: {output_path}")
        
        # Generate summary report
        self.generate_summary_report(all_results)
        
        return all_results
    
    def generate_summary_report(self, all_results: Dict):
        """
        Generates publication-ready summary table.
        """
        summary = []
        
        for scenario, df in all_results.items():
            summary.append({
                "Scenario": scenario.replace('_', ' ').title(),
                "Total Queries": len(df),
                "Baseline Accuracy": f"{df['baseline_correct'].mean():.2%}",
                "TIMER Accuracy": f"{df['timer_correct'].mean():.2%}",
                "Improvement": f"{(df['timer_correct'].mean() - df['baseline_correct'].mean()):+.2%}",
                "TIMER Wins": int(df['improvement'].sum())
            })
        
        summary_df = pd.DataFrame(summary)
        
        # Add overall row
        total_queries = sum(len(df) for df in all_results.values())
        if total_queries > 0:
            total_baseline = sum(df['baseline_correct'].sum() for df in all_results.values()) / total_queries
            total_timer = sum(df['timer_correct'].sum() for df in all_results.values()) / total_queries
            wins = sum(df['improvement'].sum() for df in all_results.values())
        else:
            total_baseline, total_timer, wins = 0, 0, 0
            
        summary_df.loc[len(summary_df)] = {
            "Scenario": "**OVERALL**",
            "Total Queries": total_queries,
            "Baseline Accuracy": f"{total_baseline:.2%}",
            "TIMER Accuracy": f"{total_timer:.2%}",
            "Improvement": f"{(total_timer - total_baseline):+.2%}",
            "TIMER Wins": int(wins)
        }
        
        # Save
        summary_path = self.results_dir / "summary_table.csv"
        summary_df.to_csv(summary_path, index=False)
        
        # Print
        print("\n" + "="*80)
        print("📋 PHASE 5 SUMMARY REPORT")
        print("="*80)
        print(summary_df.to_string(index=False))
        print(f"\n💾 Saved to: {summary_path}")
        
        # Success check
        if total_timer - total_baseline >= 0.20:
            print("\n✅ SUCCESS: TIMER achieves >20% improvement target!")
        else:
            print(f"\n⚠️  TIMER improvement ({total_timer - total_baseline:.2%}) below 20% target")
            print("   Consider: Adjusting β values or λ decay rate")


if __name__ == "__main__":
    try:
        evaluator = HardNegativeEvaluator()
        results = evaluator.run_full_evaluation()
    except Exception as e:
        print(f"Evaluation Failed: {e}")
