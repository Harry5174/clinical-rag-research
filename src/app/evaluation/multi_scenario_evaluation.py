"""
Multi-Scenario Evaluation Framework for Medical RAG Pipeline
Tests retrieval performance across various real-world scenarios
"""

import json
import random
import time
import re
import numpy as np
import pandas as pd
from pathlib import Path
from typing import List, Dict, Tuple, Set
from dataclasses import dataclass
from collections import defaultdict
import hashlib
from abc import ABC, abstractmethod

# Add parent directory to path
import sys
sys.path.append(str(Path(__file__).resolve().parent.parent))
from app.retrieval.search import Retriever

@dataclass
class TestScenario:
    """Define a test scenario with specific requirements"""
    name: str
    description: str
    query_generator: callable
    success_criteria: callable
    weight: float = 1.0  # Importance weight for overall score

class ScenarioEvaluator(ABC):
    """Base class for scenario evaluators"""
    
    @abstractmethod
    def generate_query(self, chunk: Dict) -> str:
        pass
    
    @abstractmethod
    def evaluate_results(self, results: List[Dict], source_chunk: Dict) -> Dict:
        pass

# ==================== SCENARIO 1: Clinical Finding Retrieval ====================
class ClinicalFindingScenario(ScenarioEvaluator):
    """Test retrieval of specific clinical findings"""
    
    def generate_query(self, chunk: Dict) -> str:
        """Extract clinical findings and create query"""
        text = chunk['chunk_text']
        
        # Patterns for clinical findings
        patterns = [
            r'(?:showed|revealed|demonstrated|found|noted)\s+([^.]+)',
            r'(?:diagnosed with|positive for|negative for)\s+([^.]+)',
            r'(?:consistent with|suggestive of|indicative of)\s+([^.]+)',
        ]
        
        for pattern in patterns:
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                finding = match.group(1).strip()[:100]  # Limit length
                return f"patient with {finding}"
        
        # Fallback
        return None
    
    def evaluate_results(self, results: List[Dict], source_chunk: Dict) -> Dict:
        """Check if clinical finding was retrieved"""
        metrics = {
            'exact_match': False,
            'semantic_match': False,
            'relevance_score': 0.0
        }
        
        source_note = source_chunk['note_id']
        
        # Check for document match
        for result in results[:3]:  # Top 3
            if result['note_id'] == source_note:
                metrics['exact_match'] = True
                
                # Check semantic similarity
                source_terms = set(re.findall(r'\b\w+\b', source_chunk['chunk_text'].lower()))
                result_terms = set(re.findall(r'\b\w+\b', result['text'].lower()))
                
                if source_terms and result_terms:
                    overlap = len(source_terms & result_terms) / len(source_terms)
                    metrics['semantic_match'] = overlap > 0.3
                    metrics['relevance_score'] = overlap
                break
        
        return metrics

# ==================== SCENARIO 2: Medication Query ====================
class MedicationScenario(ScenarioEvaluator):
    """Test retrieval of medication information"""
    
    def generate_query(self, chunk: Dict) -> str:
        """Extract medication mentions"""
        text = chunk['chunk_text']
        
        # Find medication patterns
        med_patterns = [
            r'(\w+)\s+(\d+)\s*(?:mg|mcg|ml|units)',
            r'(?:started on|prescribed|given)\s+(\w+)',
            r'(?:continue|discontinue|hold)\s+(\w+)',
        ]
        
        medications = []
        for pattern in med_patterns:
            matches = re.findall(pattern, text, re.IGNORECASE)
            for match in matches:
                if isinstance(match, tuple):
                    medications.append(match[0])
                else:
                    medications.append(match)
        
        if medications:
            # Create query with medication names
            return f"patient on {' and '.join(medications[:2])}"
        
        return None
    
    def evaluate_results(self, results: List[Dict], source_chunk: Dict) -> Dict:
        """Check if medication context was retrieved"""
        metrics = {
            'medication_found': False,
            'dosage_found': False,
            'same_patient': False
        }
        
        source_meds = set(re.findall(r'(\w+)\s+\d+\s*(?:mg|mcg|ml)', 
                                     source_chunk['chunk_text'], re.IGNORECASE))
        
        for result in results[:5]:
            if result['note_id'] == source_chunk['note_id']:
                metrics['same_patient'] = True
                
                result_meds = set(re.findall(r'(\w+)\s+\d+\s*(?:mg|mcg|ml)', 
                                            result['text'], re.IGNORECASE))
                
                if source_meds & result_meds:
                    metrics['medication_found'] = True
                    
                    # Check if dosages match
                    source_doses = re.findall(r'\d+\s*(?:mg|mcg|ml)', 
                                            source_chunk['chunk_text'], re.IGNORECASE)
                    result_doses = re.findall(r'\d+\s*(?:mg|mcg|ml)', 
                                            result['text'], re.IGNORECASE)
                    
                    if set(source_doses) & set(result_doses):
                        metrics['dosage_found'] = True
                break
        
        return metrics

# ==================== SCENARIO 3: Temporal Query ====================
class TemporalScenario(ScenarioEvaluator):
    """Test retrieval with temporal context"""
    
    def generate_query(self, chunk: Dict) -> str:
        """Extract temporal information"""
        text = chunk['chunk_text']
        
        # Temporal patterns
        patterns = [
            r'(?:on|at)\s+(?:day|hospital day|HD|POD)\s*#?\s*(\d+)',
            r'(?:after|before|during)\s+([^,\.]+)',
            r'(?:initially|subsequently|eventually|finally)\s+([^,\.]+)',
        ]
        
        for pattern in patterns:
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                temporal_context = match.group(0)
                # Extract associated event
                event_match = re.search(r'(?:' + re.escape(temporal_context) + r')\s*,?\s*([^.]+)', 
                                      text, re.IGNORECASE)
                if event_match:
                    return event_match.group(1).strip()[:100]
        
        return None
    
    def evaluate_results(self, results: List[Dict], source_chunk: Dict) -> Dict:
        """Check temporal context preservation"""
        metrics = {
            'temporal_match': False,
            'sequence_preserved': False,
            'same_admission': False
        }
        
        source_note = source_chunk['note_id']
        
        # Check for temporal markers
        temporal_markers = ['day', 'HD', 'POD', 'initially', 'subsequently', 
                           'after', 'before', 'during']
        
        for result in results[:3]:
            if result['note_id'] == source_note:
                metrics['same_admission'] = True
                
                # Check if temporal context is preserved
                for marker in temporal_markers:
                    if marker in source_chunk['chunk_text'] and marker in result['text']:
                        metrics['temporal_match'] = True
                        break
                
                # Check sequence preservation (if chunk indices available)
                if 'chunk_index' in result and 'chunk_index' in source_chunk:
                    index_diff = abs(result['chunk_index'] - source_chunk['chunk_index'])
                    metrics['sequence_preserved'] = index_diff <= 2
                
                break
        
        return metrics

# ==================== SCENARIO 4: Diagnostic Test Results ====================
class DiagnosticResultScenario(ScenarioEvaluator):
    """Test retrieval of diagnostic test results"""
    
    def generate_query(self, chunk: Dict) -> str:
        """Extract diagnostic test mentions"""
        text = chunk['chunk_text']
        
        # Diagnostic test patterns
        test_patterns = [
            r'(?:CT|MRI|XR|CXR|ECG|EKG|Echo)\s+(?:showed|revealed|demonstrated)',
            r'(?:blood|urine|sputum)\s+(?:culture|test)',
            r'(?:WBC|Hgb|Plt|Cr|BUN|Na|K|Glucose)\s*:?\s*\d+',
        ]
        
        for pattern in test_patterns:
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                return match.group(0)
        
        return None
    
    def evaluate_results(self, results: List[Dict], source_chunk: Dict) -> Dict:
        """Check if test results were retrieved"""
        metrics = {
            'test_found': False,
            'result_found': False,
            'value_match': False
        }
        
        # Extract test names and values from source
        source_tests = re.findall(r'(?:CT|MRI|XR|CXR|ECG|EKG|Echo|WBC|Hgb|Plt|Cr|BUN)', 
                                 source_chunk['chunk_text'], re.IGNORECASE)
        source_values = re.findall(r'\d+\.?\d*', source_chunk['chunk_text'])
        
        for result in results[:5]:
            result_tests = re.findall(r'(?:CT|MRI|XR|CXR|ECG|EKG|Echo|WBC|Hgb|Plt|Cr|BUN)', 
                                     result['text'], re.IGNORECASE)
            
            if set(source_tests) & set(result_tests):
                metrics['test_found'] = True
                
                # Check for result values
                result_values = re.findall(r'\d+\.?\d*', result['text'])
                if set(source_values) & set(result_values):
                    metrics['value_match'] = True
                    metrics['result_found'] = True
                break
        
        return metrics

# ==================== SCENARIO 5: Cross-Document Coherence ====================
class CrossDocumentScenario(ScenarioEvaluator):
    """Test ability to maintain context across chunks"""
    
    def generate_query(self, chunk: Dict) -> str:
        """Generate query that might span chunks"""
        text = chunk['chunk_text']
        
        # Look for references to other parts
        patterns = [
            r'as (?:mentioned|described|noted) (?:above|previously|earlier)',
            r'(?:see|refer to) (?:below|above|section)',
            r'(?:following|subsequent to) (?:the|his|her)',
        ]
        
        for pattern in patterns:
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                # Get surrounding context
                start = max(0, match.start() - 50)
                end = min(len(text), match.end() + 50)
                return text[start:end]
        
        return None
    
    def evaluate_results(self, results: List[Dict], source_chunk: Dict) -> Dict:
        """Check cross-document coherence"""
        metrics = {
            'same_document': False,
            'adjacent_chunks': False,
            'context_preserved': False
        }
        
        source_note = source_chunk['note_id']
        source_index = source_chunk.get('chunk_index', -1)
        
        retrieved_indices = []
        for result in results[:5]:
            if result['note_id'] == source_note:
                metrics['same_document'] = True
                
                if 'chunk_index' in result:
                    retrieved_indices.append(result['chunk_index'])
        
        if retrieved_indices and source_index >= 0:
            # Check if adjacent chunks were retrieved
            for idx in retrieved_indices:
                if abs(idx - source_index) <= 1:
                    metrics['adjacent_chunks'] = True
                    metrics['context_preserved'] = True
                    break
        
        return metrics

# ==================== MAIN EVALUATION FRAMEWORK ====================
class ComprehensiveEvaluator:
    """Main evaluation framework orchestrating all scenarios"""
    
    def __init__(self, retriever: Retriever, test_data: List[Dict]):
        self.retriever = retriever
        self.test_data = test_data
        
        # Initialize scenarios
        self.scenarios = {
            'clinical_findings': ClinicalFindingScenario(),
            'medications': MedicationScenario(),
            'temporal': TemporalScenario(),
            'diagnostics': DiagnosticResultScenario(),
            'cross_document': CrossDocumentScenario(),
        }
        
        # Scenario weights for overall scoring
        self.scenario_weights = {
            'clinical_findings': 0.25,
            'medications': 0.20,
            'temporal': 0.15,
            'diagnostics': 0.20,
            'cross_document': 0.20,
        }
    
    def run_comprehensive_evaluation(self, n_samples_per_scenario: int = 20):
        """Run all evaluation scenarios"""
        
        print("\n" + "="*70)
        print("COMPREHENSIVE MULTI-SCENARIO EVALUATION")
        print("="*70)
        
        all_results = {}
        overall_scores = {}
        
        for scenario_name, evaluator in self.scenarios.items():
            print(f"\nEvaluating Scenario: {scenario_name.replace('_', ' ').title()}")
            print("-" * 50)
            
            scenario_results = self._evaluate_scenario(
                evaluator, 
                n_samples_per_scenario
            )
            
            all_results[scenario_name] = scenario_results
            
            # Calculate scenario score
            if scenario_results['valid_queries'] > 0:
                score = scenario_results['success_rate']
                overall_scores[scenario_name] = score
                
                # Print scenario results
                print(f"  ✓ Valid queries: {scenario_results['valid_queries']}")
                print(f"  ✓ Success rate: {score:.1f}%")
                print(f"  ✓ Avg latency: {scenario_results['avg_latency']:.3f}s")
            else:
                print(f"No valid queries generated for this scenario")
        
        # Calculate weighted overall score
        if overall_scores:
            weighted_score = sum(
                score * self.scenario_weights.get(scenario, 0.2)
                for scenario, score in overall_scores.items()
            )
            
            print("\n" + "="*70)
            print("OVERALL EVALUATION RESULTS")
            print("="*70)
            
            self._print_summary_table(all_results)
            
            print(f"\nWEIGHTED OVERALL SCORE: {weighted_score:.1f}%")
            
            # Provide assessment
            self._provide_assessment(weighted_score, all_results)
        
        return all_results
    
    def _evaluate_scenario(self, evaluator: ScenarioEvaluator, 
                          n_samples: int) -> Dict:
        """Evaluate a single scenario"""
        
        results = {
            'total_attempts': 0,
            'valid_queries': 0,
            'successes': 0,
            'failures': 0,
            'latencies': [],
            'detailed_metrics': []
        }
        
        # Sample test data
        samples = random.sample(self.test_data, 
                              min(n_samples * 3, len(self.test_data)))
        
        for chunk in samples:
            if results['valid_queries'] >= n_samples:
                break
            
            results['total_attempts'] += 1
            
            # Generate query
            query = evaluator.generate_query(chunk)
            if not query:
                continue
            
            results['valid_queries'] += 1
            
            # Perform retrieval
            start_time = time.time()
            retrieved = self.retriever.search(query, k=5)
            latency = time.time() - start_time
            results['latencies'].append(latency)
            
            # Evaluate results
            metrics = evaluator.evaluate_results(retrieved, chunk)
            results['detailed_metrics'].append(metrics)
            
            # Determine success
            if self._is_success(metrics):
                results['successes'] += 1
            else:
                results['failures'] += 1
        
        # Calculate aggregated metrics
        if results['valid_queries'] > 0:
            results['success_rate'] = (results['successes'] / results['valid_queries']) * 100
            results['avg_latency'] = np.mean(results['latencies'])
        else:
            results['success_rate'] = 0
            results['avg_latency'] = 0
        
        return results
    
    def _is_success(self, metrics: Dict) -> bool:
        """Determine if retrieval was successful based on metrics"""
        # Success criteria varies by scenario
        if 'exact_match' in metrics:
            return metrics['exact_match']
        elif 'medication_found' in metrics:
            return metrics['medication_found'] and metrics['same_patient']
        elif 'temporal_match' in metrics:
            return metrics['same_admission']
        elif 'test_found' in metrics:
            return metrics['test_found'] and metrics['result_found']
        elif 'same_document' in metrics:
            return metrics['same_document'] and metrics['context_preserved']
        return False
    
    def _print_summary_table(self, results: Dict):
        """Print formatted summary table"""
        
        print("\n📊 SCENARIO PERFORMANCE SUMMARY")
        print("-" * 70)
        print(f"{'Scenario':<20} {'Success Rate':<15} {'Queries':<10} {'Latency':<10}")
        print("-" * 70)
        
        for scenario, data in results.items():
            if data['valid_queries'] > 0:
                print(f"{scenario.replace('_', ' ').title():<20} "
                      f"{data['success_rate']:<15.1f}% "
                      f"{data['valid_queries']:<10} "
                      f"{data['avg_latency']:<10.3f}s")
    
    def _provide_assessment(self, overall_score: float, detailed_results: Dict):
        """Provide detailed assessment and recommendations"""
        
        print("\n" + "="*70)
        print("SYSTEM ASSESSMENT & RECOMMENDATIONS")
        print("="*70)
        
        if overall_score >= 80:
            print("\nEXCELLENT: System is production-ready")
            print("   • Strong performance across all scenarios")
            print("   • Consider A/B testing with users")
            
        elif overall_score >= 70:
            print("\nGOOD: System is functional with room for improvement")
            
            # Identify weak scenarios
            weak_scenarios = [
                scenario for scenario, data in detailed_results.items()
                if data['valid_queries'] > 0 and data['success_rate'] < 70
            ]
            
            if weak_scenarios:
                print(f"\n   Weak areas: {', '.join(weak_scenarios)}")
                print("\n   Recommendations:")
                
                if 'medications' in weak_scenarios:
                    print("   • Add medical abbreviation expansion")
                    print("   • Implement fuzzy matching for drug names")
                
                if 'temporal' in weak_scenarios:
                    print("   • Enhance temporal expression parsing")
                    print("   • Add sequence-aware chunking")
                
                if 'cross_document' in weak_scenarios:
                    print("   • Increase chunk overlap")
                    print("   • Add document-level context embeddings")
        
        elif overall_score >= 60:
            print("\nMODERATE: System needs optimization")
            print("\n   Critical improvements needed:")
            print("   • Review chunking strategy (current may be fragmenting context)")
            print("   • Consider domain-specific embedding model")
            print("   • Implement query expansion and synonyms")
            
        else:
            print("\nNEEDS WORK: Significant improvements required")
            print("\n   Immediate actions:")
            print("   • Reduce chunk size to 150-200 words")
            print("   • Increase overlap to 100+ words")
            print("   • Add comprehensive data preprocessing")
            print("   • Consider switching to medical-specific embeddings")

def main():
    """Main evaluation entry point"""
    
    # Configuration
    BASE_DIR = Path(__file__).resolve().parent.parent.parent
    PROCESSED_JSON = BASE_DIR / "data" / "processed" / "discharge.json"
    VECTOR_DIR = BASE_DIR / "data" / "vector_store"
    
    print("Loading test data and retriever...")
    
    # Load data
    with open(PROCESSED_JSON, 'r') as f:
        test_data = json.load(f)
    
    print(f"Loaded {len(test_data)} chunks for evaluation")
    
    # Initialize retriever
    retriever = Retriever(VECTOR_DIR)
    
    # Run evaluation
    evaluator = ComprehensiveEvaluator(retriever, test_data)
    results = evaluator.run_comprehensive_evaluation(n_samples_per_scenario=25)
    
    # Save results
    output_file = Path("multi_scenario_results.json")
    with open(output_file, 'w') as f:
        # Convert numpy types for JSON serialization
        clean_results = {}
        for scenario, data in results.items():
            clean_results[scenario] = {
                k: float(v) if isinstance(v, np.number) else v
                for k, v in data.items()
                if k != 'detailed_metrics'  # Skip detailed metrics for file
            }
        json.dump(clean_results, f, indent=2)
    
    print(f"\nResults saved to {output_file}")
    print("\nEvaluation complete! Use these insights to optimize your system.")

if __name__ == "__main__":
    main()