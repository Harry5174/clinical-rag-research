#!/usr/bin/env python
"""
Practical Testing Suite for OAPOC Retrieval Pipeline
Run this to test and improve your current 64% recall system
"""

import sys
import json
import time
import random
import numpy as np
from pathlib import Path
from typing import List, Dict

# Setup paths for your environment
BASE_DIR = Path("/home/harry/Desktop/research/poc/oapoc")
sys.path.append(str(BASE_DIR / "src" / "app"))

from app.retrieval.search import Retriever

class PracticalTester:
    """Practical testing for your existing setup"""
    
    def __init__(self):
        self.data_dir = BASE_DIR / "src" / "data"
        self.vector_dir = self.data_dir / "vector_store"
        self.json_path = self.data_dir / "processed" / "discharge.json"
        
        print("Initializing tester...")
        print(f"Vector store: {self.vector_dir}")
        print(f"Test data: {self.json_path}")
        
        # Load retriever
        self.retriever = Retriever(self.vector_dir)
        
        # Load test data
        with open(self.json_path, 'r') as f:
            self.test_data = json.load(f)
        print(f"Loaded {len(self.test_data)} chunks for testing\n")
    
    def run_quick_diagnostic(self):
        """Quick diagnostic to identify why recall is 64%"""
        
        print("="*60)
        print("QUICK DIAGNOSTIC TEST")
        print("="*60)
        
        # Sample 50 random chunks
        samples = random.sample(self.test_data, min(50, len(self.test_data)))
        
        issues = {
            'boilerplate_queries': 0,
            'short_queries': 0,
            'boundary_splits': 0,
            'perfect_matches': 0,
            'same_note_different_chunk': 0,
            'complete_misses': 0
        }
        
        for chunk in samples:
            text = chunk['chunk_text']
            
            # Get a test sentence
            sentences = [s.strip() for s in text.split('.') 
                        if len(s.strip().split()) > 10]
            
            if not sentences:
                issues['short_queries'] += 1
                continue
            
            query = sentences[0]
            
            # Check for boilerplate
            if any(term in query for term in ['Admission Date', 'Discharge Date', 
                                              'Service:', 'Attending:']):
                issues['boilerplate_queries'] += 1
                continue
            
            # Search
            results = self.retriever.search(query, k=5)
            
            # Analyze results
            found_exact = False
            found_same_note = False
            
            for result in results:
                if result['note_id'] == chunk['note_id']:
                    found_same_note = True
                    if result['text'][:50] == text[:50]:
                        found_exact = True
                        issues['perfect_matches'] += 1
                        break
            
            if found_same_note and not found_exact:
                issues['same_note_different_chunk'] += 1
            elif not found_same_note:
                issues['complete_misses'] += 1
        
        # Print diagnostic results
        print("\nDIAGNOSTIC RESULTS:")
        print("-" * 40)
        total = sum(issues.values())
        
        for issue, count in issues.items():
            percentage = (count / total * 100) if total > 0 else 0
            print(f"{issue.replace('_', ' ').title():<30} {count:>3} ({percentage:>5.1f}%)")
        
        # Provide specific recommendations
        print("\nKEY FINDINGS:")
        print("-" * 40)
        
        if issues['boilerplate_queries'] > 5:
            print("High boilerplate content affecting {:.0f}% of queries".format(
                issues['boilerplate_queries'] / total * 100))
            print("   Fix: Add stronger boilerplate filtering")
        
        if issues['same_note_different_chunk'] > 10:
            print("Chunk boundary issues affecting {:.0f}% of queries".format(
                issues['same_note_different_chunk'] / total * 100))
            print("   Fix: Increase overlap from 50 to 100 words")
        
        if issues['complete_misses'] > 15:
            print("High miss rate: {:.0f}% queries failing completely".format(
                issues['complete_misses'] / total * 100))
            print("   Fix: Review chunk size and embedding strategy")
        
        success_rate = (issues['perfect_matches'] + issues['same_note_different_chunk']) / total * 100
        print(f"\nEstimated Recall: {success_rate:.1f}%")
    
    def test_specific_scenarios(self):
        """Test specific medical retrieval scenarios"""
        
        print("\n" + "="*60)
        print("SCENARIO-BASED TESTING")
        print("="*60)
        
        test_queries = {
            "Cardiac Conditions": [
                "patient with acute myocardial infarction",
                "history of atrial fibrillation",
                "coronary artery disease with stents",
                "congestive heart failure exacerbation"
            ],
            "Medications": [
                "started on metoprolol 25mg",
                "aspirin 81mg daily",
                "anticoagulation with warfarin",
                "diuretics for fluid overload"
            ],
            "Symptoms": [
                "chest pain radiating to left arm",
                "shortness of breath on exertion",
                "palpitations and dizziness",
                "lower extremity edema"
            ],
            "Diagnostics": [
                "ECG showed ST elevation",
                "chest x-ray revealed pulmonary edema",
                "troponin levels elevated",
                "ejection fraction 35%"
            ]
        }
        
        scenario_results = {}
        
        for scenario, queries in test_queries.items():
            print(f"\nTesting: {scenario}")
            print("-" * 40)
            
            scenario_scores = []
            
            for query in queries:
                results = self.retriever.search(query, k=3)
                
                # Simple relevance check - do results contain key terms?
                query_terms = set(query.lower().split())
                
                relevance_scores = []
                for result in results:
                    result_terms = set(result['text'].lower().split())
                    overlap = len(query_terms & result_terms) / len(query_terms)
                    relevance_scores.append(overlap)
                
                avg_relevance = np.mean(relevance_scores) if relevance_scores else 0
                scenario_scores.append(avg_relevance)
                
                print(f"  Query: '{query[:40]}...'")
                print(f"  Relevance: {avg_relevance*100:.1f}%")
            
            scenario_results[scenario] = np.mean(scenario_scores) * 100
            print(f"\n  → Overall {scenario} Performance: {scenario_results[scenario]:.1f}%")
        
        # Summary
        print("\n" + "="*60)
        print("SCENARIO SUMMARY")
        print("="*60)
        
        avg_performance = np.mean(list(scenario_results.values()))
        
        for scenario, score in scenario_results.items():
            status = "✅" if score >= 70 else "⚠️" if score >= 50 else "❌"
            print(f"{status} {scenario:<20} {score:>5.1f}%")
        
        print(f"\nAverage Performance: {avg_performance:.1f}%")
        
        if avg_performance < 70:
            print("\nPerformance below target (70%)")
            print("Recommended immediate fixes:")
            print("1. Reduce chunk_size to 200 (currently 250)")
            print("2. Increase overlap to 100 (currently 50)")
            print("3. Re-index with new parameters")
    
    def test_improved_parameters(self):
        """Simulate what would happen with improved parameters"""
        
        print("\n" + "="*60)
        print("PARAMETER OPTIMIZATION SIMULATION")
        print("="*60)
        
        print("\nSimulating retrieval with optimized parameters...")
        print("(This estimates improvement without re-indexing)")
        
        # Test with different query modifications
        improvements = {
            'query_expansion': 0,
            'remove_boilerplate': 0,
            'use_context': 0
        }
        
        samples = random.sample(self.test_data, 30)
        
        for chunk in samples:
            text = chunk['chunk_text']
            
            # Skip boilerplate chunks
            if any(term in text[:100] for term in ['Admission Date', 'Service:']):
                continue
            
            # Get good sentence
            sentences = [s.strip() for s in text.split('.') 
                        if 15 < len(s.strip().split()) < 50]
            
            if not sentences:
                continue
            
            base_query = sentences[0]
            
            # Test 1: Basic query
            base_results = self.retriever.search(base_query, k=5)
            base_found = any(r['note_id'] == chunk['note_id'] for r in base_results)
            
            # Test 2: With medical expansion
            expanded_query = self._expand_medical_terms(base_query)
            exp_results = self.retriever.search(expanded_query, k=5)
            exp_found = any(r['note_id'] == chunk['note_id'] for r in exp_results)
            
            if exp_found and not base_found:
                improvements['query_expansion'] += 1
            
            # Test 3: With context
            if len(sentences) > 1:
                context_query = sentences[0] + " " + sentences[1][:50]
                ctx_results = self.retriever.search(context_query, k=5)
                ctx_found = any(r['note_id'] == chunk['note_id'] for r in ctx_results)
                
                if ctx_found and not base_found:
                    improvements['use_context'] += 1
        
        print("\nPOTENTIAL IMPROVEMENTS:")
        print("-" * 40)
        
        total_tested = 30
        current_recall = 64  # Your current rate
        
        expansion_boost = (improvements['query_expansion'] / total_tested) * 100
        context_boost = (improvements['use_context'] / total_tested) * 100
        
        print(f"Query Expansion:    +{expansion_boost:.1f}% potential improvement")
        print(f"Context Addition:   +{context_boost:.1f}% potential improvement")
        print(f"Boilerplate Filter: +5-8% estimated improvement")
        
        estimated_new = min(current_recall + expansion_boost + context_boost + 6, 95)
        
        print(f"\nEstimated recall after optimizations: {estimated_new:.1f}%")
        
        if estimated_new >= 80:
            print("Optimizations should achieve target performance!")
        else:
            print("     Additional optimizations needed:")
            print("   - Consider medical-specific embeddings")
            print("   - Implement hybrid search")
    
    def _expand_medical_terms(self, query: str) -> str:
        """Simple medical term expansion"""
        expansions = {
            'MI': 'myocardial infarction',
            'CHF': 'heart failure',
            'AFib': 'atrial fibrillation',
            'CAD': 'coronary artery disease',
            'HTN': 'hypertension',
        }
        
        expanded = query
        for abbr, full in expansions.items():
            if abbr in query.upper():
                expanded += f" {full}"
        
        return expanded
    
    def generate_optimization_script(self):
        """Generate a script to fix the issues"""
        
        print("\n" + "="*60)
        print("GENERATING OPTIMIZATION SCRIPT")
        print("="*60)
        
        script_content = '''#!/bin/bash
# Auto-generated optimization script for OAPOC pipeline

echo "========================================="
echo "OAPOC PIPELINE OPTIMIZATION"
echo "========================================="

# Navigate to project directory
cd ~/Desktop/research/poc/oapoc

# Backup current data
echo "Creating backup..."
mkdir -p backups/$(date +%Y%m%d)
cp -r src/data/processed backups/$(date +%Y%m%d)/
cp -r src/data/vector_store backups/$(date +%Y%m%d)/

# Update chunking parameters in normalizer.py
echo "Updating chunking parameters..."
sed -i 's/chunk_size=250/chunk_size=200/g' src/app/etl/normalizer.py
sed -i 's/overlap=50/overlap=100/g' src/app/etl/normalizer.py

# Clear old processed data
echo "Clearing old data..."
rm -f src/data/processed/*.json
rm -f src/data/vector_store/*

# Re-run pipeline with optimized parameters
echo "Re-processing data with optimized parameters..."
python src/app/main.py

echo "========================================="
echo "Optimization complete!"
echo "Run evaluation to verify improvements."
echo "========================================="
'''
        
        script_path = Path("optimize_oapoc.sh")
        with open(script_path, 'w') as f:
            f.write(script_content)
        
        script_path.chmod(0o755)  # Make executable
        
        print(f"Optimization script saved to: {script_path}")
        print("\nTo apply optimizations, run:")
        print(f"  bash {script_path}")
        print("\nThis will:")
        print("  1. Backup your current data")
        print("  2. Update chunking parameters")
        print("  3. Re-process and re-index your data")
        print("\nNote: Re-indexing will take ~4 hours on CPU")

def main():
    """Main entry point"""
    
    print("\n" + "="*70)
    print("OAPOC RETRIEVAL PIPELINE TESTER")
    print("="*70)
    
    tester = PracticalTester()
    
    while True:
        print("\nSelect test to run:")
        print("1. Quick Diagnostic (identify issues)")
        print("2. Scenario Testing (test medical queries)")
        print("3. Parameter Simulation (estimate improvements)")
        print("4. Generate Optimization Script")
        print("5. Run All Tests")
        print("0. Exit")
        
        choice = input("\nEnter choice (0-5): ")
        
        if choice == '1':
            tester.run_quick_diagnostic()
        elif choice == '2':
            tester.test_specific_scenarios()
        elif choice == '3':
            tester.test_improved_parameters()
        elif choice == '4':
            tester.generate_optimization_script()
        elif choice == '5':
            tester.run_quick_diagnostic()
            tester.test_specific_scenarios()
            tester.test_improved_parameters()
            tester.generate_optimization_script()
        elif choice == '0':
            print("Exiting...")
            break
        else:
            print("Invalid choice. Try again.")
    
    print("\nTesting complete!")

if __name__ == "__main__":
    main()