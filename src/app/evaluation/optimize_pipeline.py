#!/usr/bin/env python
"""
Implementation Script: Advanced Chunking + Multi-Scenario Testing
For InsightAI RAG Pipeline Optimization
"""

import sys
import json
import time
from typing import Dict
import shutil
from pathlib import Path
from datetime import datetime

# Add to path for imports
sys.path.append(str(Path(__file__).resolve().parent))

from app.indexing.advanced_chunking_strategy import (
    ChunkingStrategy, ChunkConfig, AdvancedChunker,
    process_with_multiple_strategies, analyze_chunking_strategies
)

from app.evaluation.multi_scenario_evaluation import ComprehensiveEvaluator

# Your existing modules
sys.path.append(str(Path(__file__).resolve().parent.parent.parent))
from src.app.indexing.builder import build_index
from src.app.retrieval.search import Retriever

class PipelineOptimizer:
    """Orchestrates the optimization of your RAG pipeline"""
    
    def __init__(self, base_dir: Path):
        self.base_dir = base_dir
        self.data_dir = base_dir / "src" / "data"
        self.raw_data = self.data_dir / "raw" / "discharge.csv"
        self.results_dir = base_dir / "optimization_results"
        self.results_dir.mkdir(exist_ok=True)
        
        # Create timestamp for this run
        self.timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.run_dir = self.results_dir / f"run_{self.timestamp}"
        self.run_dir.mkdir(exist_ok=True)
        
    def run_optimization_pipeline(self):
        """Run complete optimization pipeline"""
        
        print("\n" + "="*70)
        print("RAG PIPELINE OPTIMIZATION FRAMEWORK")
        print("="*70)
        print(f"Timestamp: {self.timestamp}")
        print(f"Results directory: {self.run_dir}")
        
        # Step 1: Test different chunking strategies
        print("\n" + "="*70)
        print("STEP 1: CHUNKING STRATEGY COMPARISON")
        print("="*70)
        
        best_strategy = self.test_chunking_strategies()
        
        # Step 2: Build index with best strategy
        print("\n" + "="*70)
        print("STEP 2: BUILDING OPTIMIZED INDEX")
        print("="*70)
        
        index_dir = self.build_optimized_index(best_strategy)
        
        # Step 3: Run comprehensive evaluation
        print("\n" + "="*70)
        print("STEP 3: COMPREHENSIVE EVALUATION")
        print("="*70)
        
        evaluation_results = self.run_evaluation(index_dir, best_strategy)
        
        # Step 4: Generate report
        print("\n" + "="*70)
        print("STEP 4: GENERATING OPTIMIZATION REPORT")
        print("="*70)
        
        self.generate_report(best_strategy, evaluation_results)
        
        print(f"\n✅ Optimization complete! Check {self.run_dir} for results.")
    
    def test_chunking_strategies(self) -> str:
        """Test different chunking strategies and find the best"""
        
        strategies = [
            ChunkConfig(
                strategy=ChunkingStrategy.SLIDING_WINDOW,
                chunk_size=200,
                overlap=50
            ),
            ChunkConfig(
                strategy=ChunkingStrategy.SLIDING_WINDOW,
                chunk_size=200,
                overlap=100
            ),
            ChunkConfig(
                strategy=ChunkingStrategy.SLIDING_WINDOW,
                chunk_size=150,
                overlap=75
            ),
            ChunkConfig(
                strategy=ChunkingStrategy.HYBRID,
                chunk_size=250,
                overlap=50
            ),
            ChunkConfig(
                strategy=ChunkingStrategy.SEMANTIC_SECTIONS,
                chunk_size=300,
                overlap=0
            ),
        ]
        
        # Process with each strategy
        strategy_dir = self.run_dir / "chunking_strategies"
        results = process_with_multiple_strategies(
            self.raw_data,
            strategy_dir,
            strategies
        )
        
        # Analyze results
        analysis_df = analyze_chunking_strategies(results)
        analysis_df.to_csv(self.run_dir / "chunking_analysis.csv", index=False)
        
        print("\nChunking Strategy Analysis:")
        print(analysis_df.to_string())
        
        # Select best strategy based on balanced metrics
        # Prefer strategies with ~200 word average and good chunk count
        scores = {}
        for idx, row in analysis_df.iterrows():
            # Score based on ideal characteristics
            avg_words_score = 100 - abs(row['avg_words'] - 200)  # Closer to 200 is better
            consistency_score = 100 - row['std_words']  # Lower std is better
            
            scores[row['strategy']] = (avg_words_score + consistency_score) / 2
        
        best_strategy = max(scores, key=scores.get)
        print(f"\n✓ Selected best strategy: {best_strategy}")
        print(f"  Score: {scores[best_strategy]:.1f}")
        
        return best_strategy
    
    def build_optimized_index(self, strategy_name: str) -> Path:
        """Build index with the selected strategy"""
        
        # Load the chunked data
        chunk_file = self.run_dir / "chunking_strategies" / f"chunks_{strategy_name}.json"
        
        # Create index directory
        index_dir = self.run_dir / "optimized_index"
        index_dir.mkdir(exist_ok=True)
        
        print(f"Building index from {chunk_file.name}...")
        
        # Build index using your existing builder
        # Note: We'll create a wrapper to use your existing code
        build_index(
            json_paths=[chunk_file],
            output_dir=index_dir,
            model_name="BAAI/bge-base-en-v1.5"
        )
        
        return index_dir
    
    def run_evaluation(self, index_dir: Path, strategy_name: str) -> Dict:
        """Run comprehensive evaluation on the optimized index"""
        
        # Load test data
        chunk_file = self.run_dir / "chunking_strategies" / f"chunks_{strategy_name}.json"
        with open(chunk_file, 'r') as f:
            test_data = json.load(f)
        
        print(f"Running evaluation on {len(test_data)} chunks...")
        
        # Initialize retriever with new index
        retriever = Retriever(index_dir)
        
        # Run comprehensive evaluation
        evaluator = ComprehensiveEvaluator(retriever, test_data)
        results = evaluator.run_comprehensive_evaluation(n_samples_per_scenario=20)
        
        # Save results
        results_file = self.run_dir / "evaluation_results.json"
        with open(results_file, 'w') as f:
            json.dump(results, f, indent=2, default=str)
        
        return results
    
    def generate_report(self, best_strategy: str, eval_results: Dict):
        """Generate comprehensive optimization report"""
        
        report_content = f"""
# RAG Pipeline Optimization Report
**Generated:** {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}
**Run ID:** {self.timestamp}

## Executive Summary

The optimization process tested multiple chunking strategies and evaluation scenarios to improve retrieval performance.

**Selected Strategy:** {best_strategy}

## Evaluation Results

### Scenario Performance
"""
        
        # Add scenario results
        total_score = 0
        scenario_count = 0
        
        for scenario, data in eval_results.items():
            if data.get('valid_queries', 0) > 0:
                success_rate = data.get('success_rate', 0)
                report_content += f"\n**{scenario.replace('_', ' ').title()}**"
                report_content += f"\n- Success Rate: {success_rate:.1f}%"
                report_content += f"\n- Valid Queries: {data['valid_queries']}"
                report_content += f"\n- Avg Latency: {data.get('avg_latency', 0):.3f}s\n"
                
                total_score += success_rate
                scenario_count += 1
        
        if scenario_count > 0:
            avg_score = total_score / scenario_count
            report_content += f"\n### Overall Performance Score: {avg_score:.1f}%\n"
            
            # Add recommendations based on score
            report_content += "\n## Recommendations\n"
            
            if avg_score >= 80:
                report_content += """
**System is Production Ready**
- Performance exceeds target thresholds
- Consider deploying to staging environment
- Implement monitoring for production metrics
"""
            elif avg_score >= 70:
                report_content += """
**System is Functional but Can Be Improved**
- Current performance is acceptable for testing
- Consider implementing hybrid search
- Add query expansion for better recall
"""
            else:
                report_content += """
**System Needs Further Optimization**
- Review chunking parameters
- Consider domain-specific embeddings
- Implement comprehensive preprocessing
"""
        
        report_content += f"""
## Technical Details

### Chunking Configuration
- Strategy: {best_strategy}
- See `chunking_analysis.csv` for detailed comparison

### Index Statistics  
- See `optimized_index/` directory for index files

### Evaluation Details
- See `evaluation_results.json` for raw metrics

## Next Steps

1. **Immediate Actions**
   - Deploy optimized configuration
   - Monitor real-world performance
   - Collect user feedback

2. **Future Improvements**
   - Fine-tune embedding model on medical data
   - Implement hybrid search (BM25 + Vector)
   - Add cross-encoder reranking

---
*Report generated by InsightAI Pipeline Optimizer*
"""
        
        # Save report
        report_file = self.run_dir / "optimization_report.md"
        with open(report_file, 'w') as f:
            f.write(report_content)
        
        print(f"\nReport saved to {report_file}")

def main():
    """Main entry point for optimization"""
    
    # Get base directory
    base_dir = Path.cwd()
    
    print("="*70)
    print("INSIGHTAI RAG PIPELINE OPTIMIZER")
    print("="*70)
    print(f"Base directory: {base_dir}")
    print("\nThis process will:")
    print("1. Test multiple chunking strategies")
    print("2. Build an optimized index")
    print("3. Run comprehensive evaluation")
    print("4. Generate optimization report")
    
    response = input("\nProceed with optimization? (y/n): ")
    
    if response.lower() != 'y':
        print("Optimization cancelled.")
        return
    
    # Run optimization
    optimizer = PipelineOptimizer(base_dir)
    
    try:
        optimizer.run_optimization_pipeline()
    except Exception as e:
        print(f"\nError during optimization: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0

if __name__ == "__main__":
    sys.exit(main())