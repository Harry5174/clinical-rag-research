"""
Advanced Chunking Strategy for Medical Discharge Notes
Implements multiple chunking approaches with semantic awareness
"""

import re
import pandas as pd
import json
from pathlib import Path
from typing import List, Dict, Iterator, Tuple
import hashlib
from dataclasses import dataclass
from enum import Enum

class ChunkingStrategy(Enum):
    """Different chunking strategies for evaluation"""
    FIXED_SIZE = "fixed_size"
    SLIDING_WINDOW = "sliding_window"
    SEMANTIC_SECTIONS = "semantic_sections"
    SENTENCE_BASED = "sentence_based"
    HYBRID = "hybrid"
    HEADER_PROPAGATION = "header_propagation"

@dataclass
class ChunkConfig:
    """Configuration for different chunking strategies"""
    strategy: ChunkingStrategy
    chunk_size: int = 200  # words
    overlap: int = 50  # words
    min_chunk_size: int = 50  # minimum words
    max_chunk_size: int = 400  # maximum words
    sentence_window: int = 5  # sentences per chunk
    
class MedicalSectionDetector:
    """Detect and parse medical document sections"""
    
    SECTION_PATTERNS = {
        'chief_complaint': r'(?i)chief\s+complaint[:\s]+(.*?)(?=\n\n|\n[A-Z][A-Z\s]+:)',
        'hpi': r'(?i)history\s+of\s+present\s+illness[:\s]+(.*?)(?=\n\n|\n[A-Z][A-Z\s]+:)',
        'past_medical': r'(?i)past\s+medical\s+history[:\s]+(.*?)(?=\n\n|\n[A-Z][A-Z\s]+:)',
        'medications': r'(?i)(?:discharge\s+)?medications?[:\s]+(.*?)(?=\n\n|\n[A-Z][A-Z\s]+:)',
        'allergies': r'(?i)allergies[:\s]+(.*?)(?=\n\n|\n[A-Z][A-Z\s]+:)',
        'social_history': r'(?i)social\s+history[:\s]+(.*?)(?=\n\n|\n[A-Z][A-Z\s]+:)',
        'family_history': r'(?i)family\s+history[:\s]+(.*?)(?=\n\n|\n[A-Z][A-Z\s]+:)',
        'physical_exam': r'(?i)physical\s+exam(?:ination)?[:\s]+(.*?)(?=\n\n|\n[A-Z][A-Z\s]+:)',
        'lab_results': r'(?i)(?:pertinent\s+)?(?:results?|labs?)[:\s]+(.*?)(?=\n\n|\n[A-Z][A-Z\s]+:)',
        'hospital_course': r'(?i)(?:brief\s+)?hospital\s+course[:\s]+(.*?)(?=\n\n|\n[A-Z][A-Z\s]+:)',
        'discharge_diagnosis': r'(?i)discharge\s+diagnos[ie]s[:\s]+(.*?)(?=\n\n|\n[A-Z][A-Z\s]+:)',
        'discharge_instructions': r'(?i)discharge\s+instructions?[:\s]+(.*?)(?=\n\n|\n[A-Z][A-Z\s]+:)',
    }
    
    @classmethod
    def detect_sections(cls, text: str) -> Dict[str, str]:
        """Extract sections from medical text"""
        sections = {}
        
        for section_name, pattern in cls.SECTION_PATTERNS.items():
            match = re.search(pattern, text, re.DOTALL)
            if match:
                content = match.group(1).strip()
                if content and len(content) > 20:  # Minimum content threshold
                    sections[section_name] = content
        
        # Handle unstructured text
        if not sections:
            sections['unstructured'] = text
            
        return sections

class AdvancedChunker:
    """Advanced chunking with multiple strategies"""
    
    def __init__(self, config: ChunkConfig):
        self.config = config
        self.section_detector = MedicalSectionDetector()
    
    def chunk_text(self, text: str, metadata: Dict = None) -> List[Dict]:
        """Apply chunking strategy based on configuration"""
        
        if self.config.strategy == ChunkingStrategy.FIXED_SIZE:
            return self._fixed_size_chunks(text, metadata)
        elif self.config.strategy == ChunkingStrategy.SLIDING_WINDOW:
            return self._sliding_window_chunks(text, metadata)
        elif self.config.strategy == ChunkingStrategy.SEMANTIC_SECTIONS:
            return self._semantic_section_chunks(text, metadata)
        elif self.config.strategy == ChunkingStrategy.SENTENCE_BASED:
            return self._sentence_based_chunks(text, metadata)
        elif self.config.strategy == ChunkingStrategy.HYBRID:
            return self._hybrid_chunks(text, metadata)
        elif self.config.strategy == ChunkingStrategy.HEADER_PROPAGATION:
            return self._header_propagation_chunks(text, metadata)
        else:
            raise ValueError(f"Unknown strategy: {self.config.strategy}")
    
    def _fixed_size_chunks(self, text: str, metadata: Dict) -> List[Dict]:
        """Traditional fixed-size chunking"""
        chunks = []
        words = text.split()
        
        for i in range(0, len(words), self.config.chunk_size):
            chunk_words = words[i:i + self.config.chunk_size]
            if len(chunk_words) >= self.config.min_chunk_size:
                chunk_text = ' '.join(chunk_words)
                chunk_data = self._create_chunk_metadata(
                    chunk_text, i // self.config.chunk_size, metadata
                )
                chunks.append(chunk_data)
        
        return chunks
    
    def _sliding_window_chunks(self, text: str, metadata: Dict) -> List[Dict]:
        """Overlapping sliding window chunks"""
        chunks = []
        words = text.split()
        
        if not words:
            return chunks
        
        step = max(1, self.config.chunk_size - self.config.overlap)
        
        for i in range(0, len(words), step):
            end = min(i + self.config.chunk_size, len(words))
            chunk_words = words[i:end]
            
            if len(chunk_words) >= self.config.min_chunk_size:
                chunk_text = ' '.join(chunk_words)
                chunk_data = self._create_chunk_metadata(
                    chunk_text, len(chunks), metadata
                )
                chunks.append(chunk_data)
            
            # Stop if we've reached the end
            if end >= len(words):
                break
        
        return chunks
    
    def _semantic_section_chunks(self, text: str, metadata: Dict) -> List[Dict]:
        """Chunk based on semantic sections"""
        chunks = []
        sections = self.section_detector.detect_sections(text)
        
        chunk_idx = 0
        for section_name, section_content in sections.items():
            # If section is too large, subdivide it
            if len(section_content.split()) > self.config.max_chunk_size:
                sub_chunks = self._sliding_window_chunks(section_content, metadata)
                for sub_chunk in sub_chunks:
                    sub_chunk['section'] = section_name
                    sub_chunk['chunk_index'] = chunk_idx
                    chunks.append(sub_chunk)
                    chunk_idx += 1
            else:
                chunk_data = self._create_chunk_metadata(
                    section_content, chunk_idx, metadata
                )
                chunk_data['section'] = section_name
                chunks.append(chunk_data)
                chunk_idx += 1
        
        return chunks
    
    def _sentence_based_chunks(self, text: str, metadata: Dict) -> List[Dict]:
        """Chunk based on sentence boundaries"""
        chunks = []
        
        # Improved sentence splitting for medical text
        sentences = self._split_sentences(text)
        
        for i in range(0, len(sentences), self.config.sentence_window):
            chunk_sentences = sentences[i:i + self.config.sentence_window]
            chunk_text = ' '.join(chunk_sentences)
            
            # Check word count
            word_count = len(chunk_text.split())
            if word_count >= self.config.min_chunk_size:
                chunk_data = self._create_chunk_metadata(
                    chunk_text, i // self.config.sentence_window, metadata
                )
                chunks.append(chunk_data)
        
        return chunks
    
    def _hybrid_chunks(self, text: str, metadata: Dict) -> List[Dict]:
        """Hybrid approach: semantic sections with smart boundaries"""
        chunks = []
        sections = self.section_detector.detect_sections(text)
        
        chunk_idx = 0
        for section_name, section_content in sections.items():
            sentences = self._split_sentences(section_content)
            current_chunk = []
            current_word_count = 0
            
            for sentence in sentences:
                sentence_words = len(sentence.split())
                
                # Check if adding this sentence would exceed max size
                if current_word_count + sentence_words > self.config.chunk_size and current_chunk:
                    # Save current chunk
                    chunk_text = ' '.join(current_chunk)
                    chunk_data = self._create_chunk_metadata(
                        chunk_text, chunk_idx, metadata
                    )
                    chunk_data['section'] = section_name
                    chunks.append(chunk_data)
                    chunk_idx += 1
                    
                    # Start new chunk with overlap
                    overlap_sentences = current_chunk[-2:] if len(current_chunk) > 2 else current_chunk
                    current_chunk = overlap_sentences + [sentence]
                    current_word_count = sum(len(s.split()) for s in current_chunk)
                else:
                    current_chunk.append(sentence)
                    current_word_count += sentence_words
            
            # Save remaining chunk
            if current_chunk and current_word_count >= self.config.min_chunk_size:
                chunk_text = ' '.join(current_chunk)
                chunk_data = self._create_chunk_metadata(
                    chunk_text, chunk_idx, metadata
                )
                chunk_data['section'] = section_name
                chunks.append(chunk_data)
                chunk_idx += 1
        
        return chunks

    def _header_propagation_chunks(self, text: str, metadata: Dict) -> List[Dict]:
        """Header Propagation: Inject section headers into every sub-chunk"""
        chunks = []
        sections = self.section_detector.detect_sections(text)
        
        chunk_idx = 0
        for section_name, section_content in sections.items():
            # Use hybrid splitting for the content
            sentences = self._split_sentences(section_content)
            current_chunk = []
            current_word_count = 0
            
            # Formatted header to inject
            header_prefix = f"[{section_name.replace('_', ' ').title()}]\n"
            header_words = len(header_prefix.split())
            
            for sentence in sentences:
                sentence_words = len(sentence.split())
                
                # Check if adding this sentence would exceed max size (accounting for header)
                if current_word_count + sentence_words + header_words > self.config.chunk_size and current_chunk:
                    # Save current chunk with HEADER INJECTED
                    chunk_text = header_prefix + ' '.join(current_chunk)
                    chunk_data = self._create_chunk_metadata(
                        chunk_text, chunk_idx, metadata
                    )
                    chunk_data['section'] = section_name
                    chunk_data['is_header_propagated'] = True
                    chunks.append(chunk_data)
                    chunk_idx += 1
                    
                    # Start new chunk with overlap
                    overlap_sentences = current_chunk[-2:] if len(current_chunk) > 2 else current_chunk
                    current_chunk = overlap_sentences + [sentence]
                    current_word_count = sum(len(s.split()) for s in current_chunk)
                else:
                    current_chunk.append(sentence)
                    current_word_count += sentence_words
            
            # Save remaining chunk
            if current_chunk and current_word_count >= self.config.min_chunk_size:
                chunk_text = header_prefix + ' '.join(current_chunk)
                chunk_data = self._create_chunk_metadata(
                    chunk_text, chunk_idx, metadata
                )
                chunk_data['section'] = section_name
                chunk_data['is_header_propagated'] = True
                chunks.append(chunk_data)
                chunk_idx += 1
        
        return chunks
    
    def _split_sentences(self, text: str) -> List[str]:
        """Medical-aware sentence splitting"""
        # Handle common medical abbreviations that shouldn't end sentences
        text = re.sub(r'\b(Dr|Mr|Mrs|Ms|Prof|Sr|Jr)\.\s*', r'\1<DOT> ', text)
        text = re.sub(r'\b(vs|eg|ie|etc|al|et al)\.\s*', r'\1<DOT> ', text)
        text = re.sub(r'\b(mg|mcg|mL|kg|lb|oz)\.\s*', r'\1<DOT> ', text)
        
        # Split on sentence boundaries
        sentences = re.split(r'(?<=[.!?])\s+(?=[A-Z])', text)
        
        # Restore dots
        sentences = [s.replace('<DOT>', '.') for s in sentences]
        
        # Filter out very short sentences
        sentences = [s.strip() for s in sentences if len(s.strip()) > 10]
        
        return sentences
    
    def _create_chunk_metadata(self, chunk_text: str, chunk_index: int, 
                               base_metadata: Dict = None) -> Dict:
        """Create comprehensive chunk metadata"""
        chunk_id = hashlib.md5(f"{chunk_text[:50]}_{chunk_index}".encode()).hexdigest()[:8]
        
        metadata = {
            'chunk_id': chunk_id,
            'chunk_index': chunk_index,
            'chunk_text': chunk_text,
            'word_count': len(chunk_text.split()),
            'char_count': len(chunk_text),
            'strategy': self.config.strategy.value
        }
        
        if base_metadata:
            metadata.update(base_metadata)
        
        return metadata

def process_with_multiple_strategies(
    input_csv: Path,
    output_dir: Path,
    strategies: List[ChunkConfig]
) -> Dict[str, List[Dict]]:
    """Process data with multiple chunking strategies for comparison"""
    
    # Load data
    df = pd.read_csv(input_csv, nrows=1000)  # Limit for testing
    print(f"Loaded {len(df)} rows from {input_csv}")
    
    # Filter for cardiac content
    cardiac_keywords = ["cardiac", "heart", "coronary", "myocardial", "artery"]
    mask = df['text'].str.contains('|'.join(cardiac_keywords), case=False, na=False)
    df_filtered = df[mask].copy()
    print(f"Filtered to {len(df_filtered)} cardiac-related records")
    
    results = {}
    
    for config in strategies:
        print(f"\nProcessing with strategy: {config.strategy.value}")
        chunker = AdvancedChunker(config)
        
        all_chunks = []
        for idx, row in df_filtered.iterrows():
            if pd.isna(row['text']):
                continue
                
            # Prepare metadata
            metadata = {
                'note_id': str(row.get('note_id', idx)),
                'subject_id': str(row.get('subject_id', '')),
                'note_type': str(row.get('note_type', 'discharge')),
            }
            
            # Generate chunks
            chunks = chunker.chunk_text(row['text'], metadata)
            all_chunks.extend(chunks)
        
        # Save results
        strategy_name = config.strategy.value
        output_file = output_dir / f"chunks_{strategy_name}.json"
        output_dir.mkdir(parents=True, exist_ok=True)
        
        with open(output_file, 'w') as f:
            json.dump(all_chunks, f, indent=2)
        
        results[strategy_name] = all_chunks
        print(f"  Generated {len(all_chunks)} chunks")
        print(f"  Saved to {output_file}")
    
    return results

def analyze_chunking_strategies(results: Dict[str, List[Dict]]) -> pd.DataFrame:
    """Analyze and compare different chunking strategies"""
    
    analysis = []
    
    for strategy, chunks in results.items():
        word_counts = [c['word_count'] for c in chunks]
        
        analysis.append({
            'strategy': strategy,
            'total_chunks': len(chunks),
            'avg_words': sum(word_counts) / len(word_counts) if word_counts else 0,
            'min_words': min(word_counts) if word_counts else 0,
            'max_words': max(word_counts) if word_counts else 0,
            'std_words': pd.Series(word_counts).std() if word_counts else 0,
        })
    
    return pd.DataFrame(analysis)

if __name__ == "__main__":
    # Define strategies to test
    strategies = [
        ChunkConfig(ChunkingStrategy.FIXED_SIZE, chunk_size=200, overlap=0),
        ChunkConfig(ChunkingStrategy.SLIDING_WINDOW, chunk_size=200, overlap=50),
        ChunkConfig(ChunkingStrategy.SLIDING_WINDOW, chunk_size=200, overlap=100),
        ChunkConfig(ChunkingStrategy.SEMANTIC_SECTIONS),
        ChunkConfig(ChunkingStrategy.SENTENCE_BASED, sentence_window=5),
        ChunkConfig(ChunkingStrategy.HYBRID, chunk_size=250, overlap=50),
    ]
    
    # Process data
    input_csv = Path("src/data/raw/discharge.csv")
    output_dir = Path("src/data/chunking_comparison")
    
    results = process_with_multiple_strategies(input_csv, output_dir, strategies)
    
    # Analyze results
    analysis_df = analyze_chunking_strategies(results)
    print("\n" + "="*60)
    print("CHUNKING STRATEGY COMPARISON")
    print("="*60)
    print(analysis_df.to_string())
    
    # Save analysis
    analysis_df.to_csv(output_dir / "strategy_comparison.csv", index=False)
    print(f"\nAnalysis saved to {output_dir / 'strategy_comparison.csv'}")