"""
Mines real-world hard negatives from discharge.csv by finding:
1. Repeated phrases across different time periods
2. Temporal separation > 365 days
3. Clinical significance (vitals, diagnoses, medications)
"""

import pandas as pd
import re
from collections import defaultdict
from datetime import datetime
from typing import List, Dict, Tuple
import json
from pathlib import Path

class HardNegativeMiner:
    """
    Extracts hard negative scenarios from real clinical data.
    """
    
    def __init__(self, data_path: str = "data/processed/discharge.json"):
        self.data_path = Path(data_path)
        if self.data_path.exists():
            if self.data_path.suffix == '.json':
                with open(self.data_path, 'r') as f:
                    self.data = json.load(f)
                self.df = pd.DataFrame(self.data)
            else:
                self.df = pd.read_csv(data_path)
        else:
            self.df = pd.DataFrame() # Empty DataFrame if file missing
        
        # Clinical patterns to search for
        self.patterns = {
            "vitals": [
                r"blood pressure:?\s*(\d+/\d+)",
                r"heart rate:?\s*(\d+)\s*bpm",
                r"temperature:?\s*([\d.]+)\s*[°f]",
                r"weight:?\s*([\d.]+)\s*kg",
                r"BMI:?\s*([\d.]+)"
            ],
            "labs": [
                r"creatinine:?\s*([\d.]+)",
                r"hemoglobin:?\s*([\d.]+)",
                r"glucose:?\s*(\d+)",
                r"sodium:?\s*(\d+)",
                r"potassium:?\s*([\d.]+)"
            ],
            "diagnoses": [
                r"diagnosed with\s+([^.,:]+)",
                r"history of\s+([^.,:]+)",
                r"admission for\s+([^.,:]+)"
            ]
        }
    
    def extract_phrases(self, text: str, category: str) -> List[Tuple[str, str]]:
        """
        Extracts clinical phrases using regex patterns.
        Returns: [(phrase, value), ...]
        """
        matches = []
        for pattern in self.patterns.get(category, []):
            try:
                for match in re.finditer(pattern, text.lower()):
                    phrase = match.group(0)
                    value = match.group(1) if match.groups() else phrase
                    matches.append((phrase, value))
            except Exception:
                continue
        return matches
    
    def find_temporal_duplicates(
        self, 
        min_separation_days: int = 365,
        max_matches: int = 10
    ) -> List[Dict]:
        """
        Finds notes with repeated phrases separated by >min_separation_days.
        """
        if self.df.empty:
            return []

        # Group by patient
        phrase_index = defaultdict(list)
        
        from tqdm import tqdm
        print(f"Dataframe shape: {self.df.shape}")
        
        for idx, row in tqdm(self.df.iterrows(), total=len(self.df), desc="Mining"):
            # Handle JSON keys vs CSV keys
            text = str(row.get('chunk_text', row.get('text', '')))
            date_str = str(row.get('charttime', row.get('chartdate', '')))
            
            if not text or not date_str or date_str.lower() == 'nan':
                continue
            
            try:
                date = pd.to_datetime(date_str)
            except:
                continue

            patient_id = row.get('subject_id', 'unknown')
            note_id = row.get('note_id', row.get('row_id', f"note_{idx}"))
            
            # Extract all clinical phrases
            for category in self.patterns.keys():
                phrases = self.extract_phrases(text, category)
                for phrase, value in phrases:
                    key = f"{patient_id}_{category}_{value}"
                    phrase_index[key].append({
                        "note_id": note_id,
                        "text": text,
                        "date": date,
                        "phrase": phrase,
                        "value": value,
                        "category": category
                    })
        
        # Find duplicates with temporal separation
        hard_negatives = []
        
        for key, occurrences in phrase_index.items():
            if len(occurrences) < 2:
                continue
            
            # Sort by date
            occurrences.sort(key=lambda x: x['date'])
            
            # Check temporal separation
            oldest = occurrences[0]
            newest = occurrences[-1]
            
            days_diff = (newest['date'] - oldest['date']).days
            
            if days_diff >= min_separation_days:
                hard_negatives.append({
                    "id": f"mined_{len(hard_negatives)+1}",
                    "category": oldest['category'],
                    "phrase": oldest['phrase'],
                    "value": oldest['value'],
                    "temporal_separation_days": days_diff,
                    "old_note": {
                        "id": oldest['note_id'],
                        "text": oldest['text'][:200],  # Truncate for readability
                        "date": oldest['date'].strftime("%Y-%m-%d"),
                        "offset_days": days_diff
                    },
                    "new_note": {
                        "id": newest['note_id'],
                        "text": newest['text'][:200],
                        "date": newest['date'].strftime("%Y-%m-%d"),
                        "offset_days": 0
                    },
                    "suggested_queries": {
                        "historical": f"What was the patient's {oldest['category']} during their first admission?",
                        "current": f"What is the patient's current {oldest['category']}?"
                    }
                })
                
                if len(hard_negatives) >= max_matches:
                    break
        
        return hard_negatives
    
    def export_for_annotation(self, output_path: str = "data/mocks/mined_candidates.json"):
        """
        Exports mined candidates for manual review and annotation.
        """
        candidates = self.find_temporal_duplicates()
        
        annotation_template = {
            "metadata": {
                "created": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                "total_candidates": len(candidates),
                "status": "awaiting_annotation",
                "instructions": "Review each candidate and add: intent, expected_retrieval, semantic_score estimates"
            },
            "candidates": candidates
        }
        
        # Ensure directory exists
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_path, 'w') as f:
            json.dump(annotation_template, f, indent=2)
        
        print(f"✅ Mined {len(candidates)} hard negative candidates")
        print(f"📁 Saved to: {output_path}")
        print("\n📝 Next step: Manually annotate with ground truth intents and semantic scores")
        
        return candidates


if __name__ == "__main__":
    # Check if discharge.csv exists
    miner = HardNegativeMiner(data_path="data/raw/discharge.csv")
    if not miner.df.empty:
        candidates = miner.export_for_annotation()
        # Show sample
        if candidates:
            print("\n📊 Sample Candidate:")
            print(json.dumps(candidates[0], indent=2))
    else:
        print("⚠️  discharge.csv not found or empty. Using synthetic data only for Phase 5.")
        print("    To mine real data: Place discharge.csv in data/raw/ and re-run.")
