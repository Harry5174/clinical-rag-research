"""
Generates synthetic hard negative scenarios where semantic scores are identical
but temporal context differs. This isolates TIMER's temporal modulation logic.
"""

import json
from datetime import datetime, timedelta
from typing import List, Dict
import random
from pathlib import Path

class HardNegativeGenerator:
    """
    Creates scenarios that defeat semantic-only retrieval.
    """
    
    def __init__(self, reference_date: str = "2024-01-15"):
        self.reference_date = datetime.strptime(reference_date, "%Y-%m-%d")
        
    def generate_semantic_collision(self) -> List[Dict]:
        """
        Scenario 1: Identical clinical measurements at different times.
        Example: "Blood pressure 120/80" appears in 2020 and 2024.
        """
        measurements = [
            ("Blood pressure", "120/80"),
            ("Heart rate", "72 bpm"),
            ("Temperature", "98.6°F"),
            ("Oxygen saturation", "98%"),
            ("Weight", "70 kg"),
            ("Glucose", "95 mg/dL"),
            ("Creatinine", "1.0 mg/dL"),
            ("Hemoglobin", "14.2 g/dL")
        ]
        
        examples = []
        for measurement, value in measurements[:5]:  # Generate 5 examples
            
            # Create identical text
            clinical_text = f"{measurement}: {value}"
            
            # Old note (3-5 years ago)
            old_days = random.randint(1095, 1825)  # 3-5 years
            old_date = self.reference_date - timedelta(days=old_days)
            
            # New note (0-7 days ago)
            new_days = random.randint(0, 7)
            new_date = self.reference_date - timedelta(days=new_days)
            
            # Note IDs
            note_old_id = f"note_old_{len(examples)+1}"
            note_new_id = f"note_new_{len(examples)+1}"

            # Historical query (should retrieve OLD note)
            historical_query = {
                "id": f"sc_hist_{len(examples)+1}",
                "text": f"What was the patient's {measurement.lower()} during their first admission?",
                "intent": "historical",
                "intent_confidence": 0.95,
                "notes": [
                    {
                        "id": note_new_id,
                        "text": f"{clinical_text} (recorded {new_date.strftime('%Y-%m-%d')})",
                        "offset_days": new_days,
                        "note_date": new_date.strftime("%Y-%m-%d"),
                        "section": "Vital Signs",
                        "semantic_score": 0.95  # IDENTICAL
                    },
                    {
                        "id": note_old_id,
                        "text": f"{clinical_text} (recorded {old_date.strftime('%Y-%m-%d')})",
                        "offset_days": old_days,
                        "note_date": old_date.strftime("%Y-%m-%d"),
                        "section": "Vital Signs",
                        "semantic_score": 0.95  # IDENTICAL
                    }
                ],
                "expected_retrieval": note_old_id,
                "expected_rank": 1,
                "failure_mode": "semantic_collision"
            }
            
            # Current query (should retrieve NEW note)
            current_query = {
                "id": f"sc_curr_{len(examples)+1}",
                "text": f"What is the patient's current {measurement.lower()}?",
                "intent": "current",
                "intent_confidence": 0.95,
                "notes": [
                    {
                        "id": note_new_id,
                        "text": f"{clinical_text} (recorded {new_date.strftime('%Y-%m-%d')})",
                        "offset_days": new_days,
                        "note_date": new_date.strftime("%Y-%m-%d"),
                        "section": "Vital Signs",
                        "semantic_score": 0.95  # IDENTICAL
                    },
                    {
                        "id": note_old_id,
                        "text": f"{clinical_text} (recorded {old_date.strftime('%Y-%m-%d')})",
                        "offset_days": old_days,
                        "note_date": old_date.strftime("%Y-%m-%d"),
                        "section": "Vital Signs",
                        "semantic_score": 0.95  # IDENTICAL
                    }
                ],  # Same notes
                "expected_retrieval": note_new_id,
                "expected_rank": 1,
                "failure_mode": "semantic_collision"
            }
            
            examples.extend([historical_query, current_query])
        
        return examples
    
    def generate_negation_recency(self) -> List[Dict]:
        """
        Scenario 2: Recent note contains negation of historical condition.
        Example: "No longer has diabetes" (2024) vs "Diagnosed with Type 2 DM" (2015)
        """
        conditions = [
            ("diabetes", "Type 2 Diabetes Mellitus", "No current diabetes medications, condition resolved"),
            ("hypertension", "Essential Hypertension", "Blood pressure normalized, no longer hypertensive"),
            ("asthma", "Chronic Asthma", "No asthma symptoms in past 2 years, off medications"),
            ("GERD", "Gastroesophageal Reflux Disease", "GERD resolved with lifestyle modifications"),
            ("hypothyroidism", "Primary Hypothyroidism", "Thyroid function normalized, off levothyroxine")
        ]
        
        examples = []
        for condition, diagnosis_text, resolution_text in conditions:
            
            # Old diagnostic note (5-10 years ago)
            old_days = random.randint(1825, 3650)
            old_date = self.reference_date - timedelta(days=old_days)
            
            # Recent resolution note (0-30 days)
            new_days = random.randint(0, 30)
            new_date = self.reference_date - timedelta(days=new_days)
            
            # Note IDs
            note_res_id = f"note_resolution_{len(examples)+1}"
            note_diag_id = f"note_diagnosis_{len(examples)+1}"

            # Historical query - Should find DIAGNOSIS, not resolution
            query = {
                "id": f"nr_hist_{len(examples)+1}",
                "text": f"Has the patient ever been diagnosed with {condition}?",
                "intent": "historical",
                "intent_confidence": 0.92,
                "notes": [
                    {
                        "id": note_res_id,
                        "text": f"Assessment: {resolution_text}. Patient reports significant improvement.",
                        "offset_days": new_days,
                        "note_date": new_date.strftime("%Y-%m-%d"),
                        "section": "Assessment",
                        "semantic_score": 0.88  # High (contains condition name)
                    },
                    {
                        "id": note_diag_id,
                        "text": f"Diagnosis: {diagnosis_text}. Initiated treatment plan.",
                        "offset_days": old_days,
                        "note_date": old_date.strftime("%Y-%m-%d"),
                        "section": "Past Medical History",
                        "semantic_score": 0.92  # Slightly higher (more direct match)
                    }
                ],
                "expected_retrieval": note_diag_id,
                "expected_rank": 1,
                "failure_mode": "negation_recency",
                "risk": "Baseline may prioritize recent 'resolution' note and miss historical diagnosis"
            }
            
            examples.append(query)
        
        return examples
    
    def generate_terminology_drift(self) -> List[Dict]:
        """
        Scenario 3: Medical terminology has evolved over time.
        Example: "IDDM" (1990s) vs "Type 1 Diabetes" (modern)
        """
        terminology_pairs = [
            ("Type 1 Diabetes", "IDDM", "Insulin-Dependent Diabetes Mellitus"),
            ("Myocardial Infarction", "MI", "Heart attack"),
            ("Cerebrovascular Accident", "CVA", "Stroke"),
            ("Congestive Heart Failure", "CHF", "Heart failure with reduced ejection fraction"),
            ("Chronic Obstructive Pulmonary Disease", "COPD", "Chronic bronchitis and emphysema")
        ]
        
        examples = []
        for modern_term, legacy_abbrev, legacy_full in terminology_pairs:
            
            # Old note with legacy terminology (10-20 years)
            old_days = random.randint(3650, 7300)
            old_date = self.reference_date - timedelta(days=old_days)
            
            # New note with modern terminology (0-365 days)
            new_days = random.randint(0, 365)
            new_date = self.reference_date - timedelta(days=new_days)
            
            # Note IDs
            note_mod_id = f"note_modern_{len(examples)+1}"
            note_leg_id = f"note_legacy_{len(examples)+1}"

            # Query using modern terminology
            query = {
                "id": f"td_hist_{len(examples)+1}",
                "text": f"When was the patient first diagnosed with {modern_term.lower()}?",
                "intent": "historical",
                "intent_confidence": 0.88,
                "notes": [
                    {
                        "id": note_mod_id,
                        "text": f"Current status: {modern_term} managed with medications.",
                        "offset_days": new_days,
                        "note_date": new_date.strftime("%Y-%m-%d"),
                        "section": "Assessment",
                        "semantic_score": 0.93  # High (exact terminology match)
                    },
                    {
                        "id": note_leg_id,
                        "text": f"New diagnosis: {legacy_full} ({legacy_abbrev}). Patient presented with typical symptoms.",
                        "offset_days": old_days,
                        "note_date": old_date.strftime("%Y-%m-%d"),
                        "section": "History of Present Illness",
                        "semantic_score": 0.78  # Lower (terminology mismatch)
                    }
                ],
                "expected_retrieval": note_leg_id,
                "expected_rank": 1,
                "failure_mode": "terminology_drift",
                "risk": "Semantic embeddings may not capture equivalence between legacy and modern terms"
            }
            
            examples.append(query)
        
        return examples
    
    def generate_dataset(self, output_path: str = "data/mocks/hard_negatives.json"):
        """
        Generates complete hard negative dataset.
        """
        dataset = {
            "metadata": {
                "version": "1.0",
                "created": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                "reference_date": self.reference_date.strftime("%Y-%m-%d"),
                "description": "Hard negative scenarios for TIMER-Graph validation",
                "total_queries": 0
            },
            "scenarios": {
                "semantic_collision": self.generate_semantic_collision(),
                "negation_recency": self.generate_negation_recency(),
                "terminology_drift": self.generate_terminology_drift()
            }
        }
        
        # Calculate total
        total = sum(len(queries) for queries in dataset["scenarios"].values())
        dataset["metadata"]["total_queries"] = total
        
        # Ensure output directory exists
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        
        # Save
        with open(output_path, 'w') as f:
            json.dump(dataset, f, indent=2)
        
        print(f"Generated {total} hard negative queries")
        print(f"   - Semantic Collision: {len(dataset['scenarios']['semantic_collision'])}")
        print(f"   - Negation Recency: {len(dataset['scenarios']['negation_recency'])}")
        print(f"   - Terminology Drift: {len(dataset['scenarios']['terminology_drift'])}")
        print(f"Saved to: {output_path}")
        
        return dataset


if __name__ == "__main__":
    generator = HardNegativeGenerator(reference_date="2024-01-15")
    dataset = generator.generate_dataset()
