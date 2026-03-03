"""
Generates synthetic hard negative scenarios where semantic scores are identical
but temporal context differs. This isolates TIMER's temporal modulation logic.

Usage:
    python generate_hard_negatives.py                          # 100 total (50 SC, 30 NR, 20 TD)
    python generate_hard_negatives.py --sc 50 --nr 30 --td 20
    python generate_hard_negatives.py --output data/mocks/hard_negatives_v2.json
"""

import json
import argparse
from datetime import datetime, timedelta
from typing import List, Dict
import random
from pathlib import Path


class HardNegativeGenerator:
    """
    Creates scenarios that defeat semantic-only retrieval.
    """

    def __init__(self, reference_date: str = "2024-01-15", seed: int = 42):
        self.reference_date = datetime.strptime(reference_date, "%Y-%m-%d")
        random.seed(seed)

    # ---------------------------------------------------------------
    # Seed data (expanded well beyond the old hardcoded 5/5/5 limits)
    # ---------------------------------------------------------------

    # 25 clinical measurements — vitals, labs, assessments
    MEASUREMENTS = [
        # ── Vitals ──────────────────────────────────────────────────
        ("Blood pressure",          "120/80"),
        ("Heart rate",              "72 bpm"),
        ("Temperature",             "98.6°F"),
        ("Oxygen saturation",       "98%"),
        ("Weight",                  "70 kg"),
        ("Respiratory rate",        "16 breaths/min"),
        ("Mean arterial pressure",  "90 mmHg"),
        # ── Labs ─────────────────────────────────────────────────────
        ("Glucose",         "95 mg/dL"),
        ("Creatinine",      "1.0 mg/dL"),
        ("Hemoglobin",      "14.2 g/dL"),
        ("BUN",             "18 mg/dL"),
        ("WBC",             "7.2 K/uL"),
        ("Platelets",       "230 K/uL"),
        ("INR",             "1.1"),
        ("PT",              "12.8 s"),
        ("PTT",             "30 s"),
        ("Troponin",        "0.02 ng/mL"),
        ("BNP",             "80 pg/mL"),
        ("Lactate",         "1.1 mmol/L"),
        ("Sodium",          "140 mEq/L"),
        ("Potassium",       "4.0 mEq/L"),
        # ── Assessments ──────────────────────────────────────────────
        ("LVEF",            "55%"),
        ("GCS",             "15"),
        ("BMI",             "24.5"),
        ("eGFR",            "85 mL/min/1.73m²"),
    ]

    # 15 condition pairs for negation-recency: (condition_slug, diagnosis_text, resolution_text)
    CONDITIONS = [
        ("diabetes",
         "Type 2 Diabetes Mellitus",
         "No current diabetes medications, condition resolved"),
        ("hypertension",
         "Essential Hypertension",
         "Blood pressure normalized, no longer hypertensive"),
        ("asthma",
         "Chronic Asthma",
         "No asthma symptoms in past 2 years, off medications"),
        ("GERD",
         "Gastroesophageal Reflux Disease",
         "GERD resolved with lifestyle modifications"),
        ("hypothyroidism",
         "Primary Hypothyroidism",
         "Thyroid function normalized, off levothyroxine"),
        ("DVT",
         "Deep Vein Thrombosis",
         "DVT resolved; anticoagulation therapy completed"),
        ("pneumonia",
         "Community-Acquired Pneumonia",
         "Pneumonia resolved; chest X-ray clear"),
        ("sepsis",
         "Sepsis secondary to urinary tract infection",
         "Sepsis resolved; source controlled, antibiotics completed"),
        ("metformin use",
         "Started Metformin 500mg for Type 2 Diabetes management",
         "Metformin discontinued; patient transitioned to insulin therapy"),
        ("CABG",
         "CABG performed for three-vessel coronary artery disease",
         "No further surgical intervention planned; medically managed"),
        ("atrial fibrillation",
         "New-onset Atrial Fibrillation",
         "Atrial fibrillation converted to sinus rhythm; no recurrence"),
        ("chronic kidney disease",
         "Chronic Kidney Disease Stage 3",
         "Renal function improved to CKD Stage 2 with conservative management"),
        ("depression",
         "Major Depressive Disorder",
         "Depression in remission; psychiatric medications tapered"),
        ("anemia",
         "Iron Deficiency Anemia",
         "Anemia resolved after iron supplementation; Hgb now normal"),
        ("heart failure",
         "Congestive Heart Failure with reduced EF",
         "Heart failure compensated; LVEF improved to 55%, no fluid overload"),
    ]

    # 10 terminology drift pairs: (modern_term, legacy_abbreviation, legacy_full_name)
    TERMINOLOGY_PAIRS = [
        ("Type 1 Diabetes",
         "IDDM",
         "Insulin-Dependent Diabetes Mellitus"),
        ("Myocardial Infarction",
         "MI",
         "Heart attack"),
        ("Cerebrovascular Accident",
         "CVA",
         "Stroke"),
        ("Congestive Heart Failure",
         "CHF",
         "Heart failure with reduced ejection fraction"),
        ("Chronic Obstructive Pulmonary Disease",
         "COPD",
         "Chronic bronchitis and emphysema"),
        ("Type 2 Diabetes",
         "NIDDM",
         "Non-Insulin-Dependent Diabetes Mellitus"),
        ("Edema",
         "Dropsy",
         "Generalized fluid accumulation and swelling"),
        ("Warfarin",
         "Coumadin",
         "Coumadin anticoagulation therapy initiated"),
        ("Metformin",
         "Glucophage",
         "Glucophage initiated for blood sugar control"),
        ("Type 1 Diabetes",
         "Juvenile diabetes",
         "Juvenile-onset diabetes mellitus requiring insulin"),
    ]

    # ---------------------------------------------------------------
    # Generators
    # ---------------------------------------------------------------

    def generate_semantic_collision(self, n: int = 50) -> List[Dict]:
        """
        Scenario 1: Identical clinical measurements at different times.
        Each measurement seed produces 2 queries (historical + current).
        So we need ceil(n / 2) unique seeds.
        """
        pool = list(self.MEASUREMENTS)
        # Repeat pool if n requested exceeds pool*2
        while len(pool) * 2 < n:
            pool = pool + self.MEASUREMENTS

        examples: List[Dict] = []

        for measurement, value in pool:
            if len(examples) >= n:
                break

            clinical_text = f"{measurement}: {value}"

            old_days = random.randint(1095, 1825)   # 3-5 years
            new_days = random.randint(0, 7)
            old_date = self.reference_date - timedelta(days=old_days)
            new_date = self.reference_date - timedelta(days=new_days)

            note_old_id = f"note_old_{len(examples) // 2 + 1}"
            note_new_id = f"note_new_{len(examples) // 2 + 1}"

            notes = [
                {
                    "id": note_new_id,
                    "text": f"{clinical_text} (recorded {new_date.strftime('%Y-%m-%d')})",
                    "offset_days": new_days,
                    "note_date": new_date.strftime("%Y-%m-%d"),
                    "section": "Vital Signs",
                    "semantic_score": 0.95,
                },
                {
                    "id": note_old_id,
                    "text": f"{clinical_text} (recorded {old_date.strftime('%Y-%m-%d')})",
                    "offset_days": old_days,
                    "note_date": old_date.strftime("%Y-%m-%d"),
                    "section": "Vital Signs",
                    "semantic_score": 0.95,
                },
            ]

            if len(examples) < n:
                examples.append({
                    "id": f"sc_hist_{len(examples) + 1}",
                    "text": f"What was the patient's {measurement.lower()} during their first admission?",
                    "intent": "historical",
                    "intent_confidence": 0.95,
                    "notes": notes,
                    "expected_retrieval": note_old_id,
                    "expected_rank": 1,
                    "failure_mode": "semantic_collision",
                })

            if len(examples) < n:
                examples.append({
                    "id": f"sc_curr_{len(examples) + 1}",
                    "text": f"What is the patient's current {measurement.lower()}?",
                    "intent": "current",
                    "intent_confidence": 0.95,
                    "notes": notes,
                    "expected_retrieval": note_new_id,
                    "expected_rank": 1,
                    "failure_mode": "semantic_collision",
                })

        return examples[:n]

    def generate_negation_recency(self, n: int = 30) -> List[Dict]:
        """
        Scenario 2: Recent note contains negation of historical condition.
        One query per condition seed; repeat pool if n > pool size.
        """
        pool = list(self.CONDITIONS)
        while len(pool) < n:
            pool = pool + self.CONDITIONS

        examples: List[Dict] = []

        for condition, diagnosis_text, resolution_text in pool:
            if len(examples) >= n:
                break

            old_days = random.randint(1825, 3650)   # 5-10 years
            new_days = random.randint(0, 30)
            old_date = self.reference_date - timedelta(days=old_days)
            new_date = self.reference_date - timedelta(days=new_days)

            note_res_id  = f"note_resolution_{len(examples) + 1}"
            note_diag_id = f"note_diagnosis_{len(examples) + 1}"

            examples.append({
                "id": f"nr_hist_{len(examples) + 1}",
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
                        "semantic_score": 0.88,
                    },
                    {
                        "id": note_diag_id,
                        "text": f"Diagnosis: {diagnosis_text}. Initiated treatment plan.",
                        "offset_days": old_days,
                        "note_date": old_date.strftime("%Y-%m-%d"),
                        "section": "Past Medical History",
                        "semantic_score": 0.92,
                    },
                ],
                "expected_retrieval": note_diag_id,
                "expected_rank": 1,
                "failure_mode": "negation_recency",
                "risk": "Baseline may prioritize recent 'resolution' note and miss historical diagnosis",
            })

        return examples[:n]

    def generate_terminology_drift(self, n: int = 20) -> List[Dict]:
        """
        Scenario 3: Medical terminology has evolved over time.
        One query per pair; repeat pool if n > pool size.
        """
        pool = list(self.TERMINOLOGY_PAIRS)
        while len(pool) < n:
            pool = pool + self.TERMINOLOGY_PAIRS

        examples: List[Dict] = []

        for modern_term, legacy_abbrev, legacy_full in pool:
            if len(examples) >= n:
                break

            old_days = random.randint(3650, 7300)   # 10-20 years
            new_days = random.randint(0, 365)
            old_date = self.reference_date - timedelta(days=old_days)
            new_date = self.reference_date - timedelta(days=new_days)

            note_mod_id = f"note_modern_{len(examples) + 1}"
            note_leg_id = f"note_legacy_{len(examples) + 1}"

            examples.append({
                "id": f"td_hist_{len(examples) + 1}",
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
                        "semantic_score": 0.93,
                    },
                    {
                        "id": note_leg_id,
                        "text": f"New diagnosis: {legacy_full} ({legacy_abbrev}). Patient presented with typical symptoms.",
                        "offset_days": old_days,
                        "note_date": old_date.strftime("%Y-%m-%d"),
                        "section": "History of Present Illness",
                        "semantic_score": 0.78,
                    },
                ],
                "expected_retrieval": note_leg_id,
                "expected_rank": 1,
                "failure_mode": "terminology_drift",
                "risk": "Semantic embeddings may not capture equivalence between legacy and modern terms",
            })

        return examples[:n]

    # ---------------------------------------------------------------
    # Dataset assembly
    # ---------------------------------------------------------------

    def generate_dataset(
        self,
        sc_count: int = 50,
        nr_count: int = 30,
        td_count: int = 20,
        output_path: str = "data/mocks/hard_negatives_v2.json",
    ) -> dict:
        """
        Generates complete hard negative dataset with configurable counts per scenario.
        Writes to output_path; does NOT overwrite hard_negatives.json or
        combined_hard_negatives.json.
        """
        # Safety guard
        forbidden = {"hard_negatives.json", "combined_hard_negatives.json"}
        if Path(output_path).name in forbidden:
            raise ValueError(
                f"Refusing to overwrite protected baseline file: {output_path}. "
                "Use a versioned name such as hard_negatives_v2.json"
            )

        sc = self.generate_semantic_collision(sc_count)
        nr = self.generate_negation_recency(nr_count)
        td = self.generate_terminology_drift(td_count)
        total = len(sc) + len(nr) + len(td)

        dataset = {
            "metadata": {
                "version": "2.0",
                "created": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                "reference_date": self.reference_date.strftime("%Y-%m-%d"),
                "description": "Expanded hard negative scenarios for TIMER-Graph validation (v2)",
                "total_queries": total,
                "counts": {
                    "semantic_collision": len(sc),
                    "negation_recency": len(nr),
                    "terminology_drift": len(td),
                },
            },
            "scenarios": {
                "semantic_collision": sc,
                "negation_recency": nr,
                "terminology_drift": td,
            },
        }

        Path(output_path).parent.mkdir(parents=True, exist_ok=True)

        with open(output_path, "w") as f:
            json.dump(dataset, f, indent=2)

        print(f"Generated {total} hard negative queries")
        print(f"   - Semantic Collision : {len(sc)}")
        print(f"   - Negation Recency   : {len(nr)}")
        print(f"   - Terminology Drift  : {len(td)}")
        print(f"Saved to: {output_path}")

        return dataset


# ---------------------------------------------------------------
# CLI entry-point
# ---------------------------------------------------------------

def parse_args():
    parser = argparse.ArgumentParser(
        description="Generate synthetic hard negative queries for TIMER-Graph evaluation."
    )
    parser.add_argument(
        "--sc", "--semantic-collision",
        type=int, default=50, dest="sc",
        help="Number of Semantic Collision queries (default: 50)",
    )
    parser.add_argument(
        "--nr", "--negation-recency",
        type=int, default=30, dest="nr",
        help="Number of Negation Recency queries (default: 30)",
    )
    parser.add_argument(
        "--td", "--terminology-drift",
        type=int, default=20, dest="td",
        help="Number of Terminology Drift queries (default: 20)",
    )
    parser.add_argument(
        "--output", "-o",
        type=str, default="data/mocks/hard_negatives_v2.json",
        help="Output JSON path (default: data/mocks/hard_negatives_v2.json)",
    )
    parser.add_argument(
        "--reference-date",
        type=str, default="2024-01-15",
        help="Reference date for temporal offset calculations (YYYY-MM-DD)",
    )
    parser.add_argument(
        "--seed",
        type=int, default=42,
        help="Random seed for reproducibility (default: 42)",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    generator = HardNegativeGenerator(
        reference_date=args.reference_date,
        seed=args.seed,
    )
    generator.generate_dataset(
        sc_count=args.sc,
        nr_count=args.nr,
        td_count=args.td,
        output_path=args.output,
    )
