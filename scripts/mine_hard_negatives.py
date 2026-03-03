"""
Mines real-world hard negatives from discharge.csv by finding:
1. Repeated phrases across different time periods
2. Temporal separation > min_separation_days
3. Clinical significance (vitals, diagnoses, medications)

Usage:
    python mine_hard_negatives.py                         # defaults
    python mine_hard_negatives.py --max-matches 50 --nrows 100000
    python mine_hard_negatives.py --output data/mocks/mined_candidates_v2.json
"""

import pandas as pd
import re
import argparse
from collections import defaultdict
from datetime import datetime
from typing import List, Dict, Tuple
import json
from pathlib import Path


class HardNegativeMiner:
    """
    Extracts hard negative scenarios from real clinical data.
    """

    def __init__(
        self,
        data_path: str = "data/processed/discharge.json",
        nrows: int = None,
    ):
        self.data_path = Path(data_path)
        self.nrows = nrows

        if self.data_path.exists():
            if self.data_path.suffix == ".json":
                with open(self.data_path, "r") as f:
                    self.data = json.load(f)
                self.df = pd.DataFrame(self.data)
            else:
                # CSV path — apply nrows cap to protect memory on large files
                self.df = pd.read_csv(data_path, nrows=nrows)
        else:
            self.df = pd.DataFrame()   # Empty DataFrame if file missing

        # Clinical patterns to search for
        self.patterns = {
            "vitals": [
                r"blood pressure:?\s*(\d+/\d+)",
                r"heart rate:?\s*(\d+)\s*bpm",
                r"temperature:?\s*([\d.]+)\s*[°f]",
                r"weight:?\s*([\d.]+)\s*kg",
                r"BMI:?\s*([\d.]+)",
            ],
            "labs": [
                r"creatinine:?\s*([\d.]+)",
                r"hemoglobin:?\s*([\d.]+)",
                r"glucose:?\s*(\d+)",
                r"sodium:?\s*(\d+)",
                r"potassium:?\s*([\d.]+)",
            ],
            "diagnoses": [
                r"diagnosed with\s+([^.,:]+)",
                r"history of\s+([^.,:]+)",
                r"admission for\s+([^.,:]+)",
            ],
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
        min_separation_days: int = 180,
        max_matches: int = 50,
    ) -> List[Dict]:
        """
        Finds notes with repeated phrases separated by > min_separation_days.
        """
        if self.df.empty:
            return []

        phrase_index = defaultdict(list)

        from tqdm import tqdm

        print(f"DataFrame shape: {self.df.shape}")
        if self.nrows:
            print(f"Reading limited to {self.nrows:,} rows (--nrows cap)")

        for idx, row in tqdm(self.df.iterrows(), total=len(self.df), desc="Mining"):
            text = str(row.get("chunk_text", row.get("text", "")))
            date_str = str(row.get("charttime", row.get("chartdate", "")))

            if not text or not date_str or date_str.lower() == "nan":
                continue

            try:
                date = pd.to_datetime(date_str)
            except Exception:
                continue

            patient_id = row.get("subject_id", "unknown")
            note_id = row.get("note_id", row.get("row_id", f"note_{idx}"))

            for category in self.patterns.keys():
                phrases = self.extract_phrases(text, category)
                for phrase, value in phrases:
                    key = f"{patient_id}_{category}_{value}"
                    phrase_index[key].append(
                        {
                            "note_id": note_id,
                            "text": text,
                            "date": date,
                            "phrase": phrase,
                            "value": value,
                            "category": category,
                        }
                    )

        hard_negatives = []

        for key, occurrences in phrase_index.items():
            if len(occurrences) < 2:
                continue

            occurrences.sort(key=lambda x: x["date"])

            oldest = occurrences[0]
            newest = occurrences[-1]
            days_diff = (newest["date"] - oldest["date"]).days

            if days_diff >= min_separation_days:
                hard_negatives.append(
                    {
                        "id": f"mined_{len(hard_negatives) + 1}",
                        "category": oldest["category"],
                        "phrase": oldest["phrase"],
                        "value": oldest["value"],
                        "temporal_separation_days": days_diff,
                        "old_note": {
                            "id": str(oldest["note_id"]),
                            "text": oldest["text"][:200],
                            "date": oldest["date"].strftime("%Y-%m-%d"),
                            "offset_days": days_diff,
                        },
                        "new_note": {
                            "id": str(newest["note_id"]),
                            "text": newest["text"][:200],
                            "date": newest["date"].strftime("%Y-%m-%d"),
                            "offset_days": 0,
                        },
                        "suggested_queries": {
                            "historical": (
                                f"What was the patient's {oldest['category']} "
                                f"during their first admission?"
                            ),
                            "current": (
                                f"What is the patient's current {oldest['category']}?"
                            ),
                        },
                    }
                )

                if len(hard_negatives) >= max_matches:
                    break

        return hard_negatives

    def export_for_annotation(
        self,
        output_path: str = "data/mocks/mined_candidates_v2.json",
        min_separation_days: int = 180,
        max_matches: int = 50,
    ) -> List[Dict]:
        """
        Exports mined candidates for manual review / downstream combining.
        """
        # Safety guard
        forbidden = {"mined_candidates.json"}
        if Path(output_path).name in forbidden:
            raise ValueError(
                f"Refusing to overwrite protected baseline file: {output_path}. "
                "Use a versioned name such as mined_candidates_v2.json"
            )

        candidates = self.find_temporal_duplicates(
            min_separation_days=min_separation_days,
            max_matches=max_matches,
        )

        annotation_template = {
            "metadata": {
                "created": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                "total_candidates": len(candidates),
                "status": "awaiting_annotation",
                "nrows_used": self.nrows,
                "min_separation_days": min_separation_days,
                "instructions": (
                    "Review each candidate and add: "
                    "intent, expected_retrieval, semantic_score estimates"
                ),
            },
            "candidates": candidates,
        }

        Path(output_path).parent.mkdir(parents=True, exist_ok=True)

        with open(output_path, "w") as f:
            json.dump(annotation_template, f, indent=2)

        print(f"✅ Mined {len(candidates)} hard negative candidates")
        print(f"📁 Saved to: {output_path}")
        if len(candidates) == 0:
            print(
                "⚠️  Zero candidates found. Try: --min-days 30, or check CSV column names."
            )
        else:
            print("\n📝 Next step: Run combine_hard_negatives.py to merge with synthetic data")

        return candidates


# ---------------------------------------------------------------
# CLI entry-point
# ---------------------------------------------------------------

def parse_args():
    parser = argparse.ArgumentParser(
        description="Mine real-world hard negative candidates from clinical notes."
    )
    parser.add_argument(
        "--data", type=str, default="data/raw/discharge.csv",
        help="Path to source data (CSV or JSON).",
    )
    parser.add_argument(
        "--nrows", type=int, default=100000,
        help="Rows to read from CSV (memory cap, default 100000 ≈ 50MB).",
    )
    parser.add_argument(
        "--min-days", type=int, default=180,
        help="Minimum temporal separation in days (default 180).",
    )
    parser.add_argument(
        "--max-matches", type=int, default=50,
        help="Maximum candidates to mine (default 50).",
    )
    parser.add_argument(
        "--output", "-o", type=str, default="data/mocks/mined_candidates_v2.json",
        help="Output JSON path.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    miner = HardNegativeMiner(data_path=args.data, nrows=args.nrows)

    if miner.df.empty:
        print(f"⚠️  Data not found or empty at: {args.data}")
        print("   Using synthetic data only. Exiting miner.")
    else:
        candidates = miner.export_for_annotation(
            output_path=args.output,
            min_separation_days=args.min_days,
            max_matches=args.max_matches,
        )
        if candidates:
            print("\n📊 Sample Candidate:")
            sample = {k: v for k, v in candidates[0].items() if k != "old_note" and k != "new_note"}
            print(json.dumps(sample, indent=2))
