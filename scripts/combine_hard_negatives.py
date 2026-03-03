"""
Merges synthetic hard negatives (v2) with real-world mined candidates
into a single combined dataset for evaluation.

Schema normalisation:
  - Mined candidates are transformed to match the evaluator's query schema.
  - Both notes receive mocked semantic_score=0.95 (semantic collision simulation).

Usage:
    python combine_hard_negatives.py
    python combine_hard_negatives.py \
        --synthetic data/mocks/hard_negatives_v2.json \
        --mined     data/mocks/mined_candidates_v2.json \
        --output    data/mocks/combined_hard_negatives_v2.json
"""

import json
import argparse
from datetime import datetime
from pathlib import Path
from typing import List, Dict


# ---------------------------------------------------------------
# Schema transformer
# ---------------------------------------------------------------

MOCKED_SEMANTIC_SCORE = 0.95   # Simulates semantic collision for evaluator


def transform_mined_candidate(candidate: Dict, idx: int) -> List[Dict]:
    """
    Converts a single mined candidate into one or two evaluator-compatible
    query dicts (one historical + one current query per candidate).

    Input schema (from mine_hard_negatives.py):
      { id, category, phrase, value, temporal_separation_days,
        old_note: { id, text, date, offset_days },
        new_note: { id, text, date, offset_days },
        suggested_queries: { historical, current } }

    Output schema (evaluator-compatible):
      { id, text, intent, intent_confidence, expected_retrieval,
        expected_rank, failure_mode, notes: [ {id, text, offset_days,
        note_date, section, semantic_score}, ... ] }
    """
    old_note = candidate.get("old_note", {})
    new_note = candidate.get("new_note", {})
    suggested = candidate.get("suggested_queries", {})
    category = candidate.get("category", "clinical")

    old_note_id = str(old_note.get("id", f"rw_old_{idx}"))
    new_note_id = str(new_note.get("id", f"rw_new_{idx}"))

    old_offset = int(old_note.get("offset_days", candidate.get("temporal_separation_days", 365)))
    new_offset = int(new_note.get("offset_days", 0))

    # Notes shared by both query variants
    notes = [
        {
            "id": new_note_id,
            "text": str(new_note.get("text", ""))[:300],
            "offset_days": new_offset,
            "note_date": new_note.get("date", ""),
            "section": "Recent Notes",
            "semantic_score": MOCKED_SEMANTIC_SCORE,
        },
        {
            "id": old_note_id,
            "text": str(old_note.get("text", ""))[:300],
            "offset_days": old_offset,
            "note_date": old_note.get("date", ""),
            "section": "Historical Notes",
            "semantic_score": MOCKED_SEMANTIC_SCORE,
        },
    ]

    base_id = idx * 2

    historical_query = {
        "id": f"rw_hist_{base_id + 1}",
        "text": suggested.get(
            "historical",
            f"What was the patient's {category} during their first admission?",
        ),
        "intent": "historical",
        "intent_confidence": 0.85,
        "notes": notes,
        "expected_retrieval": old_note_id,
        "expected_rank": 1,
        "failure_mode": "real_world_mining",
    }

    current_query = {
        "id": f"rw_curr_{base_id + 2}",
        "text": suggested.get(
            "current",
            f"What is the patient's current {category}?",
        ),
        "intent": "current",
        "intent_confidence": 0.85,
        "notes": notes,
        "expected_retrieval": new_note_id,
        "expected_rank": 1,
        "failure_mode": "real_world_mining",
    }

    return [historical_query, current_query]


# ---------------------------------------------------------------
# Main combiner
# ---------------------------------------------------------------

def combine(
    synthetic_path: str = "data/mocks/hard_negatives_v2.json",
    mined_path: str = "data/mocks/mined_candidates_v2.json",
    output_path: str = "data/mocks/combined_hard_negatives_v2.json",
) -> dict:
    """
    Merges synthetic and mined datasets into a single evaluation-ready JSON.
    """
    # Safety guard
    forbidden = {"combined_hard_negatives.json", "hard_negatives.json"}
    if Path(output_path).name in forbidden:
        raise ValueError(
            f"Refusing to overwrite protected baseline file: {output_path}. "
            "Use a versioned name such as combined_hard_negatives_v2.json"
        )

    # ── Load synthetic ────────────────────────────────────────────────
    print(f"📖 Loading synthetic data: {synthetic_path}")
    with open(synthetic_path, "r") as f:
        synthetic = json.load(f)

    synthetic_scenarios: Dict[str, List] = synthetic.get("scenarios", {})
    synthetic_total = sum(len(v) for v in synthetic_scenarios.values())
    print(f"   Synthetic queries: {synthetic_total}")

    # ── Load mined ────────────────────────────────────────────────────
    mined_queries: List[Dict] = []

    if Path(mined_path).exists():
        print(f"📖 Loading mined candidates: {mined_path}")
        with open(mined_path, "r") as f:
            mined_data = json.load(f)

        raw_candidates: List[Dict] = mined_data.get("candidates", [])
        print(f"   Raw mined candidates: {len(raw_candidates)}")

        for idx, candidate in enumerate(raw_candidates, start=1):
            queries = transform_mined_candidate(candidate, idx)
            mined_queries.extend(queries)

        print(f"   Transformed to {len(mined_queries)} evaluator queries")
        print(f"   Note: semantic_score mocked to {MOCKED_SEMANTIC_SCORE} on all mined notes")
    else:
        print(f"⚠️  Mined file not found: {mined_path}")
        print("   Proceeding with synthetic data only.")

    # ── Build combined dataset ────────────────────────────────────────
    combined_scenarios = dict(synthetic_scenarios)   # copy all synthetic scenarios
    if mined_queries:
        combined_scenarios["real_world_mining"] = mined_queries

    total = sum(len(v) for v in combined_scenarios.values())

    dataset = {
        "metadata": {
            "version": "2.0",
            "created": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "description": (
                "Combined hard negative dataset: synthetic v2 + real-world mined "
                "(semantic_score mocked to 0.95 for mined notes)"
            ),
            "total_queries": total,
            "counts": {k: len(v) for k, v in combined_scenarios.items()},
            "sources": {
                "synthetic": synthetic_path,
                "mined": mined_path,
            },
        },
        "scenarios": combined_scenarios,
    }

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(dataset, f, indent=2)

    # ── Summary ───────────────────────────────────────────────────────
    print(f"\n{'='*60}")
    print(f"✅ Combined dataset written: {output_path}")
    print(f"{'='*60}")
    for scenario, queries in combined_scenarios.items():
        print(f"   {scenario:<28}: {len(queries):>4} queries")
    print(f"   {'TOTAL':<28}: {total:>4} queries")

    return dataset


# ---------------------------------------------------------------
# CLI entry-point
# ---------------------------------------------------------------

def parse_args():
    parser = argparse.ArgumentParser(
        description="Combine synthetic and mined hard negative datasets."
    )
    parser.add_argument(
        "--synthetic", type=str, default="data/mocks/hard_negatives_v2.json",
        help="Synthetic hard negatives JSON (default: hard_negatives_v2.json)",
    )
    parser.add_argument(
        "--mined", type=str, default="data/mocks/mined_candidates_v2.json",
        help="Mined candidates JSON (default: mined_candidates_v2.json)",
    )
    parser.add_argument(
        "--output", "-o", type=str, default="data/mocks/combined_hard_negatives_v2.json",
        help="Output path (default: combined_hard_negatives_v2.json)",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    combine(
        synthetic_path=args.synthetic,
        mined_path=args.mined,
        output_path=args.output,
    )
