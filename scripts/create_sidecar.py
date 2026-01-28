import json
import random
from pathlib import Path

# Config
OUTPUT_PATH = "data/mocks/temporal_sidecar.json"
BASELINE_DATA_PATH = "data/evaluation/baseline_dataset.json"
REFERENCE_DATE = "2024-01-15"

def classify_intent_by_section(section):
    """Simple heuristic to assign Ground Truth Intent based on section."""
    if section in ["discharge_instructions", "hospital_course", "lab_results", "chief_complaint"]:
        return "current"
    elif section in ["past_medical", "family_history", "hpi"]:
        return "historical"
    return "current" # Default

def generate_offset(intent):
    if intent == "current":
        return random.randint(0, 7) # 0-1 week old
    else:
        return random.randint(180, 730) # 6 months to 2 years old

def create_sidecar():
    print(f"Reading {BASELINE_DATA_PATH}...")
    with open(BASELINE_DATA_PATH, "r") as f:
        baseline = json.load(f)

    # We'll take the first 50 items to ensure we have enough coverage
    # (The user asked for 10-20, using more covers potential duplicates)
    selected_items = baseline[:50] 

    sidecar = {
        "metadata_version": "1.0",
        "reference_date": REFERENCE_DATE,
        "notes": {},
        "queries": {}
    }

    count = 0
    for i, item in enumerate(selected_items):
        note_id = item["target_note_id"]
        query_text = item["query"]
        section = item["section"]
        
        # Determine Ground Truths
        intent_gt = classify_intent_by_section(section)
        
        # If note already processed, skip re-assigning (keep consistency)
        if note_id not in sidecar["notes"]:
            offset = generate_offset(intent_gt)
            sidecar["notes"][note_id] = {
                "offset_days": offset,
                "note_date": "2024-01-01", # Placeholder, logic uses offset
                "section": section,
                "intent_ground_truth": intent_gt
            }
        
        # Add Query
        query_id = f"q_{i:03d}"
        
        # Valid Window for TRA
        if intent_gt == "current":
            window = [0, 30]
        else:
            window = [0, 3650] # 10 years
            
        sidecar["queries"][query_id] = {
            "text": query_text,
            "intent_ground_truth": intent_gt,
            "valid_temporal_window": window,
            "expected_notes": [note_id]
        }
        count += 1
        if count >= 20: # Limit to 20 queries as requested
            break

    print(f"Generated sidecar with {len(sidecar['notes'])} notes and {len(sidecar['queries'])} queries.")
    
    with open(OUTPUT_PATH, "w") as f:
        json.dump(sidecar, f, indent=2)
    print(f"Saved to {OUTPUT_PATH}")

if __name__ == "__main__":
    create_sidecar()
