import json
import random
import re
from pathlib import Path
import sys

# CONFIG
BASE_DIR = Path(__file__).resolve().parent.parent.parent
INPUT_JSON = BASE_DIR / "data" / "processed" / "discharge.json"
OUTPUT_JSON = BASE_DIR / "data" / "evaluation" / "baseline_dataset.json"
TARGET_SAMPLE_SIZE = 100

BOILERPLATE_TRIGGERS = [
    "Admission Date:", "Discharge Date:", "Date of Birth:", 
    "Service: MEDICINE", "No Known Allergies", "Social History:",
    "Family History:", "Discharge Diagnosis:", "Attending:", "Chief Complaint:"
]

def generate_clean_query(chunk_text):
    # 1. Reject if chunk looks like a header/footer
    if any(phrase in chunk_text for phrase in BOILERPLATE_TRIGGERS):
        return None

    # 2. Split into sentences
    sentences = re.split(r'(?<!\w\.\w.)(?<![A-Z][a-z]\.)(?<=\.|\?)\s', chunk_text)
    # Filter: at least 10 words, no too many placeholders
    valid_sentences = [s.strip() for s in sentences if len(s.split()) > 10 and s.count("___") < 3]
    
    if not valid_sentences:
        return None
    
    # 3. Pick the longest non-boilerplate sentence
    query = max(valid_sentences, key=len)
    
    # 4. Final safety check on the QUERY itself
    if len(query) < 50 or any(phrase in query for phrase in BOILERPLATE_TRIGGERS):
        return None
        
    return query

def main():
    print(f"Loading data from {INPUT_JSON}...")
    if not INPUT_JSON.exists():
        print(f"Error: Processed data not found at {INPUT_JSON}")
        return

    with open(INPUT_JSON, "r") as f:
        data = json.load(f)

    print(f"Total chunks loaded: {len(data)}")

    # Group by section
    section_groups = {}
    for item in data:
        section = item.get("section", "unstructured")
        if section not in section_groups:
            section_groups[section] = []
        section_groups[section].append(item)

    print(f"Found sections: {list(section_groups.keys())}")

    baseline_set = []
    
    # We want a diverse set, so we try to pick from each section
    sections = sorted(section_groups.keys())
    per_section_target = max(1, TARGET_SAMPLE_SIZE // len(sections))
    
    for section in sections:
        candidates = section_groups[section]
        random.shuffle(candidates)
        
        section_count = 0
        for cand in candidates:
            if section_count >= per_section_target + 5: # Small buffer
                break
                
            query = generate_clean_query(cand["chunk_text"])
            if query:
                baseline_set.append({
                    "query": query,
                    "target_chunk_id": cand.get("chunk_id"),
                    "target_note_id": cand.get("note_id"),
                    "section": section,
                    "original_text": cand["chunk_text"]
                })
                section_count += 1
    
    # If we don't have enough, fill from the rest
    if len(baseline_set) < TARGET_SAMPLE_SIZE:
        print(f"Currently have {len(baseline_set)} queries. Filling to {TARGET_SAMPLE_SIZE}...")
        all_remaining = []
        for section in sections:
            all_remaining.extend(section_groups[section])
        random.shuffle(all_remaining)
        
        for cand in all_remaining:
            if len(baseline_set) >= TARGET_SAMPLE_SIZE:
                break
            # Skip if already in baseline
            if any(b["target_chunk_id"] == cand.get("chunk_id") for b in baseline_set):
                continue
                
            query = generate_clean_query(cand["chunk_text"])
            if query:
                baseline_set.append({
                    "query": query,
                    "target_chunk_id": cand.get("chunk_id"),
                    "target_note_id": cand.get("note_id"),
                    "section": cand.get("section", "unstructured"),
                    "original_text": cand["chunk_text"]
                })

    # Final trim
    baseline_set = baseline_set[:TARGET_SAMPLE_SIZE]
    
    print(f"Saving {len(baseline_set)} baseline samples to {OUTPUT_JSON}...")
    OUTPUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT_JSON, "w") as f:
        json.dump(baseline_set, f, indent=2)

    print("Done!")

if __name__ == "__main__":
    main()
