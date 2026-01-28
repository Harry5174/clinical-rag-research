import sys
import pickle
from pathlib import Path

# Setup paths
BASE_DIR = Path(__file__).resolve().parent.parent
sys.path.append(str(BASE_DIR / "src"))

def verify_metadata():
    index_dir = BASE_DIR / "data" / "vector_store"
    meta_path = index_dir / "poc_metadata.pkl"

    if not meta_path.exists():
        print(f"Error: Metadata file not found at {meta_path}")
        return

    print(f"Loading metadata from {meta_path}...")
    try:
        with open(meta_path, "rb") as f:
            metadata = pickle.load(f)
    except Exception as e:
        print(f"Error loading metadata: {e}")
        return

    if not metadata:
        print("Metadata is empty.")
        return

    # Check the first item
    if isinstance(metadata, dict):
        sample_key = next(iter(metadata))
        sample = metadata[sample_key]
    else:
        sample = metadata[0]
    
    # Check fields
    keys = sample.keys()
    has_offset = "offset_days" in keys
    has_section = "section" in keys
    
    print("-" * 40)
    print(f"Total Metadata Records: {len(metadata)}")
    print(f"Sample Record Keys: {list(keys)}")
    print(f"Has 'offset_days'?: {has_offset}")
    print(f"Has 'section'?: {has_section}")
    print("-" * 40)
    
    if not has_offset:
        print(">>> FINDING: Temporal metadata missing. Use Sidecar Injection (Option A).")
    else:
        print(">>> FINDING: temporal metadata present.")

if __name__ == "__main__":
    verify_metadata()
