import pandas as pd
import json
import sys
from pathlib import Path
from typing import Optional

sys.path.append(str(Path(__file__).resolve().parents[3]))

from src.app.indexing.advanced_chunking_strategy import AdvancedChunker, ChunkConfig, ChunkingStrategy

def process_discharge_data(
    input_path: Path, 
    output_path: Path, 
    limit: Optional[int] = None
):
    print(f"--- ETL STARTED: {input_path.name} (Strategy: Hybrid Semantic) ---")
    
    if not input_path.exists():
        raise FileNotFoundError(f"Input file not found: {input_path}")

    # Load Data (Respecting the limit for speed)
    if limit:
        print(f"Limiting import to first {limit} rows...")
        df = pd.read_csv(input_path, nrows=limit)
    else:
        df = pd.read_csv(input_path)

    # Filter for Cardiac Keywords
    cardiac_keywords = ["cardiac", "heart", "myocardial", "coronary", "ECG", "heart failure", "atrial"]
    df["text"] = df["text"].fillna("")
    df_filtered = df[df["text"].str.contains("|".join(cardiac_keywords), case=False, na=False)]
    
    print(f"Processing {len(df_filtered)} cardiac-related records...")

    # --- CONFIGURATION: The 'Smart' Part ---
    # Hybrid strategy respects medical headers (e.g. 'Discharge Diagnosis')
    config = ChunkConfig(
        strategy=ChunkingStrategy.HYBRID,
        chunk_size=250,  
        overlap=50,     
        min_chunk_size=50
    )
    chunker = AdvancedChunker(config)
    # ----------------------------------------

    documents = []
    
    for idx, row in df_filtered.iterrows():
        base_meta = {
            "note_id": str(row.get("note_id", "")),
            "subject_id": str(row.get("subject_id", "")),
            "hadm_id": str(row.get("hadm_id", "")),
            "note_type": str(row.get("note_type", "discharge")),
            "charttime": str(row.get("charttime", ""))
        }

        try:
            # Use the Advanced Chunker
            row_chunks = chunker.chunk_text(row["text"], base_meta)
            documents.extend(row_chunks)
        except Exception as e:
            print(f"Error chunking note {base_meta['note_id']}: {e}")
            continue

    # Save Output
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(documents, f, indent=2)
    
    print(f"--- ETL COMPLETE: Saved {len(documents)} semantic chunks to {output_path.name} ---")
    return len(documents)

# import pandas as pd
# import json
# from pathlib import Path
# from typing import Optional, List, Dict
# import sys

# # Add the src directory to path to allow imports
# sys.path.append(str(Path(__file__).resolve().parents[3]))

# # Import your Advanced Chunker
# from src.app.indexing.advanced_chunking_strategy import AdvancedChunker, ChunkConfig, ChunkingStrategy

# def process_discharge_data(
#     input_path: Path, 
#     output_path: Path, 
#     limit: Optional[int] = None
# ):
#     print(f"--- ETL STARTED: {input_path.name} (Strategy: Hybrid Semantic) ---")
    
#     if not input_path.exists():
#         raise FileNotFoundError(f"Input file not found: {input_path}")

#     # Load Data
#     if limit:
#         print(f"Limiting import to first {limit} rows...")
#         df = pd.read_csv(input_path, nrows=limit)
#     else:
#         df = pd.read_csv(input_path)

#     # Filter for Cardiac Keywords
#     cardiac_keywords = ["cardiac", "heart", "myocardial", "coronary", "ECG", "heart failure", "atrial"]
#     df["text"] = df["text"].fillna("")
#     df_filtered = df[df["text"].str.contains("|".join(cardiac_keywords), case=False, na=False)]
    
#     print(f"Processing {len(df_filtered)} cardiac-related records...")

#     # --- NEW: Initialize Advanced Chunker ---
#     # We use HYBRID strategy: Semantic Sections (keeping headers) + Smart Sentence Splitting
#     config = ChunkConfig(
#         strategy=ChunkingStrategy.HYBRID,
#         chunk_size=200,  # Keep reasonable size
#         overlap=75,      # High overlap for safety
#         min_chunk_size=50
#     )
#     chunker = AdvancedChunker(config)
#     # ----------------------------------------

#     documents = []
    
#     for idx, row in df_filtered.iterrows():
#         # Create base metadata
#         base_meta = {
#             "note_id": str(row.get("note_id", "")),
#             "subject_id": str(row.get("subject_id", "")),
#             "hadm_id": str(row.get("hadm_id", "")),
#             "note_type": str(row.get("note_type", "discharge")),
#             "charttime": str(row.get("charttime", ""))
#         }

#         # Use Advanced Chunker instead of simple loop
#         try:
#             row_chunks = chunker.chunk_text(row["text"], base_meta)
#             documents.extend(row_chunks)
#         except Exception as e:
#             print(f"Error chunking note {base_meta['note_id']}: {e}")
#             continue

#     # Save Output
#     output_path.parent.mkdir(parents=True, exist_ok=True)
#     with open(output_path, "w") as f:
#         json.dump(documents, f, indent=2)
    
#     print(f"--- ETL COMPLETE: Saved {len(documents)} semantic chunks to {output_path.name} ---")
#     return len(documents)

# -----------------------------------------------------------------------------------------------------
# # import pandas as pd
# # import re
# # import json
# # from pathlib import Path
# # from typing import Iterator, Dict, Any, Optional

# # def clean_text(text: Any) -> str:
# #     """Cleans text: normalizes whitespace and removes underscores."""
# #     text = str(text)
# #     text = re.sub(r"\s+", " ", text)
# #     return text.replace("___", "").strip()

# # def chunk_text_with_overlap(text: str, chunk_size: int, overlap: int) -> Iterator[str]:
# #     """Chunks text into overlapping segments."""
# #     words = text.split()
# #     if not words:
# #         return
    
# #     step = max(1, chunk_size - overlap)
    
# #     for start in range(0, len(words), step):
# #         if start > 0 and start + chunk_size > len(words) + overlap:
# #              break 
# #         yield " ".join(words[start:start + chunk_size])

# # def process_discharge_data(
# #     input_path: Path, 
# #     output_path: Path, 
# #     chunk_size: int = 250, 
# #     overlap: int = 50,
# #     limit: Optional[int] = None  # <-- NEW PARAMETER
# # ):
# #     print(f"--- ETL STARTED: {input_path.name} ---")
    
# #     if not input_path.exists():
# #         raise FileNotFoundError(f"Input file not found: {input_path}")

# #     # OPTIMIZATION: Only read the first 'limit' rows
# #     if limit:
# #         print(f"Limiting import to first {limit} rows...")
# #         df = pd.read_csv(input_path, nrows=limit)
# #     else:
# #         df = pd.read_csv(input_path)
        
# #     print(f"Loaded {len(df)} rows into memory.")

# #     # 1. Clean
# #     df["clean_text"] = df["text"].apply(clean_text)

# #     # 2. Filter (Cardiac Context)
# #     cardiac_keywords = ["cardiac", "heart", "myocardial", "coronary", "ECG", "heart failure", "atrial"]
# #     df_filtered = df[df["clean_text"].str.contains("|".join(cardiac_keywords), case=False, na=False)]
    
# #     print(f"Rows after cardiac keyword filter: {len(df_filtered)} (dropped {len(df) - len(df_filtered)})")

# #     if df_filtered.empty:
# #         print("Warning: No data matched keywords in this subset.")
# #         return 0

# #     # 3. Chunk & Structure
# #     documents = []
# #     metadata_cols = ["note_id", "subject_id", "hadm_id", "note_type", "charttime"]

# #     for _, row in df_filtered.iterrows():
# #         base_meta = {col: str(row.get(col)) if pd.notna(row.get(col)) else None for col in metadata_cols}
# #         base_meta["source_file"] = input_path.name

# #         for idx, chunk in enumerate(chunk_text_with_overlap(row["clean_text"], chunk_size, overlap)):
# #             doc = base_meta.copy()
# #             doc["chunk_index"] = idx
# #             doc["chunk_text"] = chunk
# #             documents.append(doc)

# #     # 4. Save
# #     output_path.parent.mkdir(parents=True, exist_ok=True)
# #     with open(output_path, "w") as f:
# #         json.dump(documents, f, indent=2)
    
# #     print(f"--- ETL COMPLETE: Saved {len(documents)} chunks to {output_path.name} ---")
# #     return len(documents)