from pathlib import Path
from app.etl.normalizer import process_discharge_data
from app.indexing.builder import build_index
from app.retrieval.search import Retriever

# CONFIG PATHS
BASE_DIR = Path(__file__).resolve().parent.parent
RAW_DATA = BASE_DIR / "data" / "raw" / "discharge.csv"
PROCESSED_JSON = BASE_DIR / "data" / "processed" / "discharge.json"
VECTOR_DIR = BASE_DIR / "data" / "vector_store"

# CONFIG PARAMS
ROW_LIMIT = 500 # <-- Set to 5000 for fast POC iteration

def run_pipeline():
    # 1. Run ETL
    print(f"\n1. RUNNING ETL (Limit: {ROW_LIMIT} rows)...")
    count = process_discharge_data(RAW_DATA, PROCESSED_JSON, limit=ROW_LIMIT)
    
    if not count:
        print("ETL produced 0 documents. Stopping.")
        return

    # 2. Run Indexing
    print("\n2. RUNNING INDEXER...")
    build_index([PROCESSED_JSON], VECTOR_DIR)

    # 3. Test Retrieval
    print("\n3. TESTING RETRIEVAL...")
    try:
        retriever = Retriever(VECTOR_DIR)
        
        # Test Query
        query = "patient diagnosed with acute myocardial infarction"
        results = retriever.search(query, k=2)
        
        print(f"\nQuery: {query}")
        for i, res in enumerate(results):
            print(f"\n[Result {i+1}] (Score L2: {res['score']:.4f})")
            print(f"Note ID: {res['note_id']}")
            print(f"Snippet: {res['text'][:150]}...")
            
    except Exception as e:
        print(f"Retrieval failed: {e}")

if __name__ == "__main__":
    run_pipeline()