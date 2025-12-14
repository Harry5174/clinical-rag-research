from pathlib import Path
import json

BASE_DIR = Path().resolve().parent
VECTOR_DIR = BASE_DIR / ".." / "data" / "vector_store"

from app.retrieval.search import Retriever
retriever = Retriever(VECTOR_DIR)

# 1. Test: Specific Procedure & Volume (Targets Chunk 4 & 5)
# This tests if it can find specific numbers and procedural context.
query_1 = "therapeutic paracentesis with 4.3L fluid removed"
print(f"\n--- Query 1: {query_1} ---")
results_1 = retriever.search(query_1, k=3)
print(json.dumps(results_1, indent=2))

# 2. Test: Medication Non-Compliance (Targets Chunk 3)
# This tests if it captures the concept of a patient not taking meds.
query_2 = "patient noncompliant with lactulose prescription"
print(f"\n--- Query 2: {query_2} ---")
results_2 = retriever.search(query_2, k=3)
print(json.dumps(results_2, indent=2))

# 3. Test: Complex Comorbidities (Targets Chunk 1 & 6)
# This tests if it links multiple conditions (HIV, COPD, Cirrhosis) together.
query_3 = "patient with HIV on HAART and HCV cirrhosis"
print(f"\n--- Query 3: {query_3} ---")
results_3 = retriever.search(query_3, k=3)
print(json.dumps(results_3, indent=2))

# 4. Test: Specific Lab/Diagnostic Finding (Targets Chunk 6)
# This tests retrieval of specific numerical test results.
query_4 = "COPD with FEV1 0.88"
print(f"\n--- Query 4: {query_4} ---")
results_4 = retriever.search(query_4, k=3)
print(json.dumps(results_4, indent=2))