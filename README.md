# Clinical RAG Research Framework

**A research-grade Retrieval-Augmented Generation (RAG) framework for unstructured clinical notes.**

## 🔬 Project Goal
To develop and benchmark **structure-aware retrieval techniques** that outperform naive RAG baselines in the medical domain. This project focuses on high-precision retrieval from complex EHR documents (e.g., Discharge Summaries) by addressing domain-specific challenges such as:
- **Long-range Dependencies**: Linking "Medications" headers to list items hundreds of lines away.
- **Ambiguous Terminology**: Disambiguating clinical abbreviations ("Pt", "SOB") using context.
- **Section Relevance**: Prioritizing "History of Present Illness" over "Family History" for diagnostic queries.

## 🏗️ Architecture
This repository implements a modular pipeline for incremental experimentation:

1.  **ETL & Chunking**
    -   `Hybrid Semantic Chunking`: Preserves medical section boundaries.
    -   *Experimental*: **Header Propagation** (Injecting section context into individual chunks).

2.  **Indexing & Retrieval**
    -   **Vector Store**: FAISS (Sparse/Dense extraction).
    -   **Embeddings**: Support for `BGE-Base`, `BioBERT`, and other clinical encoders.
    -   *Experimental*: **Two-Stage Retrieval** (ColBERTv2 Reranking).

3.  **Evaluation**
    -   `MultiScenario` Benchmark: Automated testing for Clinical Findings, Medication Regimens, and Temporal Reasoning.

## 🚀 Getting Started

### Prerequisites
- Python 3.10+
- [MIMIC-IV](https://physionet.org/content/mimiciv/) or equivalent clinical text dataset (CSV format).

### Installation
```bash
# Clone the repository
git clone https://github.com/your-org/clinical-rag-research.git
cd clinical-rag-research

# Install dependencies
pip install -r requirements.txt
```

### Usage
1.  **Sourcing Data**: Place your `discharge.csv` in `data/raw/`.
2.  **Indexing**:
    ```bash
    python src/app/indexing/builder.py --config configs/hybrid_chunking.yaml
    ```
3.  **Evaluation**:
    ```bash
    python src/app/evaluation/run_eval.py
    ```

## 📚 Modules & Roadmap
- [x] **Baseline**: Hybrid Chunking + BGE Embedding.
- [ ] **Module 1**: Header Propagation Indexing (In Progress).
- [ ] **Module 2**: Two-Stage Reranking with ColBERT.
