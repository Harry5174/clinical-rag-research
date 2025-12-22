# Clinical RAG Research Framework

**A research-grade Retrieval-Augmented Generation (RAG) framework for unstructured clinical notes.**

> **Status**: ✅ Phase 3 Complete (Recall@5: **0.93**)

## 🔬 Project Goal
To develop and benchmark **structure-aware retrieval techniques** that outperform naive RAG baselines in the medical domain. This project focuses on high-precision retrieval from complex EHR documents (e.g., Discharge Summaries) by addressing domain-specific challenges such as:
- **Long-range Dependencies**: Linking "Medications" headers to list items hundreds of lines away.
- **Ambiguous Terminology**: Disambiguating clinical abbreviations ("Pt", "SOB") using context.

## 📊 Key Results
We systematically evaluated three RAG strategies on the MIMIC-IV dataset.

| Strategy | Architecture | Recall@5 | MRR | Notes |
| :--- | :--- | :--- | :--- | :--- |
| **Phase 0: Naive** | Fixed-size Chunking (250w) | 0.5300 | 0.4158 | Baseline failure; loses section context. |
| **Phase 1: Hybrid** | **Structure-Aware Chunking** | **0.8700** | 0.7670 | **+34% gain**. Respects medical headers. |
| **Phase 2: HeaderProp** | Explicit Header Injection | 0.8500 | 0.7590 | Regression. Introduced semantic noise. |
| **Phase 3: Reranking** | **Hybrid + Cross-Encoder** | **0.9300** | **0.8482** | **State-of-the-Art**. Filters false positives. |

**Conclusion**: The optimal pipeline is **Hybrid Semantic Chunking** coupled with **Cross-Encoder Reranking**.

## 🏗️ Architecture
This repository implements a modular pipeline for incremental experimentation:

1.  **ETL & Chunking** (`src/app/research/chunking`)
    -   `AdvancedChunker`: Implements both "Hybrid" (Rule-based sectioning) and "Header Propagation" strategies.
    
2.  **Indexing & Retrieval** (`src/app/research/retrieval`)
    -   **Bi-Encoder**: `BAAI/bge-base-en-v1.5` for candidate generation.
    -   **Cross-Encoder**: `BAAI/bge-reranker-base` for precision reranking (Phase 3).
    -   **Vector Store**: FAISS IndexFlatL2.

3.  **Evaluation** (`src/app/evaluation`)
    -   Automated benchmarking scripts for Recall@K, MRR, and Latency.

## 🚀 Getting Started

### Prerequisites
- Python 3.10+
- [MIMIC-IV](https://physionet.org/content/mimiciv/) or equivalent clinical text dataset (CSV format).

### Installation
```bash
# Clone the repository
git clone https://github.com/Harry5174/research.git
cd clinical-rag-research

# Install dependencies
pip install -r requirements.txt
```

### Usage

**1. Sourcing Data**
Place your `discharge.csv` in `data/raw/`.

**2. Baseline Pipeline (Phase 1)**
```bash
python src/app/baseline/pipelines.py
```

**3. Advanced Indexing (Phase 2)**
```bash
python src/app/research/indexing/prepare_header_prop_index.py
```

**4. Run Benchmarks (Phase 3)**
```bash
python src/app/evaluation/runners/run_phase3_eval.py
```
*Note: Phase 3 requires downloading the ~1.1GB Reranker model.*

## 📚 Reports
Detailed experimental logs and analysis are available in `reports/`:
- [x] [Phase 1: Baseline Benchmarking](reports/phase1_baseline.md)
- [x] [Phase 2: Header Propagation Analysis](reports/phase2_header_prop.md)
- [x] [Phase 3: Multi-Stage Reranking](reports/phase3_reranking.md)

## 👤 Author
Harry
