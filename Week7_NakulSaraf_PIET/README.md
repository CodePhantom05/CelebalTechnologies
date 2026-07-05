# 📄 RAG Document Question Answering System

A Retrieval-Augmented Generation (RAG) based Document Question Answering system built using Python, FAISS, Sentence Transformers, Hugging Face Transformers, and Streamlit.

The application allows users to upload PDF, TXT, or Markdown documents, indexes them into a vector database, retrieves relevant information using semantic search, and generates context-aware answers.

---

## Features

- Upload PDF, TXT, and Markdown documents
- Automatic document ingestion and text chunking
- Semantic search using Sentence Transformers
- FAISS vector database for efficient retrieval
- Vector and Hybrid (BM25 + Vector) retrieval
- Optional keyword-based reranking
- Local answer generation using FLAN-T5
- Optional Cohere API support
- Interactive Streamlit interface
- Validation log generation
- System metrics report generation
- Offline testing mode

---

## Project Structure

```
project/
│
├── data/                      # Documents
├── logs/                      # Validation and metrics reports
├── src/
│   ├── document_ingestion.py
│   ├── text_chunking.py
│   ├── embeddings.py
│   ├── vector_store.py
│   ├── retrieval.py
│   ├── generation.py
│   ├── rag_pipeline.py
│   ├── evaluation.py
│   └── metrics_report.py
│
├── streamlit_app.py
├── validate_pipeline.py
├── requirements.txt
└── README.md
```

---

## Workflow

```
Documents
     │
     ▼
Document Ingestion
     │
     ▼
Text Chunking
     │
     ▼
Embedding Generation
     │
     ▼
FAISS Vector Store
     │
     ▼
Query Embedding
     │
     ▼
Retriever
     │
     ▼
Language Model
     │
     ▼
Generated Answer
```

---

## Technologies Used

- Python
- Streamlit
- FAISS
- Sentence Transformers
- Hugging Face Transformers
- PyMuPDF
- NumPy
- Rank-BM25
- Cohere API (Optional)

---

## Installation

Clone the repository

```bash
git clone <repository-url>
cd <repository-folder>
```

Create a virtual environment

```bash
python -m venv venv
```

Activate the virtual environment

### Windows

```bash
venv\Scripts\activate
```

### Linux / macOS

```bash
source venv/bin/activate
```

Install dependencies

```bash
pip install -r requirements.txt
```

---

## Running the Application

Start the Streamlit application

```bash
streamlit run streamlit_app.py
```

The application will open in your browser.

---

## Using the Application

1. Upload one or more PDF, TXT, or Markdown files.
2. Select the embedding model.
3. Select the generation model.
4. Configure chunk size and retrieval settings.
5. Ask questions about the uploaded documents.
6. View retrieved document chunks.
7. Generate validation logs and system metrics.

---

## Supported Embedding Models

- sentence-transformers/all-MiniLM-L6-v2
- sentence-transformers/all-mpnet-base-v2
- BAAI/bge-small-en-v1.5

---

## Supported Generation Models

### Local

- google/flan-t5-small
- google/flan-t5-base
- google/flan-t5-large

### Cloud (Optional)

- Cohere Command-R

---

## Retrieval Modes

### Vector Search

Uses semantic similarity between document and query embeddings.

### Hybrid Search

Combines:

- FAISS semantic search
- BM25 keyword search

---

## Validation

Run validation from the Streamlit interface or using:

```bash
python validate_pipeline.py --docs data
```

Options

```bash
python validate_pipeline.py --docs data --mode hybrid --rerank
```

Offline mode

```bash
python validate_pipeline.py --docs data --offline
```

Generated files

```
logs/
│
├── validation_log.md
├── validation_log.json
└── system_metrics_report.md
```

---

## System Components

### Document Ingestion

Loads PDF, TXT, and Markdown documents.

### Text Chunking

Splits documents into overlapping chunks.

### Embedding Generation

Converts chunks into dense vector embeddings.

### Vector Store

Stores embeddings using FAISS.

### Retrieval

Retrieves the most relevant document chunks.

### Answer Generation

Uses FLAN-T5 or Cohere to generate grounded responses.

### Evaluation

Produces validation logs and retrieval metrics.

### Metrics Report

Summarizes the system configuration.

---

## Future Improvements

- Cross-Encoder reranking
- Multi-document summarization
- Persistent vector database
- Metadata filtering
- OCR support for scanned PDFs
- Citation highlighting
- Conversation memory
- Multi-language document support

---

## Example

**Question**

```
What are the main topics discussed in the uploaded document?
```

**Answer**

```
The system retrieves the most relevant document sections and generates a response based only on the uploaded document content.
```