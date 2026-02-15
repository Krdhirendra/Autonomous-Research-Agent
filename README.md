# ARA — Autonomous Research Agent

> deployed version: [aragent.streamlit.app](https://aragent.streamlit.app/)

> An end-to-end multi-source AI research assistant that autonomously searches, verifies, retrieves, and synthesizes structured academic reports from research papers, blogs, and web articles.
---

## Overview

It is not a simple pdf RAG based chatbot instead:
ARA (Autonomous Research Agent) is a multi-stage agentic system which:

1. Understands a user’s research query
2. Generates academic search queries
3. Collects relevant research papers and web articles
4. Extracts and embeds the acquired content (PDF + HTML supported)
5. Performs multi-query RAG retrieval
6. Verifies relevance using an LLM
7. Synthesizes a structured research-style report
8. Exports the final output as a downloadable PDF
---

## System Architecture
<img width="3456" height="1204" alt="image" src="https://github.com/user-attachments/assets/27b1e5df-7e06-4bc8-b19a-522341f85aaa" />

---

## Project Structure

```
ARA_vector_store/        # Persistent ChromaDB embeddings
Version-1/               # Initial implementation
Version-2/               # Final improved implementation
hosting_files/           # Streamlit deployment files
reports/                 # Sample generated reports
```

### Core Files

* `hosting.py` — Streamlit app entry point 
* `hosting_V2_RAG_classes.py` — RAG architecture classes 
* `V2_project_ARA.ipynb` — Development notebook
* `V2_RAG_classes.py` — Modular RAG implementation

---

## Key Technical Components

### 1️. Task Understanding

Uses LLM to classify user query into:
* research
* survey
* comparison

and Extracts:
* topic
* methods
* constraints
---

### 2️. Multi-Source Ingestion

It supports:
* Research PDFs
* Blogs
* Web articles
* HTML content

PDF extraction using `pypdf`
HTML extraction using `readability` + `BeautifulSoup`

No file downloads is required instead data is loaded and converted into embeddings and only those embeddings are stored with their respective chunks.

---

### 3️. Embedding & Vector Store

* Model: `all-MiniLM-L6-v2`
* Vector DB: ChromaDB (persistent)
* Chunking: RecursiveCharacterTextSplitter
* Metadata tracking per source
---

### 4️. Multi-Query Retrieval

Two-level expansion:

* Search query expansion (for web search)
* Retrieval query expansion (for RAG)

Improves recall and coverage.

---

### 5️. Paper-Level Reasoning

Chunks are grouped into paper-level objects:

```python
{
  "source": url,
  "chunks": [...],
  "best_chunk": "...",
  "score": similarity_score
}
```

This enables:

* Relevance filtering
* Structured reasoning
* Better synthesis
---

### 6️. Verification Layer

Each paper is evaluated by an LLM and classified as:

* survey
* comparison
* method-specific
* tutorial

Non-relevant sources are discarded.

---

### 7️. Report Generation

Final LLM synthesizes:

* Structured academic-style report
* Explicit references (URLs included)
* No hallucinated information
* Acknowledges missing evidence when necessary
---

## Tech Stack

* Python
* Streamlit
* LangChain
* ChromaDB
* SentenceTransformers
* Groq LLM API
* Tavily Search API
* pypdf
* BeautifulSoup
* readability-lxml
---

## ⚙️ Installation

```bash
git clone <https://github.com/Krdhirendra/Autonomous-Research-Agent/>
cd ARA
pip install -r requirements.txt
```

Add environment variables:
```
GROQ_TOKEN=your_key
TAVILY_API_KEY=your_key
```

Run locally:

```bash
streamlit run hosting.py
```
---

## Improvements Over Version 1

| Feature             | V1    | V2         |
| ------------------- | ----- | ---------- |
| PDF-only            | ✅     | ❌          |
| Multi-source        | ❌     | ✅          |
| Query expansion     | ❌     | ✅          |
| Verification stage  | Basic | Structured |
| In-memory ingestion | ❌     | ✅          |
| Report formatting   | Basic | Structured |

---
##  Author

**Dhirendra Kumar**

Autonomous Systems | Retrieval AI | ML Infrastructure

---
