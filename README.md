# 🐙 Ollama Vector Embedding Example

<p align="center">
   
[![Status](https://img.shields.io/badge/status-experimental-6B2B44?style=for-the-badge)](#)
[![Made with Ollama](https://img.shields.io/badge/Made%20with-Ollama-AA1745?style=for-the-badge)](#)
<img src="https://img.shields.io/badge/Python-3B36E9?style=for-the-badge&logo=gnu&logoColor=white" />

</p>

A tiny, focused example showing how to generate and store vector embeddings with Ollama and Chroma — ideal for retrieval, semantic search, or RAG prototypes, to practice RAG basics to make this (https://github.com/AbdelrahmanYassien11/ReAct-Agent-to-generate-README-Test-Matrix-for-Verification-Flows-using-Ollama) a functional agent integrated with a RAG system.

✨ Clean • Small • Practical

---

## What’s inside
1) embed_text_list.py — simple in-memory example storing a handful of documents in Chroma
2) embed_pdf_text.py — reads a PDF (PyMuPDF/fitz), chunks text, embeds with Ollama, and indexes to Chroma

## Requirements
This repo uses:
- ollama
- chromadb
- PyMuPDF (imported as `fitz`)

Install them with:
```bash
pip install -r requirements.txt
```

## Quickstart

1. Clone the repo
   ```bash
   git clone https://github.com/AbdelrahmanYassien11/Ollama-Vector-Embedding-Example.git
   cd Ollama-Vector-Embedding-Example
   ```

2. Install dependencies
   ```bash
   python -m venv .venv
   source .venv/bin/activate   # macOS / Linux
   .venv\Scripts\activate      # Windows
   pip install -r requirements.txt
   ```

3. Run an example
   ```bash
   # Example: embed the small in-repo text list
   python embed_text_list.py

   # Example: index a PDF (adjust filename in line 55)
   python embed_pdf_text.py
   ```

Notes:
- Ensure your Ollama instance/CLI is running and accessible if the scripts call a local Ollama server.
- Update file paths (e.g., the PDF path in embed_pdf_text.py) as needed.
