# Indian Legal Query Assistant ⚖️

An AI-powered legal assistant for Indian constitutional law, criminal procedure (CrPC), and penal code (IPC) queries, combining retrieval-augmented generation (RAG) with semantic search.

## Features ✨
- Answers questions about:
  - Indian Constitution articles
  - Criminal Procedure Code (CrPC)
  - Indian Penal Code (IPC) sections
- Provides:
  - Exact legal provisions
  - Plain-language explanations
  - Relevant case law references
- Supports:
  - Direct article lookup ("Article 21")
  - Conceptual queries ("right to privacy")
  - Comparative analysis ("Difference between IPC 299 and 300")

## Tech Stack 🛠️
| Component               | Technology |
|-------------------------|------------|
| NLP                     | Hugging Face Transformers, Sentence-Transformers |
| Language Model          | LLaMA-2-7B |
| Text Embeddings         | paraphrase-MiniLM-L12-v2 |
| Vector Database         | FAISS |
| PDF Processing          | PyMuPDF (fitz) |
| Backend Framework       | Python 3.10+ |
| GPU Acceleration        | CUDA 11.7 |
