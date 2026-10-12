# Semantic Retrieval Engine for Research Papers

A lightweight Retrieval-Augmented Generation (RAG) pipeline to semantically query dense academic literature.

## Key Features
- **PDF Extraction**: Extracts raw text from academic papers using `pypdf`.
- **Sliding-Window Chunking**: 800-character segments with 100-character overlap for context retention.
- **Dense Embeddings**: Encodes context chunks into 384-dimensional dense vectors using `all-MiniLM-L6-v2`.
- **Information Retrieval**: Performs Cosine Similarity retrieval to surface high-relevance context for queries.
- **Frontend**: Interactive UI built with Streamlit and cached resource loading.
