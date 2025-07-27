# Ollama Setup Guide

This guide covers setting up Ollama for the Docling RAG system, including the specific models used in this project.

## Models Used in This Project

This project is configured to use the following Ollama models:

### Embedding Model
- **jina/jina-embeddings-v2-base-en** - High-quality multilingual embedding model for document vectorization

### Chat Model  
- **qwen3:0.6b** - Lightweight and fast chat model optimized for document Q&A

## Installation

### Install Ollama

**macOS:**
```bash
brew install ollama
```

**Linux:**
```bash
curl -fsSL https://ollama.ai/install.sh | sh
```

**Windows:**
Download from [ollama.ai](https://ollama.ai/download)

### Install Required Models

```bash
# Start Ollama service
ollama serve

# Install embedding model
ollama pull jina/jina-embeddings-v2-base-en

# Install chat model
ollama pull qwen3:0.6b

# Verify installation
ollama list
```

## Configuration

Update your `.env` file:

```env
# Ollama Configuration
EMBEDDING_PROVIDER=ollama
OLLAMA_EMBEDDING_MODEL=jina/jina-embeddings-v2-base-en
OLLAMA_BASE_URL=http://localhost:11434

CHAT_PROVIDER=ollama
OLLAMA_CHAT_MODEL=qwen3:0.6b
```

## Usage

```bash
# Generate embeddings with Ollama
python scripts/generate_embeddings.py --input chunks/ --db-path chroma_db/ --provider ollama

# Chat with documents using Ollama
python scripts/chat_with_docs.py --db-path chroma_db/
```

## Alternative Models

### For Lower Resource Systems:
- Chat: `llama3.2:1b` (1GB model)
- Embedding: `nomic-embed-text`

### For Higher Performance:
- Chat: `llama3.2:3b` or `qwen2.5:7b`
- Embedding: `mxbai-embed-large`

## Troubleshooting

**Service Issues:**
```bash
# Restart Ollama
pkill ollama
ollama serve
```

**Memory Issues:**
- Use smaller models like `qwen3:0.6b`
- Reduce chunk size in processing
- Monitor with `ollama ps`

For detailed setup and troubleshooting, see the [official Ollama documentation](https://ollama.ai/docs).