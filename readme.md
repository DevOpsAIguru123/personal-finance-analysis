# Docling RAG - Document Processing and AI Chat System

A comprehensive document processing pipeline built with Docling that converts PDFs to markdown, chunks documents with LangChain, generates embeddings, stores them in ChromaDB, and provides an AI-powered chat interface for document Q&A.

---

## Features

- **PDF to Markdown Conversion:** Extract and convert PDF documents to structured markdown format.
- **Intelligent Document Chunking:** Use LangChain's RecursiveCharacterTextSplitter for semantic chunking.
- **Vector Embeddings:** Generate embeddings using OpenAI or Ollama models.
- **Vector Storage:** Persistent storage with ChromaDB for efficient similarity search.
- **AI Chat Interface:** Interactive chat with your documents using Retrieval-Augmented Generation (RAG).
- **Multiple Embedding Providers:** Support for OpenAI and Ollama embedding models.

---

## Architecture

The system follows a modular pipeline architecture:

1. **Document Conversion:** PDF files converted to markdown using Docling.
2. **Text Chunking:** Markdown content split into semantic chunks with LangChain.
3. **Embedding Generation:** Text chunks converted to vector embeddings.
4. **Vector Storage:** Embeddings stored in ChromaDB for efficient retrieval.
5. **RAG Chat:** AI-powered Q&A using context from retrieved document chunks.

---

## Installation

### Prerequisites

- Python 3.9+ (< 3.13)
- OpenAI API key (for OpenAI embeddings/chat) or Ollama (for local models)

### Environment Setup

```bash
# Clone the repository
git clone <repository-url>
cd docling

# Install the package in development mode
pip install -e .

# Install additional dependencies for PDF processing and embeddings
pip install langchain langchain-openai openai chromadb
```

Set up environment variables by creating a `.env` file:

```env
# Embedding Model Configuration
EMBEDDING_PROVIDER=openai  # or 'ollama'
OPENAI_EMBEDDING_MODEL=text-embedding-ada-002
OLLAMA_EMBEDDING_MODEL=jina/jina-embeddings-v2-base-en
OLLAMA_BASE_URL=http://localhost:11434

# Chat Provider Configuration
# Options: 'openai' or 'ollama'


# OpenAI Chat Configuration (when CHAT_PROVIDER=openai)
CHAT_PROVIDER=ollama # ollama or openai

# Openai Chat Configuration (when CHAT_PROVIDER=openai)
OPENAI_API_KEY=
OPENAI_CHAT_MODEL=gpt-4o
MAX_TOKENS=500

# Ollama Chat Configuration (when CHAT_PROVIDER=ollama)
OLLAMA_CHAT_MODEL=qwen3:0.6b
OLLAMA_BASE_URL=http://localhost:1143

# ChromaDB Configuration
CHROMA_DB_PATH=./chroma_db
CHROMA_COLLECTION_NAME=documents
```

---

## Usage

### 1. Convert PDFs to Markdown

```bash
# Convert all PDFs from docs/ folder to markdown in docs_md/
python scripts/pdf_to_markdown.py --input docs/ --output docs_md/

# Convert single PDF file
python scripts/pdf_to_markdown.py --input docs/document.pdf --output docs_md/document.md
```

### 2. Chunk Documents

```bash
# Chunk all markdown files with default settings (1000 chars, 200 overlap)
python scripts/chunk_documents.py --input docs_md/ --output chunks/ --chunk-size 1000 --overlap 200

# Custom chunking parameters
python scripts/chunk_documents.py --input docs_md/ --output chunks/ --chunk-size 500 --overlap 100 --chunk-method recursive
```

### 3. Generate and Store Embeddings

```bash
# Initialize ChromaDB collection (run once)
python scripts/init_chroma.py --db-path chroma_db/ --collection-name documents

# Generate embeddings using OpenAI (default)
python scripts/generate_embeddings.py --input chunks/ --db-path chroma_db/ --provider openai --model text-embedding-ada-002

# Generate embeddings using Ollama
python scripts/generate_embeddings.py --input chunks/ --db-path chroma_db/ --provider ollama --model jina/jina-embeddings-v2-base-en

# Use environment variables (set EMBEDDING_PROVIDER in .env)
python scripts/generate_embeddings.py --input chunks/ --db-path chroma_db/
```

### 4. Chat with Your Documents

```bash
# Interactive chat mode with OpenAI
python scripts/chat_with_docs.py --db-path chroma_db/ --model gpt-4

# Query embeddings directly
python scripts/query_embeddings.py --query "your question here" --db-path chroma_db/ --model gpt-4 --embedding-provider openai
```

---

## Complete Workflow Example

```bash
# 1. Convert PDFs to markdown
python scripts/pdf_to_markdown.py --input docs/ --output docs_md/

# 2. Chunk the documents
python scripts/chunk_documents.py --input docs_md/ --output chunks/ --chunk-size 1000 --overlap 200

# 3. Initialize ChromaDB
python scripts/init_chroma.py --db-path chroma_db/ --collection-name documents

# 4. Generate embeddings
python scripts/generate_embeddings.py --input chunks/ --db-path chroma_db/ --provider openai

# 5. Start chatting with your documents
python scripts/chat_with_docs.py --db-path chroma_db/ --model gpt-4
```

---

## Configuration Options

### Embedding Providers

**OpenAI (Default):**
- Model: `text-embedding-ada-002`
- Requires: `OPENAI_API_KEY` environment variable
- High quality embeddings with API costs

**Ollama (Local):**
- Model: `jina/jina-embeddings-v2-base-en` (configurable)
- Requires: Local Ollama installation
- Free local processing, requires more setup

### Chunking Strategies

- **Recursive Character Text Splitter:** Smart splitting that preserves semantic boundaries
- **Chunk Size:** Default 1000 characters (adjustable)
- **Overlap:** Default 200 characters for context preservation

### ChromaDB Configuration

- **Persistent Storage:** Vector embeddings stored locally in `chroma_db/`
- **Collection Name:** Default "documents" (configurable)
- **Metadata:** Includes source file and chunk information

---

## Project Structure

```
docling/
  scripts/                    # Main processing scripts
    pdf_to_markdown.py        # PDF conversion
    chunk_documents.py        # Document chunking
    generate_embeddings.py    # Embedding generation
    query_embeddings.py       # Direct querying
    chat_with_docs.py         # Interactive chat
    init_chroma.py            # ChromaDB initialization
    embedding_models.py       # Model configuration
  docling/                    # Core Docling library
  data/                       # Input documents
  docs_md/                    # Converted markdown files
  chunks/                     # Chunked text files
  chroma_db/                  # ChromaDB vector storage
  .env                        # Environment configuration
  pyproject.toml              # Project dependencies
```

---

## Development

### Testing

```bash
# Run all tests
pytest

# Run specific test file
pytest tests/test_document_converter.py

# Run tests with coverage
pytest --cov=docling
```

### Code Quality

```bash
# Format code
black docling/
isort docling/

# Lint code
flake8 docling/
mypy docling/
```

---

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests for new functionality
5. Run the test suite and linting
6. Submit a pull request

---

## License

This project is licensed under the MIT License - see the LICENSE file for details.

---

## Troubleshooting

### Common Issues

- **ChromaDB Errors:** Ensure ChromaDB is properly initialized with `init_chroma.py`
- **OpenAI API Errors:** Verify your API key is set in the `.env` file
- **Ollama Connection Issues:** Check that Ollama is running locally on the specified port
- **Memory Issues:** For large documents, reduce chunk size or process in smaller batches

### Support

For issues and questions:
1. Check the troubleshooting section above
2. Review the script help messages with `--help`
3. Open an issue on the project repository

---

## Dependencies

- **chromadb:** Vector database for embedding storage
- **langchain:** Document processing and chunking
- **langchain-openai:** OpenAI integration for LangChain
- **openai:** OpenAI API client
- **python-dotenv:** Environment variable management
- **filetype:** File type detection

---

## Version

Current version: **0.1.0**

---

## Requirements

- Python: >= 3.9, < 3.13
- See `pyproject.toml` for complete dependency list
