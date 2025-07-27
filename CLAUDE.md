# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Development Commands

### Environment Setup
```bash
# Install in development mode
pip install -e .

# Install with all optional dependencies
pip install -e ".[all]"

# Install additional dependencies for PDF processing and embeddings
pip install langchain langchain-openai openai chromadb
```

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

### PDF Processing Workflow

#### Convert PDFs to Markdown
```bash
# Convert all PDFs from docs/ folder to markdown in docs_md/
python scripts/pdf_to_markdown.py --input docs/ --output docs_md/

# Convert single PDF file
python scripts/pdf_to_markdown.py --input docs/document.pdf --output docs_md/document.md
```

#### Chunk Markdown Files with LangChain
```bash
# Chunk all markdown files from docs_md/ using LangChain
python scripts/chunk_documents.py --input docs_md/ --output chunks/ --chunk-size 1000 --overlap 200

# Custom chunking parameters
python scripts/chunk_documents.py --input docs_md/ --output chunks/ --chunk-size 500 --overlap 100 --chunk-method recursive
```

#### Generate Embeddings and Store in ChromaDB
```bash
# Generate embeddings using OpenAI (default)
python scripts/generate_embeddings.py --input chunks/ --db-path chroma_db/ --provider openai --model text-embedding-ada-002

# Generate embeddings using Ollama
python scripts/generate_embeddings.py --input chunks/ --db-path chroma_db/ --provider ollama --model jina/jina-embeddings-v2-base-en

# Use environment variables (set EMBEDDING_PROVIDER in .env)
python scripts/generate_embeddings.py --input chunks/ --db-path chroma_db/

# Initialize ChromaDB collection
python scripts/init_chroma.py --db-path chroma_db/ --collection-name documents
```

#### Generate Text from Embeddings
```bash
# Query embeddings and generate text with OpenAI embeddings
python scripts/query_embeddings.py --query "your question here" --db-path chroma_db/ --model gpt-4 --embedding-provider openai

# Query embeddings and generate text with Ollama embeddings
python scripts/query_embeddings.py --query "your question here" --db-path chroma_db/ --model gpt-4 --embedding-provider ollama

# Interactive chat mode
python scripts/chat_with_docs.py --db-path chroma_db/ --model gpt-4
```

## Architecture Overview

Docling is a Python library for document parsing and conversion. The architecture follows a modular design:

### Core Components

1. **DocumentConverter** (`docling/document_converter.py`) - Main entry point for document conversion operations. Orchestrates the conversion pipeline and manages different backends.

2. **Backends** (`docling/backend/`) - Pluggable conversion engines:
   - Each backend handles specific document formats or conversion methods
   - Backends implement a common interface for consistent integration
   - Examples include PDF, DOCX, and other format-specific processors

3. **Data Models** (`docling/datamodel/`) - Structured representations:
   - `base.py` contains core data structures and base classes
   - Defines document structure, metadata, and content representations
   - Provides serialization/deserialization capabilities

4. **Pipeline Architecture** - The conversion process follows a pipeline pattern:
   - Input validation and format detection
   - Backend selection based on document type
   - Content extraction and processing
   - Output generation in target format

### PDF Processing Pipeline

1. **PDF to Markdown Conversion**:
   - Uses Docling's PDF backend to extract text and structure
   - Converts to markdown format preserving headers, tables, and formatting
   - Saves processed files to `docs_md/` directory

2. **Document Chunking**:
   - LangChain RecursiveCharacterTextSplitter for intelligent chunking
   - Maintains semantic coherence across chunk boundaries
   - Configurable chunk size and overlap parameters

3. **Embedding Generation**:
   - OpenAI text-embedding-ada-002 model for embeddings
   - ChromaDB for persistent vector storage
   - Batch processing for efficient API usage

4. **Text Generation**:
   - Retrieval-Augmented Generation (RAG) using ChromaDB
   - OpenAI GPT models for response generation
   - Context-aware responses based on document content

### Key Design Patterns

- **Plugin Architecture**: Backends are dynamically loaded and registered
- **Strategy Pattern**: Different conversion strategies based on document type
- **Builder Pattern**: Complex document structures are built incrementally
- **Factory Pattern**: Backend instances are created through factory methods
- **RAG Pattern**: Retrieval-Augmented Generation for document-based Q&A

### Extension Points

- Add new backends by implementing the backend interface
- Extend data models for new document types or metadata
- Customize conversion pipelines through configuration
- Implement custom chunking strategies in LangChain
- Extend ChromaDB collections for different document types

### Configuration

#### Environment Configuration
```bash
# OpenAI API Configuration
OPENAI_API_KEY=your_api_key

# Embedding Model Configuration
EMBEDDING_PROVIDER=openai  # or 'ollama'
OPENAI_EMBEDDING_MODEL=text-embedding-ada-002
OLLAMA_EMBEDDING_MODEL=jina/jina-embeddings-v2-base-en
OLLAMA_BASE_URL=http://localhost:11434

# ChromaDB Configuration
CHROMA_DB_PATH=./chroma_db
CHROMA_COLLECTION_NAME=documents
```

### Testing Structure

Tests are organized by component in the `tests/` directory, mirroring the source structure. Integration tests validate end-to-end conversion workflows including PDF processing, chunking, embedding generation, and ChromaDB storage.