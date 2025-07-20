# PDF Processing and RAG System Usage Guide

This guide shows how to use the implemented PDF processing pipeline with ChromaDB and OpenAI.

## Prerequisites

1. Install required dependencies:
```bash
pip install langchain langchain-openai openai chromadb
```

2. Set your OpenAI API key:
```bash
export OPENAI_API_KEY="your-api-key-here"
```

## Quick Start

### Option 1: Run Full Pipeline
```bash
# Process all PDFs in docs/ folder and start interactive chat
python scripts/run_full_pipeline.py --start-chat
```

### Option 2: Step by Step

1. **Convert PDFs to Markdown**
```bash
# Convert all PDFs from docs/ to docs_md/
python scripts/pdf_to_markdown.py --input docs/ --output docs_md/
```

2. **Chunk Markdown Files**
```bash
# Chunk files with 1000 character chunks and 200 overlap
python scripts/chunk_documents.py --input docs_md/ --output chunks/ --chunk-size 1000 --overlap 200
```

3. **Generate Embeddings and Store in ChromaDB**
```bash
# Generate embeddings and store in ChromaDB
python scripts/generate_embeddings.py --input chunks/ --db-path chroma_db/
```

4. **Query Your Documents**
```bash
# Ask a single question
python scripts/query_embeddings.py --query "What is the main topic?" --db-path chroma_db/

# Start interactive chat
python scripts/chat_with_docs.py --db-path chroma_db/
```

## Directory Structure

After running the pipeline, you'll have:

```
docling/
├── docs/           # Put your PDF files here
├── docs_md/        # Generated markdown files
├── chunks/         # Chunked documents (JSON files)
├── chroma_db/      # ChromaDB vector database
└── scripts/        # Processing scripts
```

## Script Details

### pdf_to_markdown.py
Converts PDF files to markdown using Docling.

**Options:**
- `--input`: PDF file or directory
- `--output`: Output file or directory

### chunk_documents.py
Chunks markdown files using LangChain text splitters.

**Options:**
- `--input`: Markdown file or directory
- `--output`: Output directory for chunks
- `--chunk-size`: Chunk size in characters (default: 1000)
- `--overlap`: Overlap between chunks (default: 200)
- `--chunk-method`: "recursive" or "markdown" (default: recursive)

### generate_embeddings.py
Generates embeddings using OpenAI and stores in ChromaDB.

**Options:**
- `--input`: Directory containing chunk files
- `--db-path`: ChromaDB database path
- `--collection-name`: Collection name (default: documents)
- `--model`: OpenAI embedding model (default: text-embedding-ada-002)

### query_embeddings.py
Query the ChromaDB and generate responses.

**Options:**
- `--query`: Question to ask
- `--db-path`: ChromaDB database path
- `--model`: OpenAI model for text generation (default: gpt-4)
- `--max-tokens`: Maximum response tokens (default: 500)
- `--n-results`: Number of documents to retrieve (default: 5)

### chat_with_docs.py
Interactive chat interface with your documents.

**Commands in chat:**
- `help`: Show help
- `sources`: Toggle source display
- `history`: Show conversation history
- `clear`: Clear history
- `stats`: Show database stats
- `exit`/`quit`: Exit chat

## Examples

### Processing Finance Documents
```bash
# Put your financial PDFs in docs/
python scripts/run_full_pipeline.py --chunk-size 800 --start-chat
```

### Custom Configuration
```bash
# Use smaller chunks for better precision
python scripts/chunk_documents.py --input docs_md/ --output chunks/ --chunk-size 500 --overlap 50

# Use different models
python scripts/generate_embeddings.py --input chunks/ --db-path chroma_db/ --model text-embedding-ada-002
python scripts/chat_with_docs.py --db-path chroma_db/ --model gpt-3.5-turbo
```

### Query Examples
```bash
# Financial analysis
python scripts/query_embeddings.py --query "What are the key financial metrics mentioned?" --db-path chroma_db/

# Summary questions
python scripts/query_embeddings.py --query "Summarize the main findings" --db-path chroma_db/
```

## Troubleshooting

1. **OpenAI API Key Error**: Make sure `OPENAI_API_KEY` is set in your environment
2. **ChromaDB Permission Error**: Ensure the `chroma_db/` directory is writable
3. **PDF Conversion Issues**: Check that your PDF files are readable and not password-protected
4. **Memory Issues**: Use smaller chunk sizes or process fewer files at once

## Advanced Usage

### Batch Processing Multiple Collections
```bash
# Process different document types into separate collections
python scripts/generate_embeddings.py --input financial_chunks/ --db-path chroma_db/ --collection-name finance
python scripts/generate_embeddings.py --input legal_chunks/ --db-path chroma_db/ --collection-name legal

# Query specific collections
python scripts/chat_with_docs.py --db-path chroma_db/ --collection-name finance
```

### Custom Chunking Strategies
```bash
# Use markdown-aware chunking for structured documents
python scripts/chunk_documents.py --input docs_md/ --output chunks/ --chunk-method markdown

# Smaller chunks for Q&A
python scripts/chunk_documents.py --input docs_md/ --output chunks/ --chunk-size 300 --overlap 50
```