#!/usr/bin/env python3
"""
Full Pipeline Runner
Runs the complete PDF processing pipeline from PDFs to interactive chat
"""

import argparse
import os
import subprocess
import sys
from pathlib import Path


def run_command(command: list, description: str) -> bool:
    """Run a command and return success status."""
    print(f"\n🔄 {description}...")
    print(f"Command: {' '.join(command)}")
    
    try:
        result = subprocess.run(command, check=True, capture_output=True, text=True)
        if result.stdout:
            print(result.stdout)
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ Error: {e}")
        if e.stderr:
            print(f"Error output: {e.stderr}")
        return False


def check_prerequisites() -> bool:
    """Check if required environment variables and directories exist."""
    print("🔍 Checking prerequisites...")
    
    # Check OpenAI API key
    if not os.getenv("OPENAI_API_KEY"):
        print("❌ OPENAI_API_KEY environment variable not set")
        return False
    
    print("✓ OpenAI API key found")
    return True


def main():
    parser = argparse.ArgumentParser(description="Run the full PDF processing pipeline")
    parser.add_argument("--input-dir", default="docs", help="Directory containing PDF files")
    parser.add_argument("--markdown-dir", default="docs_md", help="Directory for markdown files")
    parser.add_argument("--chunks-dir", default="chunks", help="Directory for chunk files")
    parser.add_argument("--db-path", default="chroma_db", help="ChromaDB database path")
    parser.add_argument("--collection-name", default="documents", help="ChromaDB collection name")
    parser.add_argument("--chunk-size", type=int, default=1000, help="Chunk size")
    parser.add_argument("--chunk-overlap", type=int, default=200, help="Chunk overlap")
    parser.add_argument("--embedding-model", default="text-embedding-ada-002", help="OpenAI embedding model")
    parser.add_argument("--chat-model", default="gpt-4", help="OpenAI chat model")
    parser.add_argument("--skip-pdf", action="store_true", help="Skip PDF conversion step")
    parser.add_argument("--skip-chunk", action="store_true", help="Skip chunking step")
    parser.add_argument("--skip-embed", action="store_true", help="Skip embedding generation step")
    parser.add_argument("--start-chat", action="store_true", help="Start interactive chat after processing")
    
    args = parser.parse_args()
    
    # Check prerequisites
    if not check_prerequisites():
        sys.exit(1)
    
    # Create directories
    for dir_path in [args.markdown_dir, args.chunks_dir, args.db_path]:
        Path(dir_path).mkdir(parents=True, exist_ok=True)
    
    scripts_dir = Path(__file__).parent
    
    # Step 1: Convert PDFs to Markdown
    if not args.skip_pdf:
        pdf_script = scripts_dir / "pdf_to_markdown.py"
        if not run_command([
            "python", str(pdf_script),
            "--input", args.input_dir,
            "--output", args.markdown_dir
        ], f"Converting PDFs from {args.input_dir} to markdown"):
            print("❌ PDF conversion failed")
            sys.exit(1)
    else:
        print("⏭️  Skipping PDF conversion")
    
    # Step 2: Chunk markdown files
    if not args.skip_chunk:
        chunk_script = scripts_dir / "chunk_documents.py"
        if not run_command([
            "python", str(chunk_script),
            "--input", args.markdown_dir,
            "--output", args.chunks_dir,
            "--chunk-size", str(args.chunk_size),
            "--overlap", str(args.chunk_overlap),
            "--chunk-method", "recursive"
        ], f"Chunking markdown files from {args.markdown_dir}"):
            print("❌ Chunking failed")
            sys.exit(1)
    else:
        print("⏭️  Skipping chunking")
    
    # Step 3: Generate embeddings and store in ChromaDB
    if not args.skip_embed:
        embed_script = scripts_dir / "generate_embeddings.py"
        if not run_command([
            "python", str(embed_script),
            "--input", args.chunks_dir,
            "--db-path", args.db_path,
            "--collection-name", args.collection_name,
            "--model", args.embedding_model
        ], f"Generating embeddings and storing in ChromaDB"):
            print("❌ Embedding generation failed")
            sys.exit(1)
    else:
        print("⏭️  Skipping embedding generation")
    
    print("\n🎉 Pipeline completed successfully!")
    print(f"📊 Documents are now available in ChromaDB at: {args.db_path}")
    print(f"🤖 You can now query your documents using:")
    print(f"   python scripts/query_embeddings.py --query 'your question' --db-path {args.db_path}")
    print(f"   python scripts/chat_with_docs.py --db-path {args.db_path}")
    
    # Optionally start interactive chat
    if args.start_chat:
        print("\n🚀 Starting interactive chat...")
        chat_script = scripts_dir / "chat_with_docs.py"
        subprocess.run([
            "python", str(chat_script),
            "--db-path", args.db_path,
            "--collection-name", args.collection_name,
            "--model", args.chat_model
        ])


if __name__ == "__main__":
    main()