#!/usr/bin/env python3
"""
Document Chunking Script
Chunks markdown files using LangChain text splitters
"""

import argparse
import json
import os
import sys
from pathlib import Path
from typing import List, Dict, Any

from langchain.text_splitter import RecursiveCharacterTextSplitter, MarkdownHeaderTextSplitter


def load_markdown_file(file_path: str) -> str:
    """Load markdown content from file."""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            return f.read()
    except Exception as e:
        print(f"Error loading {file_path}: {str(e)}")
        return ""


def chunk_text_recursive(text: str, chunk_size: int = 1000, overlap: int = 200) -> List[str]:
    """Chunk text using RecursiveCharacterTextSplitter."""
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=overlap,
        length_function=len,
        separators=["\n\n", "\n", " ", ""]
    )
    return splitter.split_text(text)


def chunk_text_markdown_headers(text: str, chunk_size: int = 1000, overlap: int = 200) -> List[Dict[str, Any]]:
    """Chunk text using MarkdownHeaderTextSplitter to preserve structure."""
    headers_to_split_on = [
        ("#", "Header 1"),
        ("##", "Header 2"),
        ("###", "Header 3"),
        ("####", "Header 4"),
    ]
    
    markdown_splitter = MarkdownHeaderTextSplitter(
        headers_to_split_on=headers_to_split_on
    )
    
    md_header_splits = markdown_splitter.split_text(text)
    
    # Further split large chunks
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=overlap
    )
    
    chunks = []
    for i, split in enumerate(md_header_splits):
        content = split.page_content
        metadata = split.metadata
        
        # Split large content further
        if len(content) > chunk_size:
            sub_chunks = text_splitter.split_text(content)
            for j, sub_chunk in enumerate(sub_chunks):
                chunks.append({
                    "content": sub_chunk,
                    "metadata": {
                        **metadata,
                        "chunk_id": f"{i}_{j}",
                        "sub_chunk": j
                    }
                })
        else:
            chunks.append({
                "content": content,
                "metadata": {
                    **metadata,
                    "chunk_id": str(i)
                }
            })
    
    return chunks


def save_chunks(chunks: List[Dict[str, Any]], output_dir: str, filename: str) -> None:
    """Save chunks to JSON file."""
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    output_file = output_path / f"{filename}.json"
    
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(chunks, f, indent=2, ensure_ascii=False)
    
    print(f"✓ Saved {len(chunks)} chunks to {output_file}")


def process_markdown_file(
    file_path: str, 
    output_dir: str, 
    chunk_size: int, 
    overlap: int, 
    method: str
) -> None:
    """Process a single markdown file."""
    content = load_markdown_file(file_path)
    if not content:
        return
    
    filename = Path(file_path).stem
    
    if method == "recursive":
        chunks_text = chunk_text_recursive(content, chunk_size, overlap)
        chunks = [
            {
                "content": chunk,
                "metadata": {
                    "source": filename,
                    "chunk_id": str(i),
                    "method": "recursive"
                }
            }
            for i, chunk in enumerate(chunks_text)
        ]
    
    elif method == "markdown":
        chunks = chunk_text_markdown_headers(content, chunk_size, overlap)
        for chunk in chunks:
            chunk["metadata"]["source"] = filename
            chunk["metadata"]["method"] = "markdown"
    
    else:
        print(f"Unknown chunking method: {method}")
        return
    
    save_chunks(chunks, output_dir, filename)


def process_directory(
    input_dir: str, 
    output_dir: str, 
    chunk_size: int, 
    overlap: int, 
    method: str
) -> None:
    """Process all markdown files in a directory."""
    input_path = Path(input_dir)
    
    if not input_path.exists():
        print(f"Error: Input directory '{input_dir}' does not exist")
        return
    
    # Find all markdown files
    md_files = list(input_path.glob("*.md"))
    
    if not md_files:
        print(f"No markdown files found in {input_dir}")
        return
    
    print(f"Found {len(md_files)} markdown files to chunk...")
    print(f"Chunking method: {method}")
    print(f"Chunk size: {chunk_size}, Overlap: {overlap}")
    
    for md_file in md_files:
        process_markdown_file(str(md_file), output_dir, chunk_size, overlap, method)
    
    print(f"\nChunking complete: {len(md_files)} files processed")


def main():
    parser = argparse.ArgumentParser(description="Chunk markdown documents using LangChain")
    parser.add_argument("--input", required=True, help="Input markdown file or directory")
    parser.add_argument("--output", required=True, help="Output directory for chunks")
    parser.add_argument("--chunk-size", type=int, default=1000, help="Chunk size in characters (default: 1000)")
    parser.add_argument("--overlap", type=int, default=200, help="Overlap between chunks (default: 200)")
    parser.add_argument("--chunk-method", choices=["recursive", "markdown"], default="recursive", 
                       help="Chunking method (default: recursive)")
    
    args = parser.parse_args()
    
    input_path = Path(args.input)
    
    if not input_path.exists():
        print(f"Error: Input path '{args.input}' does not exist")
        sys.exit(1)
    
    if input_path.is_file():
        # Single file processing
        if not input_path.suffix.lower() == '.md':
            print("Error: Input file must be a markdown file (.md)")
            sys.exit(1)
        
        process_markdown_file(
            str(input_path), 
            args.output, 
            args.chunk_size, 
            args.overlap, 
            args.chunk_method
        )
    
    elif input_path.is_dir():
        # Directory processing
        process_directory(
            str(input_path), 
            args.output, 
            args.chunk_size, 
            args.overlap, 
            args.chunk_method
        )
    
    else:
        print(f"Error: '{args.input}' is neither a file nor a directory")
        sys.exit(1)


if __name__ == "__main__":
    main()