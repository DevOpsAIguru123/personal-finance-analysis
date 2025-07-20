#!/usr/bin/env python3
"""
PDF to Markdown Conversion Script
Converts PDF files from docs/ folder to markdown format in docs_md/ folder
"""

import argparse
import os
import sys
from pathlib import Path
from typing import List

from docling.document_converter import DocumentConverter


def convert_pdf_to_markdown(input_path: str, output_path: str) -> bool:
    """Convert a single PDF file to markdown format."""
    try:
        converter = DocumentConverter()
        result = converter.convert(input_path)
        
        # Export to markdown
        markdown_content = result.document.export_to_markdown()
        
        # Write to output file
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(markdown_content)
        
        print(f"✓ Converted: {input_path} -> {output_path}")
        return True
    
    except Exception as e:
        print(f"✗ Error converting {input_path}: {str(e)}")
        return False


def process_directory(input_dir: str, output_dir: str) -> None:
    """Process all PDF files in a directory."""
    input_path = Path(input_dir)
    output_path = Path(output_dir)
    
    # Create output directory if it doesn't exist
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Find all PDF files
    pdf_files = list(input_path.glob("*.pdf"))
    
    if not pdf_files:
        print(f"No PDF files found in {input_dir}")
        return
    
    print(f"Found {len(pdf_files)} PDF files to convert...")
    
    success_count = 0
    for pdf_file in pdf_files:
        # Generate output filename
        output_file = output_path / f"{pdf_file.stem}.md"
        
        if convert_pdf_to_markdown(str(pdf_file), str(output_file)):
            success_count += 1
    
    print(f"\nConversion complete: {success_count}/{len(pdf_files)} files converted successfully")


def main():
    parser = argparse.ArgumentParser(description="Convert PDF files to Markdown format")
    parser.add_argument("--input", required=True, help="Input PDF file or directory containing PDFs")
    parser.add_argument("--output", required=True, help="Output markdown file or directory")
    
    args = parser.parse_args()
    
    input_path = Path(args.input)
    output_path = Path(args.output)
    
    if not input_path.exists():
        print(f"Error: Input path '{args.input}' does not exist")
        sys.exit(1)
    
    if input_path.is_file():
        # Single file conversion
        if not input_path.suffix.lower() == '.pdf':
            print("Error: Input file must be a PDF")
            sys.exit(1)
        
        # Ensure output has .md extension
        if output_path.suffix.lower() != '.md':
            output_path = output_path.with_suffix('.md')
        
        # Create output directory if needed
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        success = convert_pdf_to_markdown(str(input_path), str(output_path))
        sys.exit(0 if success else 1)
    
    elif input_path.is_dir():
        # Directory conversion
        process_directory(str(input_path), str(output_path))
    
    else:
        print(f"Error: '{args.input}' is neither a file nor a directory")
        sys.exit(1)


if __name__ == "__main__":
    main()