#!/usr/bin/env python3
"""
Embedding Generation Script
Generates embeddings using OpenAI and stores them in ChromaDB
"""

import argparse
import json
import os
import sys
from pathlib import Path
from typing import List, Dict, Any
import uuid

from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

import chromadb
from chromadb.config import Settings
from chromadb.utils import embedding_functions
from openai import OpenAI


class EmbeddingGenerator:
    def __init__(self, model: str = "text-embedding-ada-002", api_key: str = None):
        """Initialize the embedding generator."""
        self.model = model
        self.client = OpenAI(api_key=api_key or os.getenv("OPENAI_API_KEY"))
        
        if not self.client.api_key:
            raise ValueError("OpenAI API key not found. Set OPENAI_API_KEY environment variable.")
    
    def generate_embedding(self, text: str) -> List[float]:
        """Generate embedding for a single text."""
        try:
            response = self.client.embeddings.create(
                model=self.model,
                input=text
            )
            return response.data[0].embedding
        except Exception as e:
            print(f"Error generating embedding: {str(e)}")
            return []
    
    def generate_embeddings_batch(self, texts: List[str], batch_size: int = 100) -> List[List[float]]:
        """Generate embeddings for multiple texts in batches."""
        embeddings = []
        
        for i in range(0, len(texts), batch_size):
            batch = texts[i:i + batch_size]
            print(f"Processing batch {i//batch_size + 1}/{(len(texts) + batch_size - 1)//batch_size}")
            
            try:
                response = self.client.embeddings.create(
                    model=self.model,
                    input=batch
                )
                batch_embeddings = [data.embedding for data in response.data]
                embeddings.extend(batch_embeddings)
            except Exception as e:
                print(f"Error processing batch: {str(e)}")
                # Add empty embeddings for failed batch
                embeddings.extend([[] for _ in batch])
        
        return embeddings


class ChromaDBManager:
    def __init__(self, db_path: str, collection_name: str = "documents"):
        """Initialize ChromaDB manager."""
        self.db_path = db_path
        self.collection_name = collection_name
        
        # Create ChromaDB client
        self.client = chromadb.PersistentClient(
            path=db_path,
            settings=Settings(
                anonymized_telemetry=False
            )
        )
        
        # Create OpenAI embedding function
        openai_ef = embedding_functions.OpenAIEmbeddingFunction(
            api_key=os.getenv("OPENAI_API_KEY"),
            model_name="text-embedding-ada-002"
        )
        
        # Get or create collection with OpenAI embedding function
        try:
            self.collection = self.client.get_collection(
                name=collection_name,
                embedding_function=openai_ef
            )
            print(f"Using existing collection: {collection_name}")
        except:
            self.collection = self.client.create_collection(
                name=collection_name,
                embedding_function=openai_ef
            )
            print(f"Created new collection: {collection_name}")
    
    def add_documents(
        self, 
        documents: List[str], 
        embeddings: List[List[float]], 
        metadatas: List[Dict[str, Any]], 
        ids: List[str] = None
    ) -> None:
        """Add documents with embeddings to ChromaDB."""
        if ids is None:
            ids = [str(uuid.uuid4()) for _ in documents]
        
        # Filter out empty embeddings
        valid_items = [
            (doc, emb, meta, id_) 
            for doc, emb, meta, id_ in zip(documents, embeddings, metadatas, ids) 
            if emb
        ]
        
        if not valid_items:
            print("No valid embeddings to add")
            return
        
        documents, embeddings, metadatas, ids = zip(*valid_items)
        
        try:
            self.collection.add(
                documents=list(documents),
                embeddings=list(embeddings),
                metadatas=list(metadatas),
                ids=list(ids)
            )
            print(f"✓ Added {len(documents)} documents to ChromaDB")
        except Exception as e:
            print(f"Error adding documents to ChromaDB: {str(e)}")
    
    def query(self, query_text: str, n_results: int = 5) -> Dict[str, Any]:
        """Query the collection."""
        try:
            results = self.collection.query(
                query_texts=[query_text],
                n_results=n_results
            )
            return results
        except Exception as e:
            print(f"Error querying ChromaDB: {str(e)}")
            return {}


def load_chunks_from_file(file_path: str) -> List[Dict[str, Any]]:
    """Load chunks from JSON file."""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            return json.load(f)
    except Exception as e:
        print(f"Error loading chunks from {file_path}: {str(e)}")
        return []


def process_chunks_directory(
    input_dir: str, 
    db_path: str, 
    collection_name: str, 
    model: str
) -> None:
    """Process all chunk files in a directory."""
    input_path = Path(input_dir)
    
    if not input_path.exists():
        print(f"Error: Input directory '{input_dir}' does not exist")
        return
    
    # Find all JSON chunk files
    chunk_files = list(input_path.glob("*.json"))
    
    if not chunk_files:
        print(f"No chunk files found in {input_dir}")
        return
    
    print(f"Found {chunk_files} chunk files to process...")
    
    # Initialize embedding generator and ChromaDB
    embedding_gen = EmbeddingGenerator(model=model)
    chroma_db = ChromaDBManager(db_path, collection_name)
    
    total_chunks = 0
    
    for chunk_file in chunk_files:
        print(f"\nProcessing {chunk_file.name}...")
        
        chunks = load_chunks_from_file(str(chunk_file))
        if not chunks:
            continue
        
        # Extract texts and metadata
        documents = [chunk["content"] for chunk in chunks]
        metadatas = [chunk["metadata"] for chunk in chunks]
        
        # Add file source to metadata
        for metadata in metadatas:
            metadata["chunk_file"] = chunk_file.stem
        
        print(f"Generating embeddings for {len(documents)} chunks...")
        embeddings = embedding_gen.generate_embeddings_batch(documents)
        
        # Generate unique IDs
        ids = [f"{chunk_file.stem}_{i}" for i in range(len(documents))]
        
        # Add to ChromaDB
        chroma_db.add_documents(documents, embeddings, metadatas, ids)
        total_chunks += len(documents)
    
    print(f"\n✓ Processing complete: {total_chunks} total chunks processed")


def main():
    parser = argparse.ArgumentParser(description="Generate embeddings and store in ChromaDB")
    parser.add_argument("--input", required=True, help="Input directory containing chunk files")
    parser.add_argument("--db-path", required=True, help="ChromaDB database path")
    parser.add_argument("--collection-name", default="documents", help="ChromaDB collection name")
    parser.add_argument("--model", default="text-embedding-ada-002", help="OpenAI embedding model")
    
    args = parser.parse_args()
    
    # Check if OpenAI API key is set
    if not os.getenv("OPENAI_API_KEY"):
        print("Error: OPENAI_API_KEY environment variable not set")
        sys.exit(1)
    
    process_chunks_directory(
        args.input, 
        args.db_path, 
        args.collection_name, 
        args.model
    )


if __name__ == "__main__":
    main()