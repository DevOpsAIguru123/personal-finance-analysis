#!/usr/bin/env python3
"""
Query Embeddings Script
Query ChromaDB embeddings and generate text using OpenAI
"""

import argparse
import os
import sys
from typing import List, Dict, Any

from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

import chromadb
from chromadb.config import Settings
from chromadb.utils import embedding_functions
from openai import OpenAI

# Import our custom embedding models
from embedding_models import create_embedding_model


class RAGSystem:
    def __init__(
        self, 
        db_path: str, 
        collection_name: str = "documents",
        openai_model: str = "gpt-4",
        api_key: str = None,
        embedding_provider: str = None
    ):
        """Initialize the RAG system."""
        self.openai_model = openai_model
        self.client = OpenAI(api_key=api_key or os.getenv("OPENAI_API_KEY"))
        
        if not self.client.api_key:
            raise ValueError("OpenAI API key not found. Set OPENAI_API_KEY environment variable.")
        
        # Get embedding provider from environment if not specified
        embedding_provider = embedding_provider or os.getenv("EMBEDDING_PROVIDER", "openai")
        
        # Initialize ChromaDB
        try:
            self.chroma_client = chromadb.PersistentClient(
                path=db_path,
                settings=Settings(anonymized_telemetry=False)
            )
            
            # Create embedding function based on provider
            if embedding_provider.lower() == "openai":
                model_name = os.getenv("OPENAI_EMBEDDING_MODEL", "text-embedding-ada-002")
                embedding_function = embedding_functions.OpenAIEmbeddingFunction(
                    api_key=api_key or os.getenv("OPENAI_API_KEY"),
                    model_name=model_name
                )
            else:
                # For Ollama and other providers, create our custom embedding model
                self.custom_embedding_model = create_embedding_model(
                    provider=embedding_provider
                )
                embedding_function = None
            
            self.collection = self.chroma_client.get_collection(
                name=collection_name,
                embedding_function=embedding_function
            )
            print(f"✓ Connected to ChromaDB collection: {collection_name}")
        except Exception as e:
            raise ValueError(f"Error connecting to ChromaDB: {str(e)}")
    
    def retrieve_relevant_docs(self, query: str, n_results: int = 5) -> List[Dict[str, Any]]:
        """Retrieve relevant documents from ChromaDB."""
        try:
            # Check if we need to provide query embeddings manually
            if hasattr(self, 'custom_embedding_model'):
                query_embedding = self.custom_embedding_model.generate_embedding(query)
                results = self.collection.query(
                    query_embeddings=[query_embedding],
                    n_results=n_results
                )
            else:
                results = self.collection.query(
                    query_texts=[query],
                    n_results=n_results
                )
            
            # Format results
            documents = []
            if results['documents'] and results['documents'][0]:
                for i, (doc, metadata, distance) in enumerate(zip(
                    results['documents'][0],
                    results['metadatas'][0] if results['metadatas'] else [{}] * len(results['documents'][0]),
                    results['distances'][0] if results['distances'] else [0] * len(results['documents'][0])
                )):
                    documents.append({
                        'content': doc,
                        'metadata': metadata,
                        'similarity_score': 1 - distance,  # Convert distance to similarity
                        'rank': i + 1
                    })
            
            return documents
        
        except Exception as e:
            print(f"Error retrieving documents: {str(e)}")
            return []
    
    def generate_response(
        self, 
        query: str, 
        context_docs: List[Dict[str, Any]], 
        max_tokens: int = 500,
        temperature: float = 0.7
    ) -> str:
        """Generate response using OpenAI with retrieved context."""
        # Prepare context
        context_text = "\n\n".join([
            f"Document {doc['rank']} (Score: {doc['similarity_score']:.3f}):\n{doc['content']}"
            for doc in context_docs
        ])
        
        # Create prompt
        system_prompt = """You are a helpful assistant that answers questions based on the provided context documents. 
Use only the information from the context to answer questions. If the context doesn't contain enough information 
to answer the question, say so clearly. Always cite which document(s) you're referencing in your answer."""
        
        user_prompt = f"""Context Documents:
{context_text}

Question: {query}

Please provide a comprehensive answer based on the context documents above."""
        
        try:
            response = self.client.chat.completions.create(
                model=self.openai_model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                max_tokens=max_tokens,
                temperature=temperature
            )
            
            return response.choices[0].message.content
        
        except Exception as e:
            return f"Error generating response: {str(e)}"
    
    def query(
        self, 
        question: str, 
        n_results: int = 5, 
        max_tokens: int = 500,
        show_sources: bool = True
    ) -> Dict[str, Any]:
        """Complete RAG query pipeline."""
        print(f"🔍 Searching for relevant documents...")
        
        # Retrieve relevant documents
        relevant_docs = self.retrieve_relevant_docs(question, n_results)
        
        if not relevant_docs:
            return {
                "question": question,
                "answer": "No relevant documents found.",
                "sources": []
            }
        
        print(f"📄 Found {len(relevant_docs)} relevant documents")
        
        # Generate response
        print(f"🤖 Generating response...")
        answer = self.generate_response(question, relevant_docs, max_tokens)
        
        result = {
            "question": question,
            "answer": answer,
            "sources": relevant_docs if show_sources else []
        }
        
        return result


def print_results(result: Dict[str, Any], show_sources: bool = True) -> None:
    """Print the query results in a formatted way."""
    print("\n" + "="*60)
    print(f"QUESTION: {result['question']}")
    print("="*60)
    print(f"ANSWER:\n{result['answer']}")
    
    if show_sources and result['sources']:
        print("\n" + "-"*60)
        print("SOURCES:")
        print("-"*60)
        
        for doc in result['sources']:
            print(f"\n📄 Document {doc['rank']} (Similarity: {doc['similarity_score']:.3f})")
            if 'source' in doc['metadata']:
                print(f"   Source: {doc['metadata']['source']}")
            if 'chunk_id' in doc['metadata']:
                print(f"   Chunk: {doc['metadata']['chunk_id']}")
            print(f"   Content: {doc['content'][:200]}{'...' if len(doc['content']) > 200 else ''}")


def main():
    parser = argparse.ArgumentParser(description="Query embeddings and generate responses")
    parser.add_argument("--query", required=True, help="Question to ask")
    parser.add_argument("--db-path", required=True, help="ChromaDB database path")
    parser.add_argument("--collection-name", default="documents", help="ChromaDB collection name")
    parser.add_argument("--model", default="gpt-4", help="OpenAI model for text generation")
    parser.add_argument("--max-tokens", type=int, default=500, help="Maximum tokens for response")
    parser.add_argument("--n-results", type=int, default=5, help="Number of relevant documents to retrieve")
    parser.add_argument("--no-sources", action="store_true", help="Don't show source documents")
    parser.add_argument("--embedding-provider", choices=["openai", "ollama"], help="Embedding provider (openai or ollama)")
    
    args = parser.parse_args()
    
    # Check if OpenAI API key is set
    if not os.getenv("OPENAI_API_KEY"):
        print("Error: OPENAI_API_KEY environment variable not set")
        sys.exit(1)
    
    try:
        # Initialize RAG system
        rag = RAGSystem(
            db_path=args.db_path,
            collection_name=args.collection_name,
            openai_model=args.model,
            embedding_provider=args.embedding_provider
        )
        
        # Perform query
        result = rag.query(
            question=args.query,
            n_results=args.n_results,
            max_tokens=args.max_tokens,
            show_sources=not args.no_sources
        )
        
        # Print results
        print_results(result, show_sources=not args.no_sources)
        
    except Exception as e:
        print(f"Error: {str(e)}")
        sys.exit(1)


if __name__ == "__main__":
    main()