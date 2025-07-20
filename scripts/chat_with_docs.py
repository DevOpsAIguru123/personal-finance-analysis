#!/usr/bin/env python3
"""
Interactive Chat with Documents
Interactive chat interface for querying documents using RAG
"""

import argparse
import os
import sys
from typing import List, Dict, Any

from dotenv import load_dotenv
import chromadb

# Load environment variables from .env file
load_dotenv()
from chromadb.config import Settings
from chromadb.utils import embedding_functions
from openai import OpenAI


class InteractiveRAG:
    def __init__(
        self, 
        db_path: str, 
        collection_name: str = "documents",
        openai_model: str = "gpt-4",
        api_key: str = None
    ):
        """Initialize the interactive RAG system."""
        self.openai_model = openai_model
        self.client = OpenAI(api_key=api_key or os.getenv("OPENAI_API_KEY"))
        self.conversation_history = []
        
        if not self.client.api_key:
            raise ValueError("OpenAI API key not found. Set OPENAI_API_KEY environment variable.")
        
        # Initialize ChromaDB with OpenAI embedding function
        try:
            self.chroma_client = chromadb.PersistentClient(
                path=db_path,
                settings=Settings(anonymized_telemetry=False)
            )
            
            # Create OpenAI embedding function
            openai_ef = embedding_functions.OpenAIEmbeddingFunction(
                api_key=api_key or os.getenv("OPENAI_API_KEY"),
                model_name="text-embedding-ada-002"
            )
            
            self.collection = self.chroma_client.get_collection(
                name=collection_name,
                embedding_function=openai_ef
            )
            print(f"✓ Connected to ChromaDB collection: {collection_name}")
            
            # Get collection info
            count = self.collection.count()
            print(f"📊 Collection contains {count} documents")
            
        except Exception as e:
            raise ValueError(f"Error connecting to ChromaDB: {str(e)}")
    
    def retrieve_relevant_docs(self, query: str, n_results: int = 5) -> List[Dict[str, Any]]:
        """Retrieve relevant documents from ChromaDB."""
        try:
            results = self.collection.query(
                query_texts=[query],
                n_results=n_results
            )
            
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
                        'similarity_score': 1 - distance,
                        'rank': i + 1
                    })
            
            return documents
        
        except Exception as e:
            print(f"❌ Error retrieving documents: {str(e)}")
            return []
    
    def generate_response(
        self, 
        query: str, 
        context_docs: List[Dict[str, Any]], 
        max_tokens: int = 500,
        temperature: float = 0.7
    ) -> str:
        """Generate response using OpenAI with retrieved context and conversation history."""
        # Prepare context
        context_text = "\n\n".join([
            f"Document {doc['rank']} (Score: {doc['similarity_score']:.3f}):\n{doc['content']}"
            for doc in context_docs
        ])
        
        # Prepare conversation history
        history_text = ""
        if self.conversation_history:
            history_text = "\n\nPrevious conversation:\n"
            for entry in self.conversation_history[-3:]:  # Last 3 exchanges
                history_text += f"Q: {entry['question']}\nA: {entry['answer']}\n\n"
        
        # Create prompt
        system_prompt = """You are a helpful assistant that answers questions based on the provided context documents. 
Use the information from the context to answer questions accurately and comprehensively. 
If the context doesn't contain enough information to answer the question, say so clearly. 
You can reference previous conversation when relevant. Be conversational and helpful."""
        
        user_prompt = f"""Context Documents:
{context_text}
{history_text}
Current Question: {query}

Please provide a helpful answer based on the context documents above."""
        
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
            return f"❌ Error generating response: {str(e)}"
    
    def query(self, question: str, n_results: int = 5, max_tokens: int = 500) -> Dict[str, Any]:
        """Process a single query."""
        # Retrieve relevant documents
        relevant_docs = self.retrieve_relevant_docs(question, n_results)
        
        if not relevant_docs:
            answer = "I couldn't find any relevant documents to answer your question."
        else:
            # Generate response
            answer = self.generate_response(question, relevant_docs, max_tokens)
        
        # Store in conversation history
        conversation_entry = {
            "question": question,
            "answer": answer,
            "sources": relevant_docs
        }
        self.conversation_history.append(conversation_entry)
        
        return conversation_entry
    
    def print_response(self, result: Dict[str, Any], show_sources: bool = False) -> None:
        """Print the response in a formatted way."""
        print(f"\n🤖 Assistant: {result['answer']}")
        
        if show_sources and result['sources']:
            print(f"\n📚 Sources ({len(result['sources'])} documents):")
            for doc in result['sources'][:3]:  # Show top 3 sources
                source_info = ""
                if 'source' in doc['metadata']:
                    source_info = f" from {doc['metadata']['source']}"
                print(f"   • Document {doc['rank']} (Score: {doc['similarity_score']:.2f}){source_info}")


def print_welcome():
    """Print welcome message."""
    print("\n" + "="*60)
    print("🚀 Welcome to Document Chat!")
    print("="*60)
    print("Ask questions about your documents. Type 'help' for commands.")
    print("Commands:")
    print("  help     - Show this help message")
    print("  sources  - Toggle showing sources")
    print("  history  - Show conversation history")
    print("  clear    - Clear conversation history")
    print("  stats    - Show collection statistics")
    print("  exit/quit - Exit the chat")
    print("-"*60)


def print_help():
    """Print help message."""
    print("\n📖 Help:")
    print("• Ask questions about your documents naturally")
    print("• Use 'sources' to toggle source document display")
    print("• Use 'history' to see previous questions and answers")
    print("• Use 'clear' to start a fresh conversation")
    print("• Use 'stats' to see database information")


def main():
    parser = argparse.ArgumentParser(description="Interactive chat with your documents")
    parser.add_argument("--db-path", required=True, help="ChromaDB database path")
    parser.add_argument("--collection-name", default="documents", help="ChromaDB collection name")
    parser.add_argument("--model", default="gpt-4", help="OpenAI model for text generation")
    parser.add_argument("--max-tokens", type=int, default=500, help="Maximum tokens for response")
    parser.add_argument("--n-results", type=int, default=5, help="Number of relevant documents to retrieve")
    
    args = parser.parse_args()
    
    # Check if OpenAI API key is set
    if not os.getenv("OPENAI_API_KEY"):
        print("❌ Error: OPENAI_API_KEY environment variable not set")
        sys.exit(1)
    
    try:
        # Initialize RAG system
        rag = InteractiveRAG(
            db_path=args.db_path,
            collection_name=args.collection_name,
            openai_model=args.model
        )
        
        print_welcome()
        
        show_sources = False
        
        while True:
            try:
                # Get user input
                question = input("\n💭 You: ").strip()
                
                # Handle commands
                if question.lower() in ['exit', 'quit', 'q']:
                    print("\n👋 Goodbye!")
                    break
                
                elif question.lower() == 'help':
                    print_help()
                    continue
                
                elif question.lower() == 'sources':
                    show_sources = not show_sources
                    print(f"\n🔧 Source display: {'ON' if show_sources else 'OFF'}")
                    continue
                
                elif question.lower() == 'history':
                    if not rag.conversation_history:
                        print("\n📝 No conversation history yet.")
                    else:
                        print(f"\n📝 Conversation History ({len(rag.conversation_history)} exchanges):")
                        for i, entry in enumerate(rag.conversation_history, 1):
                            print(f"\n{i}. Q: {entry['question']}")
                            print(f"   A: {entry['answer'][:100]}{'...' if len(entry['answer']) > 100 else ''}")
                    continue
                
                elif question.lower() == 'clear':
                    rag.conversation_history = []
                    print("\n🧹 Conversation history cleared.")
                    continue
                
                elif question.lower() == 'stats':
                    count = rag.collection.count()
                    print(f"\n📊 Database Statistics:")
                    print(f"   Collection: {args.collection_name}")
                    print(f"   Documents: {count}")
                    print(f"   Model: {args.model}")
                    continue
                
                elif not question:
                    continue
                
                # Process the question
                print("\n🔍 Searching...")
                result = rag.query(
                    question=question,
                    n_results=args.n_results,
                    max_tokens=args.max_tokens
                )
                
                # Print the response
                rag.print_response(result, show_sources=show_sources)
                
            except KeyboardInterrupt:
                print("\n\n👋 Goodbye!")
                break
            except Exception as e:
                print(f"\n❌ Error: {str(e)}")
                continue
    
    except Exception as e:
        print(f"❌ Error: {str(e)}")
        sys.exit(1)


if __name__ == "__main__":
    main()