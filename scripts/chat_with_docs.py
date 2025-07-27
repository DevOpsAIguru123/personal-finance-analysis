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
import requests

# Load environment variables from .env file
load_dotenv()
from chromadb.config import Settings
from chromadb.utils import embedding_functions
from openai import OpenAI, AzureOpenAI

# Import our custom embedding models
from embedding_models import create_embedding_model


class InteractiveRAG:
    def __init__(
        self, 
        db_path: str, 
        collection_name: str = "documents"
    ):
        """Initialize the interactive RAG system."""
        self.conversation_history = []
        
        # Get configuration from environment variables
        self.chat_provider = os.getenv("CHAT_PROVIDER", "openai").lower()
        self.embedding_provider = os.getenv("EMBEDDING_PROVIDER", "openai").lower()
        
        # Chat model configuration
        if self.chat_provider == "openai":
            self.chat_model = os.getenv("OPENAI_CHAT_MODEL", "gpt-4")
            self.openai_client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
            if not self.openai_client.api_key:
                raise ValueError("OpenAI API key not found. Set OPENAI_API_KEY environment variable.")
        elif self.chat_provider == "azure_openai":
            self.chat_model = os.getenv("AZURE_OPENAI_CHAT_MODEL", "gpt-4.1-mini")
            endpoint = os.getenv("AZURE_OPENAI_ENDPOINT")
            api_version = os.getenv("AZURE_OPENAI_CHAT_API_VERSION", "2024-12-01-preview")
            if not endpoint:
                raise ValueError("Azure OpenAI endpoint not found. Set AZURE_OPENAI_ENDPOINT environment variable.")
            
            self.openai_client = AzureOpenAI(
                api_key=os.getenv("AZURE_OPENAI_API_KEY"),
                azure_endpoint=endpoint,
                api_version=api_version
            )
            if not self.openai_client.api_key:
                raise ValueError("Azure OpenAI API key not found. Set AZURE_OPENAI_API_KEY environment variable.")
        elif self.chat_provider == "ollama":
            self.chat_model = os.getenv("OLLAMA_CHAT_MODEL", "phi4-mini")
            self.ollama_base_url = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434").rstrip('/')
        else:
            raise ValueError(f"Unsupported chat provider: {self.chat_provider}. Supported: 'openai', 'azure_openai', 'ollama'")
        
        # Initialize ChromaDB
        try:
            self.chroma_client = chromadb.PersistentClient(
                path=db_path,
                settings=Settings(anonymized_telemetry=False)
            )
            
            # Create embedding model for queries
            self.embedding_model = create_embedding_model(provider=self.embedding_provider)
            
            # Get collection without embedding function (we'll provide embeddings manually)
            self.collection = self.chroma_client.get_collection(
                name=collection_name
            )
            print(f"✓ Connected to ChromaDB collection: {collection_name}")
            print(f"🔧 Chat Provider: {self.chat_provider} ({self.chat_model})")
            print(f"🔧 Embedding Provider: {self.embedding_provider} ({self.embedding_model.model_name})")
            
            # Get collection info
            count = self.collection.count()
            print(f"📊 Collection contains {count} documents")
            
        except Exception as e:
            raise ValueError(f"Error connecting to ChromaDB: {str(e)}")
    
    def retrieve_relevant_docs(self, query: str, n_results: int = 5) -> List[Dict[str, Any]]:
        """Retrieve relevant documents from ChromaDB."""
        try:
            # Generate query embedding using our embedding model
            query_embedding = self.embedding_model.generate_embedding(query)
            
            if not query_embedding:
                print("❌ Failed to generate query embedding")
                return []
            
            results = self.collection.query(
                query_embeddings=[query_embedding],
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
        
        # Create prompt - get system prompt from environment or use default
        system_prompt = os.getenv("SYSTEM_PROMPT", """You are a helpful assistant that answers questions based on the provided context documents. 
Use the information from the context to answer questions accurately and comprehensively. 
If the context doesn't contain enough information to answer the question, say so clearly. 
You can reference previous conversation when relevant. Be conversational and helpful.""")
        
        user_prompt = f"""Context Documents:
{context_text}
{history_text}
Current Question: {query}

Please provide a helpful answer based on the context documents above."""
        
        try:
            if self.chat_provider == "openai" or self.chat_provider == "azure_openai":
                response = self.openai_client.chat.completions.create(
                    model=self.chat_model,
                    messages=[
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": user_prompt}
                    ],
                    max_tokens=max_tokens,
                    temperature=temperature
                )
                return response.choices[0].message.content
                
            elif self.chat_provider == "ollama":
                # Ollama chat completion
                payload = {
                    "model": self.chat_model,
                    "messages": [
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": user_prompt}
                    ],
                    "stream": False,
                    "options": {
                        "temperature": temperature,
                        "num_predict": max_tokens
                    }
                }
                
                response = requests.post(
                    f"{self.ollama_base_url}/api/chat",
                    json=payload,
                    headers={"Content-Type": "application/json"},
                    timeout=60
                )
                
                if response.status_code == 200:
                    result = response.json()
                    return result.get("message", {}).get("content", "No response generated")
                else:
                    return f"❌ Ollama API error: {response.status_code} - {response.text}"
        
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
    parser.add_argument("--max-tokens", type=int, default=500, help="Maximum tokens for response")
    parser.add_argument("--n-results", type=int, default=5, help="Number of relevant documents to retrieve")
    
    args = parser.parse_args()
    
    # Check configuration based on providers
    chat_provider = os.getenv("CHAT_PROVIDER", "openai").lower()
    if chat_provider == "openai" and not os.getenv("OPENAI_API_KEY"):
        print("❌ Error: OPENAI_API_KEY environment variable not set for OpenAI chat provider")
        sys.exit(1)
    
    try:
        # Initialize RAG system
        rag = InteractiveRAG(
            db_path=args.db_path,
            collection_name=args.collection_name
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
                    print(f"   Chat Provider: {rag.chat_provider}")
                    print(f"   Chat Model: {rag.chat_model}")
                    print(f"   Embedding Provider: {rag.embedding_provider}")
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