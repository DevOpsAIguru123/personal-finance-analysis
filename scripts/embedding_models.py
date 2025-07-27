#!/usr/bin/env python3
"""
Embedding Models Module
Supports both OpenAI and Ollama embedding models
"""

import os
import requests
from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional
from openai import OpenAI


class EmbeddingModel(ABC):
    """Abstract base class for embedding models."""
    
    @abstractmethod
    def generate_embedding(self, text: str) -> List[float]:
        """Generate embedding for a single text."""
        pass
    
    @abstractmethod
    def generate_embeddings_batch(self, texts: List[str], batch_size: int = 100) -> List[List[float]]:
        """Generate embeddings for multiple texts in batches."""
        pass
    
    @property
    @abstractmethod
    def model_name(self) -> str:
        """Return the model name."""
        pass


class OpenAIEmbeddingModel(EmbeddingModel):
    """OpenAI embedding model implementation."""
    
    def __init__(self, model: str = "text-embedding-ada-002", api_key: str = None):
        """Initialize the OpenAI embedding model."""
        self.model = model
        self.client = OpenAI(api_key=api_key or os.getenv("OPENAI_API_KEY"))
        
        if not self.client.api_key:
            raise ValueError("OpenAI API key not found. Set OPENAI_API_KEY environment variable.")
    
    @property
    def model_name(self) -> str:
        return f"openai:{self.model}"
    
    def generate_embedding(self, text: str) -> List[float]:
        """Generate embedding for a single text."""
        try:
            response = self.client.embeddings.create(
                model=self.model,
                input=text
            )
            return response.data[0].embedding
        except Exception as e:
            print(f"Error generating OpenAI embedding: {str(e)}")
            return []
    
    def generate_embeddings_batch(self, texts: List[str], batch_size: int = 100) -> List[List[float]]:
        """Generate embeddings for multiple texts in batches."""
        embeddings = []
        
        for i in range(0, len(texts), batch_size):
            batch = texts[i:i + batch_size]
            print(f"Processing OpenAI batch {i//batch_size + 1}/{(len(texts) + batch_size - 1)//batch_size}")
            
            try:
                response = self.client.embeddings.create(
                    model=self.model,
                    input=batch
                )
                batch_embeddings = [data.embedding for data in response.data]
                embeddings.extend(batch_embeddings)
            except Exception as e:
                print(f"Error processing OpenAI batch: {str(e)}")
                # Add empty embeddings for failed batch
                embeddings.extend([[] for _ in batch])
        
        return embeddings


class OllamaEmbeddingModel(EmbeddingModel):
    """Ollama embedding model implementation."""
    
    def __init__(self, model: str = "jina/jina-embeddings-v2-base-en", base_url: str = "http://localhost:11434"):
        """Initialize the Ollama embedding model."""
        self.model = model
        self.base_url = base_url.rstrip('/')
        self.embeddings_url = f"{self.base_url}/api/embeddings"
        
        # Test connection
        self._test_connection()
    
    def _test_connection(self) -> None:
        """Test connection to Ollama server."""
        try:
            response = requests.get(f"{self.base_url}/api/tags", timeout=5)
            if response.status_code != 200:
                raise ConnectionError(f"Ollama server responded with status {response.status_code}")
        except requests.exceptions.RequestException as e:
            raise ConnectionError(f"Cannot connect to Ollama server at {self.base_url}: {str(e)}")
    
    @property
    def model_name(self) -> str:
        return f"ollama:{self.model}"
    
    def generate_embedding(self, text: str) -> List[float]:
        """Generate embedding for a single text."""
        try:
            payload = {
                "model": self.model,
                "prompt": text
            }
            
            response = requests.post(
                self.embeddings_url,
                json=payload,
                headers={"Content-Type": "application/json"},
                timeout=30
            )
            
            if response.status_code == 200:
                result = response.json()
                return result.get("embedding", [])
            else:
                print(f"Ollama API error: {response.status_code} - {response.text}")
                return []
                
        except Exception as e:
            print(f"Error generating Ollama embedding: {str(e)}")
            return []
    
    def generate_embeddings_batch(self, texts: List[str], batch_size: int = 10) -> List[List[float]]:
        """Generate embeddings for multiple texts in batches."""
        embeddings = []
        
        # Ollama typically handles smaller batches better
        for i in range(0, len(texts), batch_size):
            batch = texts[i:i + batch_size]
            print(f"Processing Ollama batch {i//batch_size + 1}/{(len(texts) + batch_size - 1)//batch_size}")
            
            for text in batch:
                embedding = self.generate_embedding(text)
                embeddings.append(embedding)
        
        return embeddings


def create_embedding_model(
    provider: str = None,
    model: str = None,
    api_key: str = None,
    base_url: str = None
) -> EmbeddingModel:
    """Factory function to create embedding models."""
    
    # Get defaults from environment if not provided
    if provider is None:
        provider = os.getenv("EMBEDDING_PROVIDER", "openai")
    
    if provider.lower() == "openai":
        if model is None:
            model = os.getenv("OPENAI_EMBEDDING_MODEL", "text-embedding-ada-002")
        return OpenAIEmbeddingModel(
            model=model,
            api_key=api_key
        )
    
    elif provider.lower() == "ollama":
        if model is None:
            model = os.getenv("OLLAMA_EMBEDDING_MODEL", "jina/jina-embeddings-v2-base-en")
        if base_url is None:
            base_url = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
        return OllamaEmbeddingModel(
            model=model,
            base_url=base_url
        )
    
    else:
        raise ValueError(f"Unsupported embedding provider: {provider}. Supported: 'openai', 'ollama'")


def get_available_ollama_models(base_url: str = None) -> List[str]:
    """Get list of available Ollama models."""
    if base_url is None:
        base_url = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
    
    try:
        response = requests.get(f"{base_url.rstrip('/')}/api/tags", timeout=5)
        if response.status_code == 200:
            data = response.json()
            return [model["name"] for model in data.get("models", [])]
        else:
            print(f"Error fetching Ollama models: {response.status_code}")
            return []
    except Exception as e:
        print(f"Error connecting to Ollama: {str(e)}")
        return []


def list_embedding_models() -> Dict[str, List[str]]:
    """List available embedding models for each provider."""
    models = {
        "openai": [
            "text-embedding-ada-002",
            "text-embedding-3-small",
            "text-embedding-3-large"
        ],
        "ollama": get_available_ollama_models()
    }
    
    return models