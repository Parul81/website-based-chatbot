"""
Embeddings and Vector Store Module
Creates embeddings and stores them in FAISS for fast similarity search.
"""

import os
import pickle
from typing import List, Dict
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np


class EmbeddingStore:
    """Manages embeddings and FAISS vector store."""
    
    def __init__(self, model_name: str = "all-MiniLM-L6-v2", index_dir: str = "data/faiss_index"):
        """
        Initialize the embedding store.
        
        Args:
            model_name: Name of the sentence transformer model
            index_dir: Directory to save/load FAISS index
        """
        self.model_name = model_name
        self.index_dir = index_dir
        self.model = None
        self.index = None
        self.chunks = []  # Store original chunks for retrieval
        self.dimension = 384  # Dimension for all-MiniLM-L6-v2
        
        # Create directory if it doesn't exist
        os.makedirs(index_dir, exist_ok=True)
        
        # Load model
        self._load_model()
    
    def _load_model(self):
        """Load the sentence transformer model."""
        print(f"Loading embedding model: {self.model_name}")
        self.model = SentenceTransformer(self.model_name)
        print("Model loaded successfully!")
    
    def create_embeddings(self, chunks: List[Dict]) -> np.ndarray:
        """
        Create embeddings for text chunks.
        
        Args:
            chunks: List of dictionaries with 'text' key
            
        Returns:
            numpy array of embeddings
        """
        texts = [chunk['text'] for chunk in chunks]
        embeddings = self.model.encode(texts, show_progress_bar=True)
        return embeddings
    
    def build_index(self, chunks: List[Dict], embeddings: np.ndarray):
        """
        Build FAISS index from embeddings.
        
        Args:
            chunks: List of chunk dictionaries
            embeddings: numpy array of embeddings
        """
        # Normalize embeddings for cosine similarity
        faiss.normalize_L2(embeddings)
        
        # Create FAISS index
        self.dimension = embeddings.shape[1]
        self.index = faiss.IndexFlatIP(self.dimension)  # Inner product for cosine similarity
        
        # Add embeddings to index
        self.index.add(embeddings.astype('float32'))
        
        # Store chunks for retrieval
        self.chunks = chunks
        
        print(f"Index built with {len(chunks)} chunks")
    
    def save_index(self):
        """Save FAISS index and chunks to disk."""
        if self.index is None:
            print("No index to save!")
            return
        
        # Save FAISS index
        index_path = os.path.join(self.index_dir, "index.faiss")
        faiss.write_index(self.index, index_path)
        
        # Save chunks
        chunks_path = os.path.join(self.index_dir, "chunks.pkl")
        with open(chunks_path, 'wb') as f:
            pickle.dump(self.chunks, f)
        
        # Save metadata
        metadata = {
            'model_name': self.model_name,
            'dimension': self.dimension,
            'num_chunks': len(self.chunks)
        }
        metadata_path = os.path.join(self.index_dir, "metadata.pkl")
        with open(metadata_path, 'wb') as f:
            pickle.dump(metadata, f)
        
        print(f"Index saved to {self.index_dir}")
    
    def load_index(self) -> bool:
        """
        Load FAISS index and chunks from disk.
        
        Returns:
            True if loaded successfully, False otherwise
        """
        index_path = os.path.join(self.index_dir, "index.faiss")
        chunks_path = os.path.join(self.index_dir, "chunks.pkl")
        
        if not os.path.exists(index_path) or not os.path.exists(chunks_path):
            print("No saved index found!")
            return False
        
        try:
            # Load FAISS index
            self.index = faiss.read_index(index_path)
            
            # Load chunks
            with open(chunks_path, 'rb') as f:
                self.chunks = pickle.load(f)
            
            # Load metadata
            metadata_path = os.path.join(self.index_dir, "metadata.pkl")
            if os.path.exists(metadata_path):
                with open(metadata_path, 'rb') as f:
                    metadata = pickle.load(f)
                    self.dimension = metadata.get('dimension', self.dimension)
            
            print(f"Index loaded with {len(self.chunks)} chunks")
            return True
        except Exception as e:
            print(f"Error loading index: {e}")
            return False
    
    def search(self, query: str, k: int = 3) -> List[Dict]:
        """
        Search for similar chunks.
        
        Args:
            query: Search query text
            k: Number of results to return
            
        Returns:
            List of chunk dictionaries with similarity scores
        """
        if self.index is None or len(self.chunks) == 0:
            return []
        
        # Create query embedding
        query_embedding = self.model.encode([query])
        faiss.normalize_L2(query_embedding)
        
        # Search
        k = min(k, len(self.chunks))
        distances, indices = self.index.search(query_embedding.astype('float32'), k)
        
        # Return results with metadata
        results = []
        for i, idx in enumerate(indices[0]):
            if idx < len(self.chunks):
                chunk = self.chunks[idx].copy()
                chunk['similarity_score'] = float(distances[0][i])
                results.append(chunk)
        
        return results

