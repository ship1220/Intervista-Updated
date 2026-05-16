# rag/vector_store.py
# Vector store for embeddings and semantic search

import os
import json
from typing import List, Dict, Any, Optional, Tuple
import numpy as np
from pathlib import Path
import faiss
from config.settings import Settings
from utils.logger import Logger

logger = Logger(__name__)


class TextEmbedder:
    """Wrapper for text embeddings (using sentence-transformers via FAISS)."""
    
    def __init__(self, model_name: str = "sentence-transformers/all-MiniLM-L6-v2"):
        """Initialize embedder."""
        try:
            from sentence_transformers import SentenceTransformer
            self.model = SentenceTransformer(model_name)
            # Use get_embedding_dimension() (replaces deprecated get_sentence_embedding_dimension())
            self.embedding_dim = self.model.get_embedding_dimension()
            logger.info(f"TextEmbedder initialized: {model_name} (dim={self.embedding_dim})")
        except ImportError:
            logger.warning("sentence-transformers not installed. Using mock embedder.")
            self.model = None
            self.embedding_dim = 384
    
    def embed(self, text: str) -> np.ndarray:
        """Convert text to embedding."""
        if self.model is None:
            # Mock embedding for when sentence-transformers not available
            return np.random.randn(self.embedding_dim).astype(np.float32)
        
        embedding = self.model.encode(text, convert_to_numpy=True)
        return embedding.astype(np.float32)
    
    def embed_batch(self, texts: List[str]) -> np.ndarray:
        """Convert multiple texts to embeddings."""
        if self.model is None:
            return np.random.randn(len(texts), self.embedding_dim).astype(np.float32)
        
        embeddings = self.model.encode(texts, convert_to_numpy=True)
        return embeddings.astype(np.float32)


class Document:
    """Represents a document in the vector store."""
    
    def __init__(self, doc_id: str, content: str, metadata: Dict[str, Any] = None):
        self.doc_id = doc_id
        self.content = content
        self.metadata = metadata or {}
        self.embedding = None
    
    def to_dict(self) -> Dict:
        """Serialize document."""
        return {
            "doc_id": self.doc_id,
            "content": self.content,
            "metadata": self.metadata
        }


class VectorStore:
    """
    Vector database backed by FAISS.
    
    Stores documents with embeddings for semantic search.
    """
    
    def __init__(self, embedding_dim: int = 384, index_type: str = "faiss"):
        self.embedding_dim = embedding_dim
        self.index_type = index_type
        self.embedder = TextEmbedder()
        self.documents: Dict[str, Document] = {}
        self.index = None
        self.doc_id_to_index: Dict[str, int] = {}  # Map doc_id to FAISS index
        self._init_index()
        logger.info(f"VectorStore initialized (dim={embedding_dim}, type={index_type})")
    
    def _init_index(self):
        """Initialize FAISS index."""
        self.index = faiss.IndexFlatL2(self.embedding_dim)
        logger.debug("FAISS index created")
    
    def add_document(self, doc_id: str, content: str, metadata: Dict[str, Any] = None) -> str:
        """
        Add document to vector store.
        
        Args:
            doc_id: Unique document ID
            content: Document text content
            metadata: Optional metadata dict
        
        Returns:
            Document ID
        """
        
        # Create document
        doc = Document(doc_id, content, metadata)
        
        # Generate embedding
        embedding = self.embedder.embed(content)
        doc.embedding = embedding
        
        # Add to FAISS index
        index_pos = self.index.ntotal
        self.index.add(np.array([embedding]))
        self.doc_id_to_index[doc_id] = index_pos
        
        # Store document
        self.documents[doc_id] = doc
        
        logger.debug(f"Document added: {doc_id}")
        return doc_id
    
    def add_documents_batch(self, documents: List[Tuple[str, str, Dict]] = None, documents_list: List[Dict] = None) -> List[str]:
        """
        Add multiple documents efficiently.
        
        Args:
            documents: List of (doc_id, content, metadata) tuples
            documents_list: List of {"doc_id": ..., "content": ..., "metadata": ...}
        
        Returns:
            List of added doc_ids
        """
        
        if documents_list is None:
            documents_list = []
        
        if documents:
            documents_list = [(d[0], d[1], d[2] if len(d) > 2 else {}) for d in documents]
        
        if not documents_list:
            return []
        
        doc_ids = []
        embeddings = []
        
        # Prepare embeddings
        texts = [doc[1] for doc in documents_list]
        batch_embeddings = self.embedder.embed_batch(texts)
        
        # Add documents
        for i, (doc_id, content, metadata) in enumerate(documents_list):
            doc = Document(doc_id, content, metadata)
            doc.embedding = batch_embeddings[i]
            
            self.doc_id_to_index[doc_id] = self.index.ntotal + i
            self.documents[doc_id] = doc
            doc_ids.append(doc_id)
            embeddings.append(batch_embeddings[i])
        
        # Add all embeddings to index at once
        self.index.add(np.array(embeddings))
        
        logger.info(f"Batch added: {len(doc_ids)} documents")
        return doc_ids
    
    def search(self, query: str, k: int = 5) -> List[Tuple[str, float]]:
        """
        Semantic search in vector store.
        
        Args:
            query: Search query text
            k: Number of results to return
        
        Returns:
            List of (doc_id, similarity_score) tuples
        """
        
        if self.index.ntotal == 0:
            logger.warning("Vector store is empty")
            return []
        
        # Embed query
        query_embedding = self.embedder.embed(query)
        
        # Search FAISS index (returns distances)
        distances, indices = self.index.search(np.array([query_embedding]), k)
        
        results = []
        for distance, idx in zip(distances[0], indices[0]):
            if idx == -1:  # Invalid index
                continue
            
            # Find doc_id for this index position
            doc_id = self._get_doc_id_by_index(idx)
            if doc_id:
                # Convert L2 distance to similarity score (0-100)
                similarity = 100 / (1 + distance)
                results.append((doc_id, similarity))
        
        logger.debug(f"Search returned {len(results)} results")
        return results
    
    def _get_doc_id_by_index(self, index_pos: int) -> Optional[str]:
        """Get doc_id by FAISS index position."""
        for doc_id, idx_pos in self.doc_id_to_index.items():
            if idx_pos == index_pos:
                return doc_id
        return None
    
    def get_document(self, doc_id: str) -> Optional[Document]:
        """Retrieve document by ID."""
        return self.documents.get(doc_id)
    
    def delete_document(self, doc_id: str) -> bool:
        """Remove document (marks as deleted)."""
        if doc_id in self.documents:
            del self.documents[doc_id]
            logger.debug(f"Document deleted: {doc_id}")
            return True
        
        return False
    
    def size(self) -> int:
        """Total documents in store."""
        return len(self.documents)
    
    def stats(self) -> Dict[str, Any]:
        """Get vector store statistics."""
        return {
            "total_documents": self.size(),
            "index_type": self.index_type,
            "embedding_dimension": self.embedding_dim,
            "faiss_index_size": self.index.ntotal
        }
    
    def save(self, path: str):
        """Save vector store to disk."""
        data = {
            "documents": {doc_id: doc.to_dict() for doc_id, doc in self.documents.items()},
            "index_type": self.index_type,
            "embedding_dim": self.embedding_dim
        }
        
        with open(path, 'w') as f:
            json.dump(data, f)
        
        logger.info(f"VectorStore saved to {path}")
    
    def load(self, path: str):
        """Load vector store from disk."""
        with open(path, 'r') as f:
            data = json.load(f)
        
        for doc_id, doc_data in data.get("documents", {}).items():
            self.add_document(
                doc_id=doc_id,
                content=doc_data["content"],
                metadata=doc_data.get("metadata", {})
            )
        
        logger.info(f"VectorStore loaded from {path}")
    
    def clear(self):
        """Clear all documents."""
        self.documents.clear()
        self.doc_id_to_index.clear()
        self._init_index()
        logger.info("VectorStore cleared")
