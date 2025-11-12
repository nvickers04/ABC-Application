# src/utils/embeddings.py
# Purpose: Embedding generation and semantic search utilities
# Provides text embeddings for memory storage and retrieval
# Supports multiple embedding models and similarity search

import logging
from typing import List, Dict, Any, Optional, Tuple
import numpy as np
from datetime import datetime

# External dependencies
try:
    from sentence_transformers import SentenceTransformer
    EMBEDDINGS_AVAILABLE = True
except ImportError:
    EMBEDDINGS_AVAILABLE = False
    SentenceTransformer = None

try:
    import faiss
    FAISS_AVAILABLE = True
except ImportError:
    FAISS_AVAILABLE = False
    faiss = None

logger = logging.getLogger(__name__)

class EmbeddingManager:
    """
    Manages text embeddings for semantic search and memory retrieval.
    """

    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        """
        Initialize embedding manager.

        Args:
            model_name: Sentence transformer model to use
        """
        if not EMBEDDINGS_AVAILABLE:
            raise ImportError("Sentence transformers not available. Install with: pip install sentence-transformers")

        self.model_name = model_name
        self.model = None
        self.dimension = None

        # Initialize model
        self._load_model()

        # FAISS index for fast similarity search
        self.index = None
        self.index_mapping = []  # Maps FAISS indices to memory keys

    def _load_model(self):
        """Load the sentence transformer model."""
        try:
            self.model = SentenceTransformer(self.model_name)
            # Get embedding dimension
            test_embedding = self.model.encode("test")
            self.dimension = len(test_embedding)
            logger.info(f"Loaded embedding model: {self.model_name} (dimension: {self.dimension})")
        except Exception as e:
            logger.error(f"Failed to load embedding model {self.model_name}: {e}")
            raise

    def encode_text(self, text: str) -> np.ndarray:
        """
        Generate embedding for text.

        Args:
            text: Input text

        Returns:
            numpy array: Text embedding
        """
        try:
            embedding = self.model.encode(text, convert_to_numpy=True)
            return embedding
        except Exception as e:
            logger.error(f"Failed to encode text: {e}")
            # Return zero vector as fallback
            return np.zeros(self.dimension)

    def encode_batch(self, texts: List[str]) -> np.ndarray:
        """
        Generate embeddings for multiple texts.

        Args:
            texts: List of input texts

        Returns:
            numpy array: Text embeddings (shape: n_texts x dimension)
        """
        try:
            embeddings = self.model.encode(texts, convert_to_numpy=True, batch_size=32)
            return embeddings
        except Exception as e:
            logger.error(f"Failed to encode batch: {e}")
            # Return zero vectors as fallback
            return np.zeros((len(texts), self.dimension))

    def compute_similarity(self, embedding1: np.ndarray, embedding2: np.ndarray) -> float:
        """
        Compute cosine similarity between two embeddings.

        Args:
            embedding1: First embedding
            embedding2: Second embedding

        Returns:
            float: Similarity score (0-1)
        """
        try:
            # Cosine similarity
            dot_product = np.dot(embedding1, embedding2)
            norm1 = np.linalg.norm(embedding1)
            norm2 = np.linalg.norm(embedding2)

            if norm1 == 0 or norm2 == 0:
                return 0.0

            similarity = dot_product / (norm1 * norm2)
            return float(similarity)
        except Exception as e:
            logger.error(f"Failed to compute similarity: {e}")
            return 0.0

    def find_similar_texts(self, query_text: str, candidate_texts: List[str],
                          top_k: int = 5) -> List[Tuple[str, float]]:
        """
        Find most similar texts to query.

        Args:
            query_text: Query text
            candidate_texts: List of candidate texts
            top_k: Number of top results to return

        Returns:
            List of (text, similarity_score) tuples
        """
        try:
            # Encode query
            query_embedding = self.encode_text(query_text)

            # Encode candidates
            candidate_embeddings = self.encode_batch(candidate_texts)

            # Compute similarities
            similarities = []
            for i, candidate_embedding in enumerate(candidate_embeddings):
                similarity = self.compute_similarity(query_embedding, candidate_embedding)
                similarities.append((candidate_texts[i], similarity))

            # Sort by similarity (descending)
            similarities.sort(key=lambda x: x[1], reverse=True)

            return similarities[:top_k]
        except Exception as e:
            logger.error(f"Failed to find similar texts: {e}")
            return []

    def build_faiss_index(self, embeddings: np.ndarray, keys: List[str]):
        """
        Build FAISS index for fast similarity search.

        Args:
            embeddings: numpy array of embeddings
            keys: Corresponding keys for each embedding
        """
        if not FAISS_AVAILABLE:
            logger.warning("FAISS not available, skipping index build")
            return

        try:
            # Normalize embeddings for cosine similarity
            norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
            norms[norms == 0] = 1  # Avoid division by zero
            normalized_embeddings = embeddings / norms

            # Create FAISS index
            self.index = faiss.IndexFlatIP(self.dimension)  # Inner product (cosine)
            self.index.add(normalized_embeddings.astype('float32'))
            self.index_mapping = keys

            logger.info(f"Built FAISS index with {len(keys)} vectors")
        except Exception as e:
            logger.error(f"Failed to build FAISS index: {e}")

    def search_faiss_index(self, query_embedding: np.ndarray, top_k: int = 5) -> List[Tuple[str, float]]:
        """
        Search FAISS index for similar embeddings.

        Args:
            query_embedding: Query embedding
            top_k: Number of results to return

        Returns:
            List of (key, similarity_score) tuples
        """
        if not self.index or not FAISS_AVAILABLE:
            logger.warning("FAISS index not available")
            return []

        try:
            # Normalize query embedding
            norm = np.linalg.norm(query_embedding)
            if norm == 0:
                return []
            normalized_query = query_embedding / norm

            # Search index
            scores, indices = self.index.search(
                normalized_query.reshape(1, -1).astype('float32'),
                min(top_k, len(self.index_mapping))
            )

            results = []
            for score, idx in zip(scores[0], indices[0]):
                if idx < len(self.index_mapping):
                    key = self.index_mapping[idx]
                    similarity = float(score)  # FAISS returns cosine similarity
                    results.append((key, similarity))

            return results
        except Exception as e:
            logger.error(f"Failed to search FAISS index: {e}")
            return []

class MemoryEmbeddings:
    """
    Handles embeddings for memory storage and semantic search.
    """

    def __init__(self):
        self.embedding_manager = EmbeddingManager()
        self.memory_embeddings = {}  # key -> embedding mapping
        self.memory_texts = {}       # key -> original text mapping
        self.memory_metadata = {}    # key -> metadata mapping

    def add_memory_embedding(self, key: str, text: str, metadata: Dict[str, Any] = None):
        """
        Add a memory with its embedding.

        Args:
            key: Memory key
            text: Text content
            metadata: Additional metadata
        """
        try:
            # Generate embedding
            embedding = self.embedding_manager.encode_text(text)

            # Store data
            self.memory_embeddings[key] = embedding
            self.memory_texts[key] = text
            self.memory_metadata[key] = metadata or {}

            logger.debug(f"Added embedding for memory: {key}")
        except Exception as e:
            logger.error(f"Failed to add memory embedding for {key}: {e}")

    def remove_memory_embedding(self, key: str):
        """
        Remove a memory embedding.

        Args:
            key: Memory key
        """
        for storage in [self.memory_embeddings, self.memory_texts, self.memory_metadata]:
            storage.pop(key, None)

    def search_similar_memories(self, query: str, top_k: int = 5,
                               threshold: float = 0.0) -> List[Dict[str, Any]]:
        """
        Search for memories similar to query.

        Args:
            query: Search query
            top_k: Number of results to return
            threshold: Minimum similarity threshold

        Returns:
            List of similar memories with metadata
        """
        try:
            if not self.memory_embeddings:
                return []

            # Get query embedding
            query_embedding = self.embedding_manager.encode_text(query)

            # Find similar memories
            similarities = []
            for key, embedding in self.memory_embeddings.items():
                similarity = self.embedding_manager.compute_similarity(query_embedding, embedding)
                if similarity >= threshold:
                    similarities.append((key, similarity))

            # Sort by similarity
            similarities.sort(key=lambda x: x[1], reverse=True)

            # Format results
            results = []
            for key, similarity in similarities[:top_k]:
                results.append({
                    "key": key,
                    "text": self.memory_texts.get(key, ""),
                    "metadata": self.memory_metadata.get(key, {}),
                    "similarity": similarity,
                    "retrieved_at": datetime.now().isoformat()
                })

            return results
        except Exception as e:
            logger.error(f"Failed to search similar memories: {e}")
            return []

    def rebuild_search_index(self):
        """
        Rebuild the FAISS search index with current memories.
        """
        try:
            if not self.memory_embeddings:
                return

            # Prepare data for index
            keys = list(self.memory_embeddings.keys())
            embeddings = np.array([self.memory_embeddings[key] for key in keys])

            # Build FAISS index
            self.embedding_manager.build_faiss_index(embeddings, keys)

            logger.info(f"Rebuilt search index with {len(keys)} memories")
        except Exception as e:
            logger.error(f"Failed to rebuild search index: {e}")

    def get_memory_stats(self) -> Dict[str, Any]:
        """
        Get memory embedding statistics.

        Returns:
            Dict with statistics
        """
        return {
            "total_memories": len(self.memory_embeddings),
            "embedding_dimension": self.embedding_manager.dimension,
            "model_name": self.embedding_manager.model_name,
            "faiss_index_built": self.embedding_manager.index is not None,
            "memory_keys": list(self.memory_embeddings.keys())
        }

# Global instances
_embedding_manager = None
_memory_embeddings = None

def get_embedding_manager() -> EmbeddingManager:
    """Get global embedding manager instance."""
    global _embedding_manager
    if _embedding_manager is None:
        _embedding_manager = EmbeddingManager()
    return _embedding_manager

def get_memory_embeddings() -> MemoryEmbeddings:
    """Get global memory embeddings instance."""
    global _memory_embeddings
    if _memory_embeddings is None:
        _memory_embeddings = MemoryEmbeddings()
    return _memory_embeddings

# Convenience functions
def encode_text(text: str) -> np.ndarray:
    """Encode text to embedding."""
    return get_embedding_manager().encode_text(text)

def search_similar_memories(query: str, top_k: int = 5) -> List[Dict[str, Any]]:
    """Search for similar memories."""
    return get_memory_embeddings().search_similar_memories(query, top_k)