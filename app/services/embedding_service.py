"""
Embedding service for generating vector representations of text.
"""
import asyncio
import logging
from typing import List, Union
import openai
import numpy as np
from config import settings

logger = logging.getLogger(__name__)


class EmbeddingService:
    """Service for generating embeddings using OpenAI API."""
    
    def __init__(self):
        """Initialize the embedding service."""
        openai.api_key = settings.openai_api_key
        self.model = settings.embedding_model
        self.client = openai.OpenAI(api_key=settings.openai_api_key)
    
    async def get_embedding(self, text: str) -> List[float]:
        """
        Get embedding for a single text.
        
        Args:
            text: Input text to embed
            
        Returns:
            List of floats representing the embedding vector
        """
        try:
            # Clean and prepare text
            cleaned_text = self._preprocess_text(text)
            
            # Get embedding from OpenAI
            response = await asyncio.to_thread(
                self.client.embeddings.create,
                input=cleaned_text,
                model=self.model
            )
            
            embedding = response.data[0].embedding
            logger.info(f"Generated embedding for text of length {len(text)}")
            return embedding
            
        except Exception as e:
            logger.error(f"Error generating embedding: {e}")
            raise
    
    async def get_embeddings_batch(self, texts: List[str]) -> List[List[float]]:
        """
        Get embeddings for multiple texts in batch.
        
        Args:
            texts: List of input texts to embed
            
        Returns:
            List of embedding vectors
        """
        try:
            # Clean and prepare texts
            cleaned_texts = [self._preprocess_text(text) for text in texts]
            
            # Get embeddings from OpenAI
            response = await asyncio.to_thread(
                self.client.embeddings.create,
                input=cleaned_texts,
                model=self.model
            )
            
            embeddings = [item.embedding for item in response.data]
            logger.info(f"Generated {len(embeddings)} embeddings in batch")
            return embeddings
            
        except Exception as e:
            logger.error(f"Error generating batch embeddings: {e}")
            raise
    
    def _preprocess_text(self, text: str) -> str:
        """
        Preprocess text before embedding.
        
        Args:
            text: Raw input text
            
        Returns:
            Cleaned and preprocessed text
        """
        # Remove excessive whitespace
        cleaned = " ".join(text.strip().split())
        
        # Truncate if too long (OpenAI has token limits)
        max_chars = 8000  # Conservative limit
        if len(cleaned) > max_chars:
            cleaned = cleaned[:max_chars]
            logger.warning(f"Text truncated to {max_chars} characters")
        
        return cleaned
    
    def compute_cosine_similarity(
        self, 
        embedding1: Union[List[float], np.ndarray], 
        embedding2: Union[List[float], np.ndarray]
    ) -> float:
        """
        Compute cosine similarity between two embeddings.
        
        Args:
            embedding1: First embedding vector
            embedding2: Second embedding vector
            
        Returns:
            Cosine similarity score between 0 and 1
        """
        # Convert to numpy arrays
        vec1 = np.array(embedding1)
        vec2 = np.array(embedding2)
        
        # Compute cosine similarity
        dot_product = np.dot(vec1, vec2)
        norm1 = np.linalg.norm(vec1)
        norm2 = np.linalg.norm(vec2)
        
        if norm1 == 0 or norm2 == 0:
            return 0.0
        
        similarity = dot_product / (norm1 * norm2)
        
        # Ensure result is between 0 and 1
        return max(0.0, min(1.0, float(similarity)))
    
    async def health_check(self) -> bool:
        """
        Check if the embedding service is healthy.
        
        Returns:
            True if service is healthy, False otherwise
        """
        try:
            # Test with a simple embedding
            test_embedding = await self.get_embedding("test")
            return len(test_embedding) == settings.vector_dimension
        except Exception as e:
            logger.error(f"Health check failed: {e}")
            return False


# Global embedding service instance
embedding_service = EmbeddingService()
