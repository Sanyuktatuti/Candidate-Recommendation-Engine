"""
Similarity calculation utilities.
"""

from typing import List
import numpy as np

try:
    from sklearn.metrics.pairwise import cosine_similarity
except ImportError:
    cosine_similarity = None


class SimilarityCalculator:
    """Handles similarity calculations between job and candidate embeddings."""
    
    @staticmethod
    def compute_similarity_scores(
        job_embedding: List[float], 
        candidate_embeddings: List[List[float]], 
        is_basic_mode: bool = False
    ) -> List[float]:
        """Compute cosine similarity scores between job and candidates.
        
        Args:
            job_embedding: The job description embedding
            candidate_embeddings: List of candidate resume embeddings
            is_basic_mode: Whether using basic TF-IDF mode (affects score scaling)
            
        Returns:
            List of similarity scores (0.0 to 1.0) for each candidate
        """
        if not job_embedding or not candidate_embeddings:
            return []
        
        if cosine_similarity is None:
            raise ImportError("scikit-learn is required for similarity computation")
        
        job_vec = np.array(job_embedding).reshape(1, -1)
        candidate_vecs = np.array(candidate_embeddings)
        
        # Compute cosine similarity
        similarities = cosine_similarity(job_vec, candidate_vecs)[0]
        
        # Enhance scores for basic mode to make them more meaningful
        if is_basic_mode:
            # Apply square root transformation to spread out low scores
            similarities = np.sqrt(np.maximum(similarities, 0)) * 0.85  # Max ~85% for basic mode
        
        # Ensure scores are between 0 and 1
        similarities = np.clip(similarities, 0, 1)
        
        return similarities.tolist()
    
    @staticmethod
    def normalize_embeddings(embeddings: List[List[float]]) -> List[List[float]]:
        """Normalize embeddings to unit vectors.
        
        Args:
            embeddings: List of embedding vectors
            
        Returns:
            List of normalized embedding vectors
        """
        normalized = []
        for embedding in embeddings:
            vec = np.array(embedding)
            norm = np.linalg.norm(vec)
            if norm > 0:
                normalized.append((vec / norm).tolist())
            else:
                normalized.append(embedding)
        return normalized
    
    @staticmethod
    def get_similarity_metrics(similarities: List[float]) -> dict:
        """Calculate various similarity metrics.
        
        Args:
            similarities: List of similarity scores
            
        Returns:
            Dictionary with various metrics
        """
        if not similarities:
            return {
                'average': 0.0,
                'max': 0.0,
                'min': 0.0,
                'std': 0.0,
                'strong_matches': 0
            }
        
        similarities_array = np.array(similarities)
        
        return {
            'average': float(np.mean(similarities_array)),
            'max': float(np.max(similarities_array)),
            'min': float(np.min(similarities_array)),
            'std': float(np.std(similarities_array)),
            'strong_matches': int(np.sum(similarities_array >= 0.7))
        }
