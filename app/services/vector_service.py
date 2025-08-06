"""
Vector similarity service using FAISS for efficient similarity search.
"""
import logging
from typing import List, Tuple, Optional
import numpy as np
import faiss
from uuid import UUID
from app.models import CandidateResponse, SimilarityMatch
from app.services.embedding_service import embedding_service

logger = logging.getLogger(__name__)


class VectorService:
    """Service for managing vector embeddings and similarity search using FAISS."""
    
    def __init__(self, dimension: int = 1536):
        """
        Initialize the vector service.
        
        Args:
            dimension: Dimension of the embedding vectors
        """
        self.dimension = dimension
        self.index = None
        self.candidates = {}  # Store candidate metadata by index
        self.candidate_embeddings = []  # Store embeddings in order
        self._build_index()
    
    def _build_index(self):
        """Build a new FAISS index."""
        # Use IndexFlatIP for exact cosine similarity search
        # Note: FAISS uses inner product, so we'll normalize vectors for cosine similarity
        self.index = faiss.IndexFlatIP(self.dimension)
        self.candidates = {}
        self.candidate_embeddings = []
        logger.info(f"Built new FAISS index with dimension {self.dimension}")
    
    def add_candidates(
        self, 
        candidates: List[CandidateResponse], 
        embeddings: List[List[float]]
    ) -> None:
        """
        Add candidates and their embeddings to the index.
        
        Args:
            candidates: List of candidate objects
            embeddings: List of embedding vectors corresponding to candidates
        """
        if len(candidates) != len(embeddings):
            raise ValueError("Number of candidates must match number of embeddings")
        
        # Clear existing index and rebuild
        self._build_index()
        
        # Normalize embeddings for cosine similarity
        normalized_embeddings = []
        for i, embedding in enumerate(embeddings):
            vec = np.array(embedding, dtype=np.float32)
            # Normalize vector for cosine similarity
            norm = np.linalg.norm(vec)
            if norm > 0:
                vec = vec / norm
            normalized_embeddings.append(vec)
            
            # Store candidate metadata
            self.candidates[i] = candidates[i]
            self.candidate_embeddings.append(embedding)
        
        # Add to FAISS index
        if normalized_embeddings:
            embeddings_matrix = np.vstack(normalized_embeddings)
            self.index.add(embeddings_matrix)
            logger.info(f"Added {len(candidates)} candidates to vector index")
    
    async def search_similar_candidates(
        self, 
        job_embedding: List[float], 
        top_k: int = 10
    ) -> List[Tuple[int, float]]:
        """
        Search for candidates similar to the job description.
        
        Args:
            job_embedding: Embedding vector of the job description
            top_k: Number of top results to return
            
        Returns:
            List of tuples (candidate_index, similarity_score)
        """
        if self.index is None or self.index.ntotal == 0:
            logger.warning("No candidates in index")
            return []
        
        # Normalize job embedding for cosine similarity
        job_vec = np.array(job_embedding, dtype=np.float32).reshape(1, -1)
        norm = np.linalg.norm(job_vec)
        if norm > 0:
            job_vec = job_vec / norm
        
        # Search in FAISS index
        k = min(top_k, self.index.ntotal)
        similarities, indices = self.index.search(job_vec, k)
        
        # Convert to list of tuples
        results = []
        for i in range(len(indices[0])):
            candidate_idx = indices[0][i]
            similarity = float(similarities[0][i])
            
            # FAISS inner product gives us cosine similarity for normalized vectors
            # Ensure similarity is between 0 and 1
            similarity = max(0.0, min(1.0, similarity))
            
            results.append((candidate_idx, similarity))
        
        logger.info(f"Found {len(results)} similar candidates")
        return results
    
    def get_candidate_by_index(self, index: int) -> Optional[CandidateResponse]:
        """
        Get candidate by index.
        
        Args:
            index: Candidate index in the vector store
            
        Returns:
            CandidateResponse object or None if not found
        """
        return self.candidates.get(index)
    
    async def create_similarity_matches(
        self, 
        job_description_text: str,
        search_results: List[Tuple[int, float]],
        include_summary: bool = True
    ) -> List[SimilarityMatch]:
        """
        Create SimilarityMatch objects from search results.
        
        Args:
            job_description_text: Original job description text
            search_results: List of (candidate_index, similarity_score) tuples
            include_summary: Whether to include AI-generated summaries
            
        Returns:
            List of SimilarityMatch objects
        """
        matches = []
        
        for candidate_idx, similarity_score in search_results:
            candidate = self.get_candidate_by_index(candidate_idx)
            if candidate is None:
                continue
            
            # Generate AI summary if requested
            ai_summary = None
            if include_summary:
                try:
                    ai_summary = await self._generate_fit_summary(
                        job_description_text, 
                        candidate.resume_text,
                        candidate.name
                    )
                except Exception as e:
                    logger.error(f"Error generating summary for candidate {candidate.name}: {e}")
                    ai_summary = "Summary generation failed"
            
            match = SimilarityMatch(
                candidate_id=candidate.id,
                candidate_name=candidate.name,
                similarity_score=similarity_score,
                resume_text=candidate.resume_text,
                ai_summary=ai_summary
            )
            matches.append(match)
        
        return matches
    
    async def _generate_fit_summary(
        self, 
        job_description: str, 
        resume_text: str, 
        candidate_name: str
    ) -> str:
        """
        Generate an AI summary of why the candidate is a good fit.
        
        Args:
            job_description: Job description text
            resume_text: Candidate resume text
            candidate_name: Candidate name
            
        Returns:
            AI-generated summary string
        """
        from app.services.ai_service import ai_service
        return await ai_service.generate_fit_summary(
            job_description, 
            resume_text, 
            candidate_name
        )
    
    def get_stats(self) -> dict:
        """
        Get statistics about the vector index.
        
        Returns:
            Dictionary with index statistics
        """
        return {
            "total_candidates": self.index.ntotal if self.index else 0,
            "dimension": self.dimension,
            "index_type": type(self.index).__name__ if self.index else None
        }


# Global vector service instance
vector_service = VectorService()
