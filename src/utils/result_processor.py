"""
Result processing utilities for candidate matching results.
"""

from typing import List
from ..models.candidate import Candidate, JobDescription, SearchResult
from ..services.ai_service import UnifiedAIService
from ..services.embedding_service import UnifiedEmbeddingService
from .similarity import SimilarityCalculator


class ResultProcessor:
    """Handles processing and ranking of candidate matching results."""
    
    def __init__(self, embedding_service: UnifiedEmbeddingService, ai_service: UnifiedAIService):
        """Initialize the result processor.
        
        Args:
            embedding_service: The embedding service to use
            ai_service: The AI service to use
        """
        self.embedding_service = embedding_service
        self.ai_service = ai_service
        self.similarity_calculator = SimilarityCalculator()
    
    def process_candidates(
        self, 
        job: JobDescription, 
        candidates: List[Candidate],
        include_summaries: bool = True,
        top_k: int = 10
    ) -> SearchResult:
        """Process candidates and return ranked results.
        
        Args:
            job: The job description
            candidates: List of candidates to process
            include_summaries: Whether to generate AI summaries
            top_k: Number of top candidates to return
            
        Returns:
            SearchResult with processed and ranked candidates
        """
        import time
        start_time = time.time()
        
        # Generate embeddings
        job_embedding, candidate_embeddings = self.embedding_service.process_job_and_candidates(
            job, candidates
        )
        
        # Set job embedding
        job.embedding = job_embedding
        
        # Calculate similarities
        is_basic_mode = self.embedding_service.active_service == "tfidf"
        similarities = self.similarity_calculator.compute_similarity_scores(
            job_embedding, candidate_embeddings, is_basic_mode
        )
        
        # Update candidates with embeddings and similarities
        for i, candidate in enumerate(candidates):
            if i < len(candidate_embeddings):
                candidate.embedding = candidate_embeddings[i]
            if i < len(similarities):
                candidate.similarity_score = similarities[i]
        
        # Generate summaries if requested
        if include_summaries:
            for candidate in candidates:
                # Use new prompt hygiene system
                summary_result = self.ai_service.generate_fit_summary(
                    job.full_text,
                    candidate.resume_text,
                    candidate.name
                )
                
                # Extract summary text for backward compatibility
                candidate.fit_summary = summary_result.get('summary', 'Analysis not available') if isinstance(summary_result, dict) else summary_result
        
        processing_time = time.time() - start_time
        
        # Create and return search result
        search_result = SearchResult(
            job_description=job,
            candidates=candidates[:top_k] if top_k else candidates,
            processing_time=processing_time,
            service_info=self.embedding_service.get_info()
        )
        
        return search_result
    
    def rank_candidates(self, candidates: List[Candidate]) -> List[Candidate]:
        """Rank candidates by similarity score.
        
        Args:
            candidates: List of candidates to rank
            
        Returns:
            List of candidates sorted by similarity score (descending)
        """
        return sorted(
            [c for c in candidates if c.similarity_score is not None],
            key=lambda x: x.similarity_score,
            reverse=True
        )
    
    def filter_candidates(
        self, 
        candidates: List[Candidate], 
        min_similarity: float = 0.0
    ) -> List[Candidate]:
        """Filter candidates by minimum similarity threshold.
        
        Args:
            candidates: List of candidates to filter
            min_similarity: Minimum similarity score threshold
            
        Returns:
            Filtered list of candidates
        """
        return [
            c for c in candidates 
            if c.similarity_score is not None and c.similarity_score >= min_similarity
        ]
