"""
Data models for candidate and job information.
"""

from dataclasses import dataclass
from typing import List, Optional, Dict, Any


@dataclass
class Candidate:
    """Represents a job candidate."""
    name: str
    resume_text: str
    embedding: Optional[List[float]] = None
    similarity_score: Optional[float] = None
    fit_summary: Optional[str] = None
    
    def __post_init__(self):
        """Validate candidate data."""
        if not self.name.strip():
            raise ValueError("Candidate name cannot be empty")
        if not self.resume_text.strip():
            raise ValueError("Resume text cannot be empty")


@dataclass
class JobDescription:
    """Represents a job description."""
    title: str
    description: str
    requirements: Optional[str] = None
    embedding: Optional[List[float]] = None
    
    def __post_init__(self):
        """Validate job description data."""
        if not self.title.strip():
            raise ValueError("Job title cannot be empty")
        if not self.description.strip():
            raise ValueError("Job description cannot be empty")
    
    @property
    def full_text(self) -> str:
        """Get full job text for embedding generation."""
        text = f"{self.title}\n\n{self.description}"
        if self.requirements:
            text += f"\n\nRequirements:\n{self.requirements}"
        return text


@dataclass
class SearchResult:
    """Represents search results for candidate matching."""
    job_description: JobDescription
    candidates: List[Candidate]
    processing_time: float
    service_info: Dict[str, Any]
    
    @property
    def ranked_candidates(self) -> List[Candidate]:
        """Get candidates ranked by similarity score."""
        return sorted(
            [c for c in self.candidates if c.similarity_score is not None],
            key=lambda x: x.similarity_score,
            reverse=True
        )
    
    @property
    def top_candidates(self, n: int = 10) -> List[Candidate]:
        """Get top N candidates."""
        return self.ranked_candidates[:n]
    
    @property
    def average_similarity(self) -> float:
        """Calculate average similarity score."""
        scores = [c.similarity_score for c in self.candidates if c.similarity_score is not None]
        return sum(scores) / len(scores) if scores else 0.0
    
    @property
    def strong_matches_count(self, threshold: float = 0.7) -> int:
        """Count candidates with similarity >= threshold."""
        return sum(
            1 for c in self.candidates 
            if c.similarity_score is not None and c.similarity_score >= threshold
        )
