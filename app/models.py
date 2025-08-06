"""
Pydantic models for API request/response schemas.
"""
from typing import List, Optional, Dict, Any
from pydantic import BaseModel, Field
from uuid import UUID, uuid4


class CandidateBase(BaseModel):
    """Base candidate model."""
    name: str = Field(..., description="Candidate name or identifier")
    resume_text: str = Field(..., description="Resume content as text")


class CandidateCreate(CandidateBase):
    """Model for creating a new candidate."""
    pass


class CandidateResponse(CandidateBase):
    """Model for candidate response with ID."""
    id: UUID = Field(default_factory=uuid4, description="Unique candidate ID")
    
    class Config:
        from_attributes = True


class JobDescription(BaseModel):
    """Model for job description input."""
    title: str = Field(..., description="Job title")
    description: str = Field(..., description="Job description content")
    requirements: Optional[str] = Field(None, description="Job requirements")


class SimilarityMatch(BaseModel):
    """Model for similarity match result."""
    candidate_id: UUID = Field(..., description="Candidate unique ID")
    candidate_name: str = Field(..., description="Candidate name")
    similarity_score: float = Field(..., ge=0, le=1, description="Cosine similarity score")
    resume_text: str = Field(..., description="Resume content")
    ai_summary: Optional[str] = Field(None, description="AI-generated fit summary")


class SearchRequest(BaseModel):
    """Model for search request."""
    job_description: JobDescription = Field(..., description="Job description to match against")
    candidates: List[CandidateCreate] = Field(..., description="List of candidates to evaluate")
    top_k: int = Field(10, ge=1, le=50, description="Number of top matches to return")
    include_summary: bool = Field(True, description="Whether to include AI-generated summaries")


class SearchResponse(BaseModel):
    """Model for search response."""
    job_description: JobDescription = Field(..., description="Original job description")
    matches: List[SimilarityMatch] = Field(..., description="Top matching candidates")
    total_candidates: int = Field(..., description="Total number of candidates evaluated")
    processing_time: float = Field(..., description="Processing time in seconds")


class HealthCheck(BaseModel):
    """Health check response model."""
    status: str = Field(..., description="Service status")
    version: str = Field(..., description="Application version")
    openai_status: str = Field(..., description="OpenAI API connectivity status")


class ErrorResponse(BaseModel):
    """Error response model."""
    detail: str = Field(..., description="Error message")
    error_code: Optional[str] = Field(None, description="Error code")
    timestamp: str = Field(..., description="Error timestamp")
