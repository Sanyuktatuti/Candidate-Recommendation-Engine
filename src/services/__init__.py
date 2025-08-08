"""
Service layer for the Candidate Recommendation Engine.
"""

from .document_processor import DocumentProcessor
from .embedding_service import UnifiedEmbeddingService
from .ai_service import UnifiedAIService

__all__ = ["DocumentProcessor", "UnifiedEmbeddingService", "UnifiedAIService"]
