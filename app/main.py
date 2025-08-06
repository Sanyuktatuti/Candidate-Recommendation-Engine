"""
FastAPI main application for Candidate Recommendation Engine.
"""
import time
import logging
from datetime import datetime
from typing import List
from fastapi import FastAPI, HTTPException, UploadFile, File, Form, Depends
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import uvicorn

from config import settings
from app.models import (
    SearchRequest, SearchResponse, HealthCheck, ErrorResponse,
    JobDescription, CandidateCreate, CandidateResponse, SimilarityMatch
)
from app.services.embedding_service import embedding_service
from app.services.vector_service import vector_service
from app.services.document_service import document_service

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Create FastAPI app
app = FastAPI(
    title=settings.app_name,
    version=settings.app_version,
    description="AI-powered candidate recommendation system",
    docs_url="/docs",
    redoc_url="/redoc"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, specify actual origins
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.exception_handler(Exception)
async def global_exception_handler(request, exc):
    """Global exception handler."""
    logger.error(f"Unhandled exception: {exc}")
    return JSONResponse(
        status_code=500,
        content=ErrorResponse(
            detail="Internal server error",
            error_code="INTERNAL_ERROR",
            timestamp=datetime.now().isoformat()
        ).dict()
    )


@app.get("/health", response_model=HealthCheck)
async def health_check():
    """Health check endpoint."""
    try:
        # Check OpenAI API connectivity
        openai_status = "healthy" if await embedding_service.health_check() else "unhealthy"
        
        return HealthCheck(
            status="healthy",
            version=settings.app_version,
            openai_status=openai_status
        )
    except Exception as e:
        logger.error(f"Health check failed: {e}")
        raise HTTPException(status_code=503, detail="Service unhealthy")


@app.post("/search", response_model=SearchResponse)
async def search_candidates(request: SearchRequest):
    """
    Search for the best matching candidates for a job description.
    
    Args:
        request: Search request containing job description and candidates
        
    Returns:
        Search response with ranked candidates
    """
    start_time = time.time()
    
    try:
        logger.info(f"Processing search request for {len(request.candidates)} candidates")
        
        # Validate input
        if not request.candidates:
            raise HTTPException(status_code=400, detail="No candidates provided")
        
        if len(request.candidates) > settings.max_candidates:
            raise HTTPException(
                status_code=400, 
                detail=f"Too many candidates. Maximum allowed: {settings.max_candidates}"
            )
        
        # Create job description text
        job_text = f"{request.job_description.title}\n\n{request.job_description.description}"
        if request.job_description.requirements:
            job_text += f"\n\nRequirements:\n{request.job_description.requirements}"
        
        # Convert candidates to response format
        candidate_responses = [
            CandidateResponse(name=c.name, resume_text=c.resume_text)
            for c in request.candidates
        ]
        
        # Generate embeddings for job description and candidates
        logger.info("Generating embeddings...")
        
        # Get job embedding
        job_embedding = await embedding_service.get_embedding(job_text)
        
        # Get candidate embeddings
        candidate_texts = [c.resume_text for c in request.candidates]
        candidate_embeddings = await embedding_service.get_embeddings_batch(candidate_texts)
        
        # Add candidates to vector index
        vector_service.add_candidates(candidate_responses, candidate_embeddings)
        
        # Search for similar candidates
        search_results = await vector_service.search_similar_candidates(
            job_embedding, 
            top_k=request.top_k
        )
        
        # Create similarity matches
        matches = await vector_service.create_similarity_matches(
            job_text,
            search_results,
            include_summary=request.include_summary
        )
        
        processing_time = time.time() - start_time
        
        logger.info(f"Search completed in {processing_time:.2f} seconds")
        
        return SearchResponse(
            job_description=request.job_description,
            matches=matches,
            total_candidates=len(request.candidates),
            processing_time=processing_time
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Search error: {e}")
        raise HTTPException(status_code=500, detail=f"Search failed: {str(e)}")


@app.post("/upload-search")
async def upload_search(
    job_title: str = Form(...),
    job_description: str = Form(...),
    job_requirements: str = Form(""),
    top_k: int = Form(10),
    include_summary: bool = Form(True),
    files: List[UploadFile] = File(...)
):
    """
    Search for candidates using uploaded resume files.
    
    Args:
        job_title: Job title
        job_description: Job description text
        job_requirements: Job requirements (optional)
        top_k: Number of top candidates to return
        include_summary: Whether to include AI summaries
        files: List of uploaded resume files
        
    Returns:
        Search response with ranked candidates
    """
    start_time = time.time()
    
    try:
        logger.info(f"Processing upload search with {len(files)} files")
        
        # Validate files
        if not files:
            raise HTTPException(status_code=400, detail="No files uploaded")
        
        if len(files) > settings.max_candidates:
            raise HTTPException(
                status_code=400,
                detail=f"Too many files. Maximum allowed: {settings.max_candidates}"
            )
        
        # Process uploaded files
        candidates = []
        for file in files:
            # Validate file
            if not document_service.validate_file_extension(file.filename):
                raise HTTPException(
                    status_code=400,
                    detail=f"Unsupported file type: {file.filename}"
                )
            
            # Extract text from file
            try:
                resume_text = await document_service.extract_text_from_upload(file)
                cleaned_text = document_service.clean_extracted_text(resume_text)
                
                # Use filename as candidate name (remove extension)
                candidate_name = file.filename.rsplit('.', 1)[0] if file.filename else "Unknown"
                
                candidates.append(CandidateCreate(
                    name=candidate_name,
                    resume_text=cleaned_text
                ))
                
            except Exception as e:
                logger.error(f"Error processing file {file.filename}: {e}")
                raise HTTPException(
                    status_code=400,
                    detail=f"Failed to process file {file.filename}: {str(e)}"
                )
        
        # Create search request
        job_desc = JobDescription(
            title=job_title,
            description=job_description,
            requirements=job_requirements if job_requirements else None
        )
        
        search_request = SearchRequest(
            job_description=job_desc,
            candidates=candidates,
            top_k=top_k,
            include_summary=include_summary
        )
        
        # Use existing search endpoint
        return await search_candidates(search_request)
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Upload search error: {e}")
        raise HTTPException(status_code=500, detail=f"Upload search failed: {str(e)}")


@app.get("/stats")
async def get_stats():
    """Get system statistics."""
    try:
        vector_stats = vector_service.get_stats()
        
        return {
            "vector_index": vector_stats,
            "settings": {
                "max_candidates": settings.max_candidates,
                "vector_dimension": settings.vector_dimension,
                "embedding_model": settings.embedding_model,
                "chat_model": settings.chat_model
            }
        }
    except Exception as e:
        logger.error(f"Stats error: {e}")
        raise HTTPException(status_code=500, detail="Failed to get stats")


if __name__ == "__main__":
    uvicorn.run(
        "app.main:app",
        host=settings.api_host,
        port=settings.api_port,
        reload=settings.debug,
        log_level="info"
    )
