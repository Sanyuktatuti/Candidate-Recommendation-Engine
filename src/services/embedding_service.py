"""
Unified embedding service with automatic hierarchy: OpenAI → Cohere → HF → TF-IDF.
"""

import os
import re
import time
from typing import List, Dict, Any, Tuple
import streamlit as st

# Load environment variables for local development
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    # dotenv not available, that's fine
    pass

# AI and ML imports
try:
    import openai
except ImportError:
    openai = None

try:
    import cohere
except ImportError:
    cohere = None

try:
    import requests
except ImportError:
    requests = None

from ..models.candidate import Candidate, JobDescription


class UnifiedEmbeddingService:
    """Unified embedding service with automatic hierarchy: OpenAI → Cohere → HF → TF-IDF."""
    
    def __init__(self):
        """Initialize the embedding service with automatic tier selection."""
        # API Keys from Streamlit Secrets (secure server-side storage)
        self._load_api_keys()
        
        # Service initialization
        self.vectorizer = None
        self.method = "unified_hierarchy"
        self.dimension = 1536  # OpenAI default
        self.sentence_model = None
        self.active_service = None
        self.openai_client = None
        self.cohere_client = None
        self.hf_headers = None
        self.hf_api_url = None
        
        # Initialize service hierarchy
        self._init_service_hierarchy()
    
    def _load_api_keys(self) -> None:
        """Load API keys prioritizing environment variables (local) then Streamlit secrets (cloud)."""
        # Priority 1: Environment variables (for local development)
        self.OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
        self.COHERE_API_KEY = os.getenv("COHERE_API_KEY", "")
        self.HF_API_TOKEN = os.getenv("HF_API_TOKEN", "")
        
        # Priority 2: Streamlit secrets (for cloud deployment) - only if env vars are empty
        if not any([self.OPENAI_API_KEY, self.COHERE_API_KEY, self.HF_API_TOKEN]):
            try:
                self.OPENAI_API_KEY = st.secrets.get("OPENAI_API_KEY", "")
                self.COHERE_API_KEY = st.secrets.get("COHERE_API_KEY", "")
                self.HF_API_TOKEN = st.secrets.get("HF_API_TOKEN", "")
            except Exception:
                # No secrets available, use empty strings (will fallback to TF-IDF)
                pass
    
    def _init_service_hierarchy(self) -> None:
        """Initialize services in priority order: OpenAI → Cohere → HuggingFace → TF-IDF."""
        services_tried = []
        
        # TIER 1: OpenAI API (Premium quality - highest priority)
        if self._init_openai():
            self.active_service = "openai"
            self.method = "openai_embeddings"
            self.dimension = 1536
            return
        else:
            services_tried.append("OpenAI")
        
        # TIER 2: Cohere API (Excellent fallback)
        if self._init_cohere():
            self.active_service = "cohere"
            self.method = "cohere_embed_v3"
            self.dimension = 1024
            return
        else:
            services_tried.append("Cohere")
        
        # TIER 3: Hugging Face Inference API (Good fallback)
        if self._init_huggingface():
            self.active_service = "huggingface"
            self.method = "hf_inference_api"
            self.dimension = 384
            return
        else:
            services_tried.append("Hugging Face")
        
        # TIER 4: Enhanced TF-IDF (Always works)
        self.active_service = "tfidf"
        self.method = "enhanced_tfidf"
        self.dimension = 2000
    
    def _init_openai(self) -> bool:
        """Initialize OpenAI API."""
        if not openai or not self.OPENAI_API_KEY:
            return False
            
        try:
            self.openai_client = openai.OpenAI(api_key=self.OPENAI_API_KEY)
            
            # Quick test
            self.openai_client.embeddings.create(
                input="test",
                model="text-embedding-ada-002"
            )
            return True
        except Exception:
            return False
    
    def _init_cohere(self) -> bool:
        """Initialize Cohere API."""
        if not cohere or not self.COHERE_API_KEY:
            return False
            
        try:
            self.cohere_client = cohere.Client(self.COHERE_API_KEY)
            
            # Quick test
            self.cohere_client.embed(
                texts=["test"],
                model="embed-english-v3.0",
                input_type="search_document"
            )
            return True
        except Exception:
            return False
    
    def _init_huggingface(self) -> bool:
        """Initialize Hugging Face Inference API."""
        if not requests or not self.HF_API_TOKEN:
            return False
            
        try:
            self.hf_headers = {"Authorization": f"Bearer {self.HF_API_TOKEN}"}
            self.hf_api_url = "https://api-inference.huggingface.co/pipeline/feature-extraction/sentence-transformers/all-MiniLM-L6-v2"
            
            # Test the API
            response = requests.post(
                self.hf_api_url,
                headers=self.hf_headers,
                json={"inputs": "test"},
                timeout=10
            )
            return response.status_code == 200
        except Exception:
            return False
    
    def get_embedding(self, text: str) -> List[float]:
        """Get embedding using the best available service."""
        processed_text = self._preprocess_text_advanced(text)
        
        if self.active_service == "openai":
            return self._get_openai_embedding(processed_text)
        elif self.active_service == "cohere":
            return self._get_cohere_embedding(processed_text)
        elif self.active_service == "huggingface":
            return self._get_hf_embedding(processed_text)
        else:
            return self._get_tfidf_embedding(processed_text)
    
    def get_embeddings_batch(self, texts: List[str]) -> List[List[float]]:
        """Get embeddings for multiple texts."""
        processed_texts = [self._preprocess_text_advanced(text) for text in texts]
        
        if self.active_service == "openai":
            return self._get_openai_embeddings_batch(processed_texts)
        elif self.active_service == "cohere":
            return self._get_cohere_embeddings_batch(processed_texts)
        elif self.active_service == "huggingface":
            return self._get_hf_embeddings_batch(processed_texts)
        else:
            return self._get_tfidf_embeddings_batch(processed_texts)
    
    def process_job_and_candidates(self, job: JobDescription, candidates: List[Candidate]) -> Tuple[List[float], List[List[float]]]:
        """Process job and candidates for embeddings."""
        candidate_texts = [c.resume_text for c in candidates]
        
        if self.active_service in ["openai", "cohere", "huggingface"]:
            # For API services, process separately
            job_embedding = self.get_embedding(job.full_text)
            candidate_embeddings = [self.get_embedding(text) for text in candidate_texts]
            return job_embedding, candidate_embeddings
        else:
            # For TF-IDF, need to fit together
            all_texts = [job.full_text] + candidate_texts
            all_embeddings = self.get_embeddings_batch(all_texts)
            return all_embeddings[0], all_embeddings[1:]
    
    def _get_openai_embedding(self, text: str) -> List[float]:
        """Get OpenAI embedding."""
        try:
            response = self.openai_client.embeddings.create(
                input=text,
                model="text-embedding-ada-002"
            )
            return response.data[0].embedding
        except Exception:
            return [0.0] * self.dimension
    
    def _get_openai_embeddings_batch(self, texts: List[str]) -> List[List[float]]:
        """Get OpenAI embeddings for batch."""
        try:
            embeddings = []
            for text in texts:
                response = self.openai_client.embeddings.create(
                    input=text,
                    model="text-embedding-ada-002"
                )
                embeddings.append(response.data[0].embedding)
            return embeddings
        except Exception:
            return [[0.0] * self.dimension for _ in texts]
    
    def _get_cohere_embedding(self, text: str) -> List[float]:
        """Get Cohere embedding."""
        try:
            response = self.cohere_client.embed(
                texts=[text],
                model="embed-english-v3.0",
                input_type="search_document"
            )
            return response.embeddings[0]
        except Exception:
            return [0.0] * self.dimension
    
    def _get_cohere_embeddings_batch(self, texts: List[str]) -> List[List[float]]:
        """Get Cohere embeddings for batch."""
        try:
            response = self.cohere_client.embed(
                texts=texts,
                model="embed-english-v3.0",
                input_type="search_document"
            )
            return response.embeddings
        except Exception:
            return [[0.0] * self.dimension for _ in texts]
    
    def _get_hf_embedding(self, text: str) -> List[float]:
        """Get Hugging Face embedding."""
        try:
            response = requests.post(
                self.hf_api_url,
                headers=self.hf_headers,
                json={"inputs": text},
                timeout=30
            )
            if response.status_code == 200:
                return response.json()
            else:
                return [0.0] * self.dimension
        except Exception:
            return [0.0] * self.dimension
    
    def _get_hf_embeddings_batch(self, texts: List[str]) -> List[List[float]]:
        """Get HF embeddings for batch (with rate limiting)."""
        embeddings = []
        for text in texts:
            embedding = self._get_hf_embedding(text)
            embeddings.append(embedding)
            time.sleep(0.1)  # Rate limiting for free tier
        return embeddings
    
    def _get_tfidf_embedding(self, text: str) -> List[float]:
        """Get TF-IDF embedding (fallback)."""
        if not hasattr(self, '_fitted') or not self._fitted:
            return [0.0] * self.dimension
        
        try:
            from sklearn.feature_extraction.text import TfidfVectorizer
            tfidf_matrix = self.vectorizer.transform([text])
            return tfidf_matrix.toarray()[0].tolist()
        except Exception:
            return [0.0] * self.dimension
    
    def _get_tfidf_embeddings_batch(self, texts: List[str]) -> List[List[float]]:
        """Get TF-IDF embeddings for batch."""
        try:
            from sklearn.feature_extraction.text import TfidfVectorizer
            
            self.vectorizer = TfidfVectorizer(
                max_features=2000,
                stop_words='english',
                ngram_range=(1, 3),
                min_df=1,
                max_df=0.8,
                sublinear_tf=True,
                norm='l2'
            )
            
            tfidf_matrix = self.vectorizer.fit_transform(texts)
            self._fitted = True
            
            dense_matrix = tfidf_matrix.toarray()
            return [row.tolist() for row in dense_matrix]
        except Exception:
            return [[0.0] * self.dimension for _ in texts]
    
    def _preprocess_text_advanced(self, text: str) -> str:
        """Advanced text preprocessing for better matching."""
        text = text.lower()
        
        # Normalize job-related terms
        replacements = {
            'artificial intelligence': 'ai machine learning',
            'machine learning': 'ml ai data science',
            'deep learning': 'dl neural networks ai',
            'natural language processing': 'nlp text analysis',
            'computer vision': 'cv image processing',
            'data science': 'datascience analytics ml',
            'software engineering': 'programming development coding',
            'full stack': 'fullstack frontend backend',
            'front end': 'frontend ui web development',
            'back end': 'backend server api development',
            'devops': 'deployment automation ci cd',
            'cloud computing': 'cloud aws azure gcp',
            'amazon web services': 'aws cloud computing',
            'google cloud platform': 'gcp cloud computing',
            'microsoft azure': 'azure cloud computing',
        }
        
        for original, enhanced in replacements.items():
            text = text.replace(original, enhanced)
        
        # Remove noise
        text = re.sub(r'\S+@\S+', '', text)  # emails
        text = re.sub(r'http\S+', '', text)  # URLs
        text = re.sub(r'\d{4}-\d{4}', '', text)  # dates
        
        return text
    
    def get_info(self) -> Dict[str, Any]:
        """Get service information."""
        service_info = {
            "openai": {
                "method": "OpenAI text-embedding-ada-002",
                "description": "Premium-grade semantic embeddings with industry-leading accuracy and understanding"
            },
            "cohere": {
                "method": "Cohere Embed v3.0 API",
                "description": "Professional-grade semantic embeddings with excellent quality and multilingual support"
            },
            "huggingface": {
                "method": "Hugging Face Inference API", 
                "description": "Production-quality semantic embeddings via managed transformer endpoints"
            },
            "tfidf": {
                "method": "Enhanced TF-IDF + Domain Knowledge",
                "description": "Advanced keyword and phrase matching with professional preprocessing"
            }
        }
        
        return service_info.get(self.active_service, service_info["tfidf"])
