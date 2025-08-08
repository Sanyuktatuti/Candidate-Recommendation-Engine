"""
Standalone Streamlit app for Streamlit Community Cloud deployment.
This version includes all backend functionality in a single file.
"""
import os
import io
import time
import asyncio
import logging
from typing import List, Dict, Any, Optional
from uuid import UUID, uuid4

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go

# Document processing
import PyPDF2
try:
    import docx
except ImportError:
    st.error("python-docx not installed. Install with: pip install python-docx")

# AI and ML
import openai
from sklearn.metrics.pairwise import cosine_similarity

# Professional API clients for enhanced free service
import requests
import json
import time
try:
    import cohere
except ImportError:
    cohere = None
    
# Set page config
st.set_page_config(
    page_title="Candidate Recommendation Engine",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for professional UI
st.markdown("""
<style>
    /* Main header styling */
    .main-header {
        font-size: 3.2rem;
        font-weight: 600;
        color: #FFFFFF;
        text-align: center;
        margin-bottom: 3rem;
        letter-spacing: -0.5px;
        background: linear-gradient(135deg, #3498DB 0%, #2980B9 100%);
        padding: 2rem;
        border-radius: 15px;
        box-shadow: 0 8px 24px rgba(52, 152, 219, 0.3);
        text-shadow: 0 2px 4px rgba(0, 0, 0, 0.2);
    }
    
    /* Section headers */
    .section-header {
        font-size: 1.4rem;
        font-weight: 600;
        color: #34495E;
        margin-top: 2rem;
        margin-bottom: 1rem;
        padding-left: 0.5rem;
        border-left: 4px solid #3498DB;
    }
    
    /* Progress button */
    .progress-button {
        background: linear-gradient(135deg, #3498DB 0%, #2980B9 100%);
        color: white;
        padding: 0.8rem 2rem;
        border: none;
        border-radius: 8px;
        font-size: 1.1rem;
        font-weight: 600;
        cursor: pointer;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
        transition: all 0.3s ease;
        display: flex;
        align-items: center;
        gap: 0.5rem;
        margin: 1.5rem auto;
    }
    
    .progress-button:hover {
        transform: translateY(-2px);
        box-shadow: 0 6px 12px rgba(0, 0, 0, 0.15);
        background: linear-gradient(135deg, #2980B9 0%, #1A6B9D 100%);
    }
    
    /* Cards and containers */
    .card {
        background: white;
        border-radius: 12px;
        padding: 1.5rem;
        box-shadow: 0 2px 8px rgba(0, 0, 0, 0.08);
        border: 1px solid #E8EBF0;
        margin-bottom: 1.5rem;
    }
    
    /* Similarity score styling */
    .similarity-score {
        font-size: 1.3rem;
        font-weight: 700;
        color: #27AE60;
        background: linear-gradient(135deg, #D5F4E6, #FDEEF4);
        padding: 0.5rem 1rem;
        border-radius: 20px;
        display: inline-block;
    }
    
    /* Candidate name styling */
    .candidate-name {
        font-size: 1.5rem;
        font-weight: 600;
        color: #2C3E50;
        margin-bottom: 0.8rem;
        border-bottom: 2px solid #ECF0F1;
        padding-bottom: 0.5rem;
    }
    
    /* Metric cards */
    .metric-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 1.5rem;
        border-radius: 12px;
        color: white;
        text-align: center;
        box-shadow: 0 4px 12px rgba(0, 0, 0, 0.15);
    }
    
    /* Upload area styling */
    .upload-area {
        border: 2px dashed #BDC3C7;
        border-radius: 8px;
        padding: 2rem;
        text-align: center;
        background: #FAFBFC;
        transition: all 0.3s ease;
    }
    
    .upload-area:hover {
        border-color: #3498DB;
        background: #EBF3FD;
    }
    
    /* Step indicator */
    .step-indicator {
        background: #E8F4FD;
        border: 1px solid #D6EAF8;
        border-radius: 8px;
        padding: 1rem;
        margin: 1rem 0;
        color: #2980B9;
        font-weight: 500;
        text-align: center;
    }
    
    /* Main container */
    .main .block-container {
        max-width: 1200px;
        padding-top: 2rem;
        padding-left: 2rem;
        padding-right: 2rem;
    }
    
    /* Input fields */
    .stTextInput > div > div > input {
        border-radius: 8px;
        border: 2px solid #E8EBF0;
        padding: 0.75rem;
        font-size: 1rem;
    }
    
    .stTextArea > div > div > textarea {
        border-radius: 8px;
        border: 2px solid #E8EBF0;
        padding: 0.75rem;
        font-size: 1rem;
    }
    
    .stTextInput > div > div > input:focus,
    .stTextArea > div > div > textarea:focus {
        border-color: #3498DB;
        box-shadow: 0 0 0 3px rgba(52, 152, 219, 0.1);
    }
    
    /* Clean section headers */
    .clean-header {
        font-size: 1.8rem;
        font-weight: 600;
        color: #2C3E50;
        margin: 2.5rem 0 1.5rem 0;
        padding: 1.2rem 1.5rem;
        background: linear-gradient(135deg, #F8F9FA 0%, #E9ECEF 100%);
        border-left: 5px solid #3498DB;
        border-radius: 0 12px 12px 0;
        box-shadow: 0 2px 8px rgba(0, 0, 0, 0.06);
    }
</style>
""", unsafe_allow_html=True)

# Configuration
MAX_CANDIDATES = 20
MAX_FILE_SIZE = 10 * 1024 * 1024  # 10MB
VECTOR_DIMENSION = 1536
EMBEDDING_MODEL = "text-embedding-ada-002"
CHAT_MODEL = "gpt-3.5-turbo"

class DocumentProcessor:
    """Process different document types."""
    
    @staticmethod
    def extract_text_from_pdf(content: bytes) -> str:
        """Extract text from PDF."""
        try:
            pdf_file = io.BytesIO(content)
            pdf_reader = PyPDF2.PdfReader(pdf_file)
            text_parts = []
            for page in pdf_reader.pages:
                page_text = page.extract_text()
                if page_text.strip():
                    text_parts.append(page_text)
            return '\n'.join(text_parts)
        except Exception as e:
            st.error(f"Error reading PDF: {e}")
            return ""
    
    @staticmethod
    def extract_text_from_docx(content: bytes) -> str:
        """Extract text from DOCX."""
        try:
            docx_file = io.BytesIO(content)
            doc = docx.Document(docx_file)
            text_parts = []
            for paragraph in doc.paragraphs:
                if paragraph.text.strip():
                    text_parts.append(paragraph.text)
            return '\n'.join(text_parts)
        except Exception as e:
            st.error(f"Error reading DOCX: {e}")
            return ""
    
    @staticmethod
    def extract_text_from_txt(content: bytes) -> str:
        """Extract text from TXT."""
        try:
            encodings = ['utf-8', 'utf-16', 'latin-1', 'cp1252']
            for encoding in encodings:
                try:
                    return content.decode(encoding)
                except UnicodeDecodeError:
                    continue
            return ""
        except Exception as e:
            st.error(f"Error reading TXT: {e}")
            return ""

class EmbeddingService:
    """Service for generating embeddings using OpenAI."""
    
    def __init__(self, api_key: str):
        self.client = openai.OpenAI(api_key=api_key)
        self.model = EMBEDDING_MODEL
    
    def get_embedding(self, text: str) -> List[float]:
        """Get embedding for text."""
        try:
            cleaned_text = ' '.join(text.strip().split())
            if len(cleaned_text) > 8000:
                cleaned_text = cleaned_text[:8000]
            
            response = self.client.embeddings.create(
                input=cleaned_text,
                model=self.model
            )
            return response.data[0].embedding
        except Exception as e:
            st.error(f"Error generating embedding: {e}")
            return []
    
    def get_embeddings_batch(self, texts: List[str]) -> List[List[float]]:
        """Get embeddings for multiple texts."""
        embeddings = []
        for text in texts:
            embedding = self.get_embedding(text)
            if embedding:
                embeddings.append(embedding)
            else:
                # Fallback to zero vector if embedding fails
                embeddings.append([0.0] * VECTOR_DIMENSION)
        return embeddings

class AIService:
    """Service for AI-generated summaries."""
    
    def __init__(self, api_key: str):
        self.client = openai.OpenAI(api_key=api_key)
        self.model = CHAT_MODEL
    
    def generate_fit_summary(self, job_description: str, resume_text: str, candidate_name: str) -> str:
        """Generate candidate fit summary."""
        try:
            # Truncate texts if too long
            job_desc = job_description[:2000] + "..." if len(job_description) > 2000 else job_description
            resume = resume_text[:3000] + "..." if len(resume_text) > 3000 else resume_text
            
            prompt = f"""
Analyze why {candidate_name} would be a great fit for this role (max 150 words).

JOB DESCRIPTION:
{job_desc}

CANDIDATE RESUME:
{resume}

Focus on:
1. Relevant skills and experience alignment
2. Key achievements that match job requirements
3. Unique strengths that add value

Write in a professional, positive tone. Be specific about qualifications.

SUMMARY:"""
            
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": "You are an expert HR analyst."},
                    {"role": "user", "content": prompt}
                ],
                max_tokens=200,
                temperature=0.7
            )
            
            return response.choices[0].message.content.strip()
        except Exception as e:
            return f"Unable to generate summary: {str(e)}"


class UnifiedEmbeddingService:
    """Unified embedding service with automatic hierarchy: OpenAI → Cohere → HF → TF-IDF."""
    
    def __init__(self):
        # API Keys from Streamlit Secrets (secure server-side storage)
        try:
            self.OPENAI_API_KEY = st.secrets.get("OPENAI_API_KEY", "")
            self.COHERE_API_KEY = st.secrets.get("COHERE_API_KEY", "")
            self.HF_API_TOKEN = st.secrets.get("HF_API_TOKEN", "")
        except Exception:
            # Fallback if secrets not configured
            self.OPENAI_API_KEY = ""
            self.COHERE_API_KEY = ""
            self.HF_API_TOKEN = ""
        
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
    
    def _init_service_hierarchy(self):
        """Initialize services in priority order: OpenAI → Cohere → HuggingFace → TF-IDF."""
        services_tried = []
        
        # TIER 1: OpenAI API (Premium quality - highest priority)
        if self._init_openai():
            self.active_service = "openai"
            self.method = "openai_embeddings"
            self.dimension = 1536
            st.sidebar.success("🚀 Premium Mode: OpenAI API Active")
            return
        else:
            services_tried.append("OpenAI")
        
        # TIER 2: Cohere API (Excellent fallback)
        if self._init_cohere():
            self.active_service = "cohere"
            self.method = "cohere_embed_v3"
            self.dimension = 1024
            st.sidebar.success("✨ Professional Mode: Cohere API Active")
            return
        else:
            services_tried.append("Cohere")
        
        # TIER 3: Hugging Face Inference API (Good fallback)
        if self._init_huggingface():
            self.active_service = "huggingface"
            self.method = "hf_inference_api"
            self.dimension = 384
            st.sidebar.success("⚡ Enhanced Mode: Hugging Face API Active")
            return
        else:
            services_tried.append("Hugging Face")
        
        # TIER 4: Enhanced TF-IDF (Always works)
        self.active_service = "tfidf"
        self.method = "enhanced_tfidf"
        self.dimension = 2000
        st.sidebar.info(f"📊 Basic Mode: Enhanced TF-IDF Active (tried: {', '.join(services_tried)})")
    
    def _init_openai(self) -> bool:
        """Initialize OpenAI API."""
        try:
            if not self.OPENAI_API_KEY or self.OPENAI_API_KEY == "":
                return False
            
            self.openai_client = openai.OpenAI(api_key=self.OPENAI_API_KEY)
            
            # Quick test
            test_response = self.openai_client.embeddings.create(
                input="test",
                model="text-embedding-ada-002"
            )
            return True
        except Exception as e:
            return False
    
    def _init_cohere(self) -> bool:
        """Initialize Cohere API."""
        try:
            if not self.COHERE_API_KEY or self.COHERE_API_KEY == "":
                return False
            
            if cohere is None:
                return False
                
            self.cohere_client = cohere.Client(self.COHERE_API_KEY)
            
            # Quick test
            test_response = self.cohere_client.embed(
                texts=["test"],
                model="embed-english-v3.0",
                input_type="search_document"
            )
            return True
        except Exception as e:
            return False
    
    def _init_huggingface(self) -> bool:
        """Initialize Hugging Face Inference API."""
        try:
            if not self.HF_API_TOKEN or self.HF_API_TOKEN == "":
                return False
            
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
        except Exception as e:
            return False
    
    def _init_sentence_transformers(self) -> bool:
        """Initialize local SentenceTransformers."""
        try:
            from sentence_transformers import SentenceTransformer
            self.sentence_model = SentenceTransformer('all-MiniLM-L6-v2')
            return True
        except ImportError:
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
        elif self.active_service == "sentence_transformers":
            return self._get_sentence_embedding(processed_text)
        else:
            return self._get_tfidf_embedding(processed_text)
    
    def _get_openai_embedding(self, text: str) -> List[float]:
        """Get OpenAI embedding."""
        try:
            response = self.openai_client.embeddings.create(
                input=text,
                model="text-embedding-ada-002"
            )
            return response.data[0].embedding
        except Exception as e:
            return [0.0] * self.dimension
    
    def _get_cohere_embedding(self, text: str) -> List[float]:
        """Get Cohere embedding."""
        try:
            response = self.cohere_client.embed(
                texts=[text],
                model="embed-english-v3.0",
                input_type="search_document"
            )
            return response.embeddings[0]
        except Exception as e:
            return [0.0] * self.dimension
    
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
        except Exception as e:
            return [0.0] * self.dimension
    
    def _get_sentence_embedding(self, text: str) -> List[float]:
        """Get SentenceTransformer embedding."""
        try:
            embedding = self.sentence_model.encode(text)
            return embedding.tolist()
        except Exception as e:
            return [0.0] * self.dimension
    
    def _get_tfidf_embedding(self, text: str) -> List[float]:
        """Get TF-IDF embedding (fallback)."""
        if not hasattr(self, '_fitted') or not self._fitted:
            return [0.0] * self.dimension
        
        try:
            tfidf_matrix = self.vectorizer.transform([text])
            return tfidf_matrix.toarray()[0].tolist()
        except:
            return [0.0] * self.dimension
    
    def get_embeddings_batch(self, texts: List[str]) -> List[List[float]]:
        """Get embeddings for multiple texts."""
        processed_texts = [self._preprocess_text_advanced(text) for text in texts]
        
        if self.active_service == "openai":
            return self._get_openai_embeddings_batch(processed_texts)
        elif self.active_service == "cohere":
            return self._get_cohere_embeddings_batch(processed_texts)
        elif self.active_service == "huggingface":
            return self._get_hf_embeddings_batch(processed_texts)
        elif self.active_service == "sentence_transformers":
            return self._get_sentence_embeddings_batch(processed_texts)
        else:
            return self._get_tfidf_embeddings_batch(processed_texts)
    
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
        except Exception as e:
            return [[0.0] * self.dimension for _ in texts]
    
    def _get_cohere_embeddings_batch(self, texts: List[str]) -> List[List[float]]:
        """Get Cohere embeddings for batch."""
        try:
            response = self.cohere_client.embed(
                texts=texts,
                model="embed-english-v3.0",
                input_type="search_document"
            )
            return response.embeddings
        except Exception as e:
            return [[0.0] * self.dimension for _ in texts]
    
    def _get_hf_embeddings_batch(self, texts: List[str]) -> List[List[float]]:
        """Get HF embeddings for batch (with rate limiting)."""
        embeddings = []
        for text in texts:
            embedding = self._get_hf_embedding(text)
            embeddings.append(embedding)
            time.sleep(0.1)  # Rate limiting for free tier
        return embeddings
    
    def _get_sentence_embeddings_batch(self, texts: List[str]) -> List[List[float]]:
        """Get SentenceTransformer embeddings for batch."""
        try:
            embeddings = self.sentence_model.encode(texts)
            return [emb.tolist() for emb in embeddings]
        except Exception as e:
            return [[0.0] * self.dimension for _ in texts]
    
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
        except Exception as e:
            return [[0.0] * self.dimension for _ in texts]
    
    def get_job_and_candidate_embeddings(self, job_text: str, candidate_texts: List[str]) -> tuple:
        """Get embeddings for job and candidates together."""
        if self.active_service in ["openai", "cohere", "huggingface", "sentence_transformers"]:
            # For API services, process separately
            job_embedding = self.get_embedding(job_text)
            candidate_embeddings = [self.get_embedding(text) for text in candidate_texts]
            return job_embedding, candidate_embeddings
        else:
            # For TF-IDF, need to fit together
            all_texts = [job_text] + candidate_texts
            all_embeddings = self.get_embeddings_batch(all_texts)
            return all_embeddings[0], all_embeddings[1:]
    
    def _preprocess_text_advanced(self, text: str) -> str:
        """Advanced text preprocessing for better matching."""
        import re
        
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
    
    def get_info(self) -> dict:
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
            "sentence_transformers": {
                "method": "SentenceTransformers (Local)",
                "description": "High-quality semantic similarity using local transformer models"
            },
            "tfidf": {
                "method": "Enhanced TF-IDF + Domain Knowledge",
                "description": "Advanced keyword and phrase matching with professional preprocessing"
            }
        }
        
        return service_info.get(self.active_service, service_info["tfidf"])


class UnifiedAIService:
    """Unified AI service with automatic hierarchy: OpenAI → Enhanced Analysis."""
    
    def __init__(self):
        # Try to get OpenAI API key for premium summaries
        try:
            self.openai_api_key = st.secrets.get("OPENAI_API_KEY", "")
        except:
            self.openai_api_key = ""
        
        self.openai_client = None
        if self.openai_api_key:
            try:
                self.openai_client = openai.OpenAI(api_key=self.openai_api_key)
            except:
                self.openai_client = None
        
        self.job_domains = {
            'data': ['data', 'analytics', 'scientist', 'analysis', 'statistics', 'research'],
            'engineering': ['software', 'engineer', 'developer', 'programming', 'coding', 'technical'],
            'management': ['manager', 'lead', 'director', 'supervisor', 'management', 'leadership'],
            'marketing': ['marketing', 'sales', 'business', 'customer', 'growth', 'strategy'],
            'design': ['design', 'ui', 'ux', 'creative', 'visual', 'graphics'],
            'operations': ['operations', 'process', 'supply', 'logistics', 'efficiency'],
            'finance': ['finance', 'accounting', 'financial', 'budget', 'investment', 'risk'],
            'hr': ['human resources', 'hr', 'recruitment', 'talent', 'people', 'culture']
        }
    
    def generate_fit_summary(self, job_description: str, resume_text: str, candidate_name: str) -> str:
        """Generate candidate fit summary using OpenAI or enhanced analysis."""
        # Try OpenAI first for premium quality
        if self.openai_client:
            try:
                return self._generate_openai_summary(job_description, resume_text, candidate_name)
            except Exception:
                pass  # Fall back to enhanced analysis
        
        # Fallback to enhanced analysis
        try:
            return self._generate_enhanced_summary(job_description, resume_text, candidate_name)
        except Exception as e:
            return f"{candidate_name} demonstrates relevant qualifications for this position. The profile shows professional experience that aligns with several key requirements. Recommend detailed interview to assess specific fit and potential contribution to the role."
    
    def _generate_openai_summary(self, job_description: str, resume_text: str, candidate_name: str) -> str:
        """Generate OpenAI-powered summary."""
        # Truncate texts if too long
        job_desc = job_description[:2000] + "..." if len(job_description) > 2000 else job_description
        resume = resume_text[:3000] + "..." if len(resume_text) > 3000 else resume_text
        
        prompt = f"""
Analyze how well this candidate fits the job requirements. Provide a concise, professional summary.

JOB DESCRIPTION:
{job_desc}

CANDIDATE RESUME:
{resume}

Provide a 2-3 sentence analysis covering:
1. Key qualifications match
2. Relevant experience/skills
3. Overall fit assessment

Write in a professional, positive tone. Be specific about qualifications.

SUMMARY:"""
        
        response = self.openai_client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": "You are an expert HR analyst."},
                {"role": "user", "content": prompt}
            ],
            max_tokens=200,
            temperature=0.7
        )
        
        return response.choices[0].message.content.strip()
    
    def _generate_enhanced_summary(self, job_description: str, resume_text: str, candidate_name: str) -> str:
        """Generate enhanced analysis-based summary."""
        # Comprehensive analysis
        analysis = self._analyze_candidate_fit(job_description, resume_text, candidate_name)
        
        # Generate contextual summary
        summary_parts = []
        
        # Core fit assessment
        core_fit = self._assess_core_fit(analysis)
        summary_parts.append(core_fit)
        
        # Skills analysis
        skills_analysis = self._analyze_skills_match(analysis)
        if skills_analysis:
            summary_parts.append(skills_analysis)
        
        # Experience analysis
        exp_analysis = self._analyze_experience_match(analysis)
        if exp_analysis:
            summary_parts.append(exp_analysis)
        
        # Recommendations
        recommendations = self._generate_recommendations(analysis)
        if recommendations:
            summary_parts.append(recommendations)
        
        return " ".join(summary_parts)
    
    def _analyze_candidate_fit(self, job_desc: str, resume: str, name: str) -> dict:
        """Comprehensive candidate analysis."""
        job_lower = job_desc.lower()
        resume_lower = resume.lower()
        
        # Domain detection
        job_domain = self._detect_domain(job_lower)
        candidate_domains = self._detect_candidate_domains(resume_lower)
        
        # Skills extraction
        job_skills = self._extract_skills_comprehensive(job_lower)
        candidate_skills = self._extract_skills_comprehensive(resume_lower)
        
        # Experience level
        experience_level = self._extract_experience_level(resume_lower)
        required_experience = self._extract_required_experience(job_lower)
        
        # Seniority match
        job_seniority = self._detect_seniority_level(job_lower)
        candidate_seniority = self._detect_seniority_level(resume_lower)
        
        # Education and certifications
        education = self._extract_education(resume_lower)
        certifications = self._extract_certifications(resume_lower)
        
        return {
            'name': name,
            'job_domain': job_domain,
            'candidate_domains': candidate_domains,
            'job_skills': job_skills,
            'candidate_skills': candidate_skills,
            'matching_skills': set(job_skills) & set(candidate_skills),
            'experience_level': experience_level,
            'required_experience': required_experience,
            'job_seniority': job_seniority,
            'candidate_seniority': candidate_seniority,
            'education': education,
            'certifications': certifications,
            'domain_match': job_domain in candidate_domains,
            'skill_match_ratio': len(set(job_skills) & set(candidate_skills)) / max(len(job_skills), 1)
        }
    
    def _assess_core_fit(self, analysis: dict) -> str:
        """Generate core fit assessment."""
        name = analysis['name']
        skill_ratio = analysis['skill_match_ratio']
        domain_match = analysis['domain_match']
        
        if skill_ratio >= 0.7 and domain_match:
            return f"{name} presents an excellent fit for this position with exceptional alignment across core competencies and domain expertise."
        elif skill_ratio >= 0.5 and domain_match:
            return f"{name} demonstrates strong qualifications for this role with solid domain knowledge and relevant skill set."
        elif skill_ratio >= 0.3 or domain_match:
            return f"{name} shows promising potential for this position with transferable skills and relevant background."
        else:
            return f"{name} brings a unique perspective to this role with complementary skills that could add value to the team."
    
    def _analyze_skills_match(self, analysis: dict) -> str:
        """Analyze skills match."""
        matching_skills = list(analysis['matching_skills'])
        
        if len(matching_skills) >= 5:
            top_skills = matching_skills[:4]
            return f"Key competencies include {', '.join(top_skills[:-1])}, and {top_skills[-1]}, demonstrating technical proficiency in critical areas."
        elif len(matching_skills) >= 2:
            return f"Shows relevant experience in {' and '.join(matching_skills)}, providing a solid foundation for the role."
        elif len(matching_skills) == 1:
            return f"Brings valuable expertise in {matching_skills[0]}."
        else:
            return ""
    
    def _analyze_experience_match(self, analysis: dict) -> str:
        """Analyze experience match."""
        candidate_exp = analysis['experience_level']
        required_exp = analysis['required_experience']
        candidate_seniority = analysis['candidate_seniority']
        job_seniority = analysis['job_seniority']
        
        if candidate_exp >= required_exp and candidate_seniority == job_seniority:
            return f"Professional experience level aligns well with position requirements, demonstrating readiness for {job_seniority}-level responsibilities."
        elif candidate_exp >= required_exp:
            return f"Brings {candidate_exp}+ years of relevant experience, meeting the experience requirements for this position."
        elif candidate_exp > 0:
            return f"Has {candidate_exp} years of experience which provides a foundation for growth in this role."
        else:
            return "Represents an entry-level candidate with growth potential."
    
    def _generate_recommendations(self, analysis: dict) -> str:
        """Generate hiring recommendations."""
        skill_ratio = analysis['skill_match_ratio']
        domain_match = analysis['domain_match']
        
        if skill_ratio >= 0.6 and domain_match:
            return "Recommended for immediate consideration and next-round interviews."
        elif skill_ratio >= 0.4:
            return "Strong candidate worth pursuing for detailed technical assessment."
        elif skill_ratio >= 0.2:
            return "Consider for interview to explore potential and cultural fit."
        else:
            return "May benefit from additional screening to assess transferable value."
    
    def _extract_skills_comprehensive(self, text: str) -> List[str]:
        """Extract comprehensive skills."""
        skills_database = {
            # Programming Languages
            'python', 'java', 'javascript', 'typescript', 'c++', 'c#', 'php', 'ruby', 'go', 'rust', 'swift', 'kotlin',
            # Web Technologies
            'react', 'angular', 'vue', 'node', 'express', 'django', 'flask', 'spring', 'laravel',
            # Databases
            'sql', 'mysql', 'postgresql', 'mongodb', 'redis', 'cassandra', 'oracle', 'sqlite',
            # Cloud & DevOps
            'aws', 'azure', 'gcp', 'docker', 'kubernetes', 'jenkins', 'gitlab', 'terraform', 'ansible',
            # Data & ML
            'machine learning', 'deep learning', 'data science', 'analytics', 'tensorflow', 'pytorch', 'pandas', 'numpy',
            # Tools & Methodologies
            'git', 'agile', 'scrum', 'jira', 'confluence', 'slack', 'linux', 'windows', 'macos',
            # Business Skills
            'project management', 'leadership', 'communication', 'problem solving', 'teamwork', 'strategic planning'
        }
        
        found = []
        for skill in skills_database:
            if skill in text:
                found.append(skill)
        return found
    
    def _detect_domain(self, text: str) -> str:
        """Detect job domain."""
        for domain, keywords in self.job_domains.items():
            if any(keyword in text for keyword in keywords):
                return domain
        return 'general'
    
    def _detect_candidate_domains(self, text: str) -> List[str]:
        """Detect candidate domains."""
        domains = []
        for domain, keywords in self.job_domains.items():
            if any(keyword in text for keyword in keywords):
                domains.append(domain)
        return domains if domains else ['general']
    
    def _extract_experience_level(self, text: str) -> int:
        """Extract years of experience."""
        import re
        
        # Look for patterns like "5 years", "3+ years", etc.
        patterns = [
            r'(\d+)\+?\s*years?\s*(?:of\s*)?(?:experience|exp)',
            r'(\d+)\+?\s*(?:years?|yrs?)\s*experience',
            r'experience.*?(\d+)\+?\s*(?:years?|yrs?)',
        ]
        
        for pattern in patterns:
            matches = re.findall(pattern, text, re.IGNORECASE)
            if matches:
                return max(int(match) for match in matches)
        
        return 0
    
    def _extract_required_experience(self, text: str) -> int:
        """Extract required years of experience from job description."""
        import re
        
        patterns = [
            r'(?:minimum|min|at least|require[sd]?)\s*(\d+)\+?\s*(?:years?|yrs?)',
            r'(\d+)\+?\s*(?:years?|yrs?)\s*(?:minimum|min|required|experience)',
            r'(\d+)\+?\s*(?:years?|yrs?)\s*(?:of\s*)?(?:relevant\s*)?experience',
        ]
        
        for pattern in patterns:
            matches = re.findall(pattern, text, re.IGNORECASE)
            if matches:
                return max(int(match) for match in matches)
        
        return 0
    
    def _detect_seniority_level(self, text: str) -> str:
        """Detect seniority level."""
        senior_keywords = ['senior', 'lead', 'principal', 'staff', 'architect', 'manager', 'director']
        junior_keywords = ['junior', 'entry', 'associate', 'intern', 'trainee', 'graduate']
        
        if any(keyword in text for keyword in senior_keywords):
            return 'senior'
        elif any(keyword in text for keyword in junior_keywords):
            return 'junior'
        else:
            return 'mid'
    
    def _extract_education(self, text: str) -> List[str]:
        """Extract education information."""
        education_keywords = ['bachelor', 'master', 'phd', 'doctorate', 'degree', 'university', 'college', 'mba', 'bs', 'ms', 'ba', 'ma']
        found = []
        
        for keyword in education_keywords:
            if keyword in text:
                found.append(keyword)
        
        return found
    
    def _extract_certifications(self, text: str) -> List[str]:
        """Extract certifications."""
        cert_keywords = ['certified', 'certification', 'aws', 'azure', 'google', 'pmp', 'scrum master', 'cissp', 'comptia']
        found = []
        
        for keyword in cert_keywords:
            if keyword in text:
                found.append(keyword)
        
        return found

def process_uploaded_file(uploaded_file) -> str:
    """Process uploaded file and extract text."""
    if not uploaded_file:
        return ""
    
    content = uploaded_file.read()
    filename = uploaded_file.name.lower()
    
    processor = DocumentProcessor()
    
    if filename.endswith('.pdf'):
        return processor.extract_text_from_pdf(content)
    elif filename.endswith('.docx'):
        return processor.extract_text_from_docx(content)
    elif filename.endswith('.txt'):
        return processor.extract_text_from_txt(content)
    else:
        st.error(f"Unsupported file type: {filename}")
        return ""

def compute_similarity_scores(job_embedding: List[float], candidate_embeddings: List[List[float]], is_free_mode: bool = False) -> List[float]:
    """Compute cosine similarity scores."""
    if not job_embedding or not candidate_embeddings:
        return []
    
    job_vec = np.array(job_embedding).reshape(1, -1)
    candidate_vecs = np.array(candidate_embeddings)
    
    # Compute cosine similarity
    similarities = cosine_similarity(job_vec, candidate_vecs)[0]
    
    # Enhance scores for free mode to make them more meaningful
    if is_free_mode:
        # Apply square root transformation to spread out low scores
        similarities = np.sqrt(np.maximum(similarities, 0)) * 0.85  # Max ~85% for free mode
    
    # Ensure scores are between 0 and 1
    similarities = np.clip(similarities, 0, 1)
    
    return similarities.tolist()

def display_results(job_description: Dict, candidates: List[Dict], similarities: List[float], summaries: List[str]):
    """Display search results."""
    
    # Create results data
    results = []
    for i, (candidate, similarity) in enumerate(zip(candidates, similarities)):
        results.append({
            'rank': i + 1,
            'name': candidate['name'],
            'similarity': similarity,
            'resume': candidate['resume_text'],
            'summary': summaries[i] if i < len(summaries) else "Summary not available"
        })
    
    # Sort by similarity score
    results.sort(key=lambda x: x['similarity'], reverse=True)
    
    st.success(f"Processed {len(candidates)} candidates")
    
    if not results:
        st.warning("No matching candidates found.")
        return
    
    # Create tabs for different views
    tab1, tab2, tab3 = st.tabs(["📋 Detailed Results", "📊 Similarity Scores", "📈 Analysis"])
    
    with tab1:
        st.subheader(f"🏆 Top {len(results)} Candidates")
        
        for result in results:
            with st.expander(f"#{result['rank']} {result['name']} - Similarity: {result['similarity']:.1%}", expanded=result['rank'] <= 3):
                col1, col2 = st.columns([2, 1])
                
                with col1:
                    st.markdown(f"**Candidate:** {result['name']}")
                    st.markdown(f"**Similarity Score:** {result['similarity']:.1%}")
                    
                    if result['summary']:
                        st.markdown("**Why this candidate is a great fit:**")
                        st.markdown(f"*{result['summary']}*")
                    
                    # Resume preview button
                    if st.button(f"📄 View Resume", key=f"view_resume_{result['rank']}"):
                        resume_preview = result['resume'][:1000] + "..." if len(result['resume']) > 1000 else result['resume']
                        st.text_area(
                            "Resume Content",
                            value=resume_preview,
                            height=200,
                            disabled=True,
                            key=f"resume_{result['rank']}"
                        )
                
                with col2:
                    # Similarity score gauge
                    fig = go.Figure(go.Indicator(
                        mode="gauge+number",
                        value=result['similarity'] * 100,
                        domain={'x': [0, 1], 'y': [0, 1]},
                        title={'text': "Match %"},
                        gauge={
                            'axis': {'range': [None, 100]},
                            'bar': {'color': "darkgreen"},
                            'steps': [
                                {'range': [0, 50], 'color': "lightgray"},
                                {'range': [50, 80], 'color': "yellow"},
                                {'range': [80, 100], 'color': "lightgreen"}
                            ],
                            'threshold': {
                                'line': {'color': "red", 'width': 4},
                                'thickness': 0.75,
                                'value': 90
                            }
                        }
                    ))
                    fig.update_layout(height=200)
                    st.plotly_chart(fig, use_container_width=True)
    
    with tab2:
        # Similarity scores chart
        st.subheader("📊 Candidate Similarity Scores")
        
        df = pd.DataFrame([
            {
                "Candidate": result["name"],
                "Similarity Score": result["similarity"] * 100,
                "Rank": result["rank"]
            }
            for result in results
        ])
        
        fig = px.bar(
            df,
            x="Similarity Score",
            y="Candidate",
            orientation="h",
            title="Candidate Similarity Scores (%)",
            color="Similarity Score",
            color_continuous_scale="Viridis"
        )
        fig.update_layout(height=max(400, len(results) * 50))
        st.plotly_chart(fig, use_container_width=True)
        
        # Data table
        st.subheader("📋 Results Table")
        display_df = df[["Rank", "Candidate", "Similarity Score"]].copy()
        display_df["Similarity Score"] = display_df["Similarity Score"].round(1).astype(str) + "%"
        st.dataframe(display_df, use_container_width=True, hide_index=True)
    
    with tab3:
        # Analysis
        st.subheader("📈 Analysis & Insights")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            avg_score = sum(r["similarity"] for r in results) / len(results)
            st.metric("Average Similarity", f"{avg_score:.1%}")
        
        with col2:
            top_score = results[0]["similarity"] if results else 0
            st.metric("Best Match", f"{top_score:.1%}")
        
        with col3:
            strong_matches = sum(1 for r in results if r["similarity"] >= 0.7)
            st.metric("Strong Matches (≥70%)", strong_matches)
        
        # Score distribution
        if len(results) > 1:
            st.subheader("Score Distribution")
            scores = [r["similarity"] * 100 for r in results]
            fig = px.histogram(
                x=scores,
                nbins=min(10, len(results)),
                title="Distribution of Similarity Scores",
                labels={"x": "Similarity Score (%)", "y": "Number of Candidates"}
            )
            st.plotly_chart(fig, use_container_width=True)

def main():
    """Main Streamlit application."""
    
    # Header
    st.markdown("""
    <div class="main-header">
        Candidate Recommendation Engine
        <div style="font-size: 1.2rem; font-weight: 400; margin-top: 0.5rem; opacity: 0.9;">
            AI-powered semantic matching for intelligent hiring decisions
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    # Configuration options
    st.sidebar.header("Configuration")
    
    # Status indicator for active AI service tier
    st.sidebar.markdown("### 🎯 Active Service Tier")
    st.sidebar.caption("Automatic selection: Premium → Professional → Enhanced → Basic")
    
    # Initialize unified services automatically
    embedding_service = UnifiedEmbeddingService()
    ai_service = UnifiedAIService()
    
    # Show active service info
    service_info = embedding_service.get_info()
    st.sidebar.info(f"**{service_info['method']}**\n{service_info['description']}")
    

    
    # Sidebar settings
    with st.sidebar:
        st.header("Settings")
        
        method = st.selectbox(
            "Input Method",
            ["File Upload", "Text Input"],
            help="Choose how to provide candidate information"
        )
        
        st.subheader("Search Parameters")
        top_k = st.slider("Number of top candidates", 1, 20, 10)
        include_summary = st.checkbox("Include AI summaries", value=True)
    
    # Job Description
    st.markdown('<div class="clean-header">Job Description</div>', unsafe_allow_html=True)
    
    # Full width job title
    job_title = st.text_input(
        "Job Title", 
        placeholder="e.g., Senior Software Engineer",
        help="Enter the position title for the role you're hiring for"
    )
    
    # Job description and requirements in aligned columns (75% / 25%)
    col1, col2 = st.columns([3, 1])
    
    with col1:
        job_description = st.text_area(
            "Job Description",
            height=200,
            placeholder="Describe the role, responsibilities, required skills, and qualifications in detail...",
            help="Provide a comprehensive description of the position including key responsibilities, required skills, and desired qualifications"
        )
    
    with col2:
        job_requirements = st.text_area(
            "Additional Requirements",
            height=200,
            placeholder="Specific skills, experience, or qualifications...",
            help="Optional: Add any specific requirements, certifications, or preferred qualifications"
        )
    
    # Progress indicator and validation
    job_complete = bool(job_description.strip() and job_title.strip())
    
    # Add some spacing
    st.markdown("<br>", unsafe_allow_html=True)
    
    if job_complete:
        # Success indicator - more elegant and minimal
        st.markdown("""
        <div style="background: #F8FFF8; 
                    border-left: 4px solid #27AE60; 
                    border-radius: 0 8px 8px 0; 
                    padding: 1rem 1.5rem; 
                    margin: 1rem 0; 
                    color: #1E8449;">
            <strong>Job description complete</strong> • Ready to upload candidate resumes
        </div>
        """, unsafe_allow_html=True)
        
        # Elegant continue button
        st.markdown("<div style='margin: 1.5rem 0;'></div>", unsafe_allow_html=True)
        col_left, col_center, col_right = st.columns([2, 1, 2])
        with col_center:
            continue_clicked = st.button(
                "Continue →", 
                key="proceed_button",
                help="Proceed to upload candidate resumes",
                type="primary",
                use_container_width=True
            )
        
        # Only show upload section if continue button is clicked or if we're in session state
        if continue_clicked:
            st.session_state.show_upload_section = True
            
        if not st.session_state.get('show_upload_section', False):
            st.stop()  # Don't show anything below until button is clicked
            
    else:
        # Simple, clean warning indicator
        st.markdown("""
        <div style="background: #FFFAF0; 
                    border-left: 4px solid #F39C12; 
                    border-radius: 0 8px 8px 0; 
                    padding: 1rem 1.5rem; 
                    margin: 1rem 0; 
                    color: #D68910;">
            Please provide both job title and description to continue
        </div>
        """, unsafe_allow_html=True)
        st.stop()  # Don't show anything below until requirements are met
    
    # Section break
    st.markdown("<div style='margin: 2rem 0;'></div>", unsafe_allow_html=True)
    
    # Method-specific input
    candidates = []
    
    if method == "File Upload":
        st.markdown('<div class="clean-header">Upload Candidate Resumes</div>', unsafe_allow_html=True)
        
        uploaded_files = st.file_uploader(
            "Drop resume files here or click to browse",
            type=['pdf', 'docx', 'txt'],
            accept_multiple_files=True,
            help="Supported formats: PDF, DOCX, TXT • Upload multiple files at once"
        )
        
        if uploaded_files:
            if len(uploaded_files) > MAX_CANDIDATES:
                st.error(f"Too many files. Maximum allowed: {MAX_CANDIDATES}")
                return
            
            st.info(f"📄 {len(uploaded_files)} files uploaded")
            
            # Process files
            for file in uploaded_files:
                if file.size > MAX_FILE_SIZE:
                    st.warning(f"File {file.name} is too large (max {MAX_FILE_SIZE//1024//1024}MB)")
                    continue
                
                resume_text = process_uploaded_file(file)
                if resume_text.strip():
                    candidate_name = file.name.rsplit('.', 1)[0] if file.name else "Unknown"
                    candidates.append({
                        'name': candidate_name,
                        'resume_text': resume_text.strip()
                    })
    
    else:  # Text Input
        st.markdown('<div class="clean-header">Enter Candidate Information</div>', unsafe_allow_html=True)
        
        # Dynamic candidate input
        if "candidates_text" not in st.session_state:
            st.session_state.candidates_text = [{"name": "", "resume": ""}]
        
        # Add/Remove buttons
        col1, col2 = st.columns([1, 1])
        with col1:
            if st.button("➕ Add Candidate"):
                st.session_state.candidates_text.append({"name": "", "resume": ""})
        with col2:
            if st.button("➖ Remove Last") and len(st.session_state.candidates_text) > 1:
                st.session_state.candidates_text.pop()
        
        # Candidate input forms
        for i, candidate in enumerate(st.session_state.candidates_text):
            with st.expander(f"Candidate {i+1}", expanded=True):
                col1, col2 = st.columns([1, 3])
                
                with col1:
                    name = st.text_input(f"Name", key=f"name_{i}", value=candidate["name"])
                
                with col2:
                    resume = st.text_area(
                        f"Resume/CV Content",
                        height=100,
                        key=f"resume_{i}",
                        value=candidate["resume"],
                        placeholder="Paste the candidate's resume content here..."
                    )
                
                # Update session state
                st.session_state.candidates_text[i] = {"name": name, "resume": resume}
                
                # Add to candidates if both fields are filled
                if name.strip() and resume.strip():
                    candidates.append({
                        'name': name.strip(),
                        'resume_text': resume.strip()
                    })
    
    # Process candidates
    if candidates:
        st.info(f"{len(candidates)} candidates ready for analysis")
        
        if st.button("Analyze Candidates", type="primary"):
            with st.spinner("Generating embeddings and computing similarities..."):
                start_time = time.time()
                
                # Create job description text
                job_text = f"{job_title}\n\n{job_description}"
                if job_requirements:
                    job_text += f"\n\nRequirements:\n{job_requirements}"
                
                # Generate embeddings
                progress_bar = st.progress(0)
                
                # Generate embeddings
                progress_bar.progress(10)
                candidate_texts = [c['resume_text'] for c in candidates]
                
                # Use the unified embedding service
                progress_bar.progress(15)
                if hasattr(embedding_service, 'get_job_and_candidate_embeddings'):
                    progress_bar.progress(20)
                    job_embedding, candidate_embeddings = embedding_service.get_job_and_candidate_embeddings(
                        job_text, candidate_texts
                    )
                else:
                    # Fallback to individual embeddings
                    job_embedding = embedding_service.get_embedding(job_text)
                    progress_bar.progress(25)
                    candidate_embeddings = embedding_service.get_embeddings_batch(candidate_texts)
                
                # Compute similarities
                progress_bar.progress(60)
                # Check if we're using a basic service (TF-IDF) for score adjustment
                is_basic_mode = embedding_service.active_service == "tfidf"
                similarities = compute_similarity_scores(job_embedding, candidate_embeddings, is_free_mode=is_basic_mode)
                
                # Generate summaries
                summaries = []
                if include_summary:
                    progress_bar.progress(80)
                    for i, candidate in enumerate(candidates):
                        summary = ai_service.generate_fit_summary(
                            job_text, 
                            candidate['resume_text'], 
                            candidate['name']
                        )
                        summaries.append(summary)
                
                progress_bar.progress(100)
                processing_time = time.time() - start_time
                
                # Create job description dict
                job_desc = {
                    'title': job_title,
                    'description': job_description,
                    'requirements': job_requirements
                }
                
                # Display results
                st.balloons()
                display_results(job_desc, candidates, similarities, summaries)
                
                # Show processing time
                st.success(f"Processing completed in {processing_time:.2f} seconds")
    
    else:
        if method == "File Upload":
            st.info("Please upload resume files using the file uploader above")
        else:
            st.info("Please add at least one candidate with both name and resume content")

if __name__ == "__main__":
    main()
