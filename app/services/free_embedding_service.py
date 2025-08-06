"""
Free embedding service using open-source models as alternatives to OpenAI.
"""
import logging
from typing import List, Union
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import warnings
warnings.filterwarnings('ignore')

logger = logging.getLogger(__name__)

try:
    from sentence_transformers import SentenceTransformer
    SENTENCE_TRANSFORMERS_AVAILABLE = True
except ImportError:
    SENTENCE_TRANSFORMERS_AVAILABLE = False

try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False


class FreeEmbeddingService:
    """
    Free embedding service with multiple fallback options:
    1. SentenceTransformers (best quality, requires download)
    2. TF-IDF (fast, works offline)
    3. Simple word matching (fastest, basic)
    """
    
    def __init__(self, method="auto"):
        """
        Initialize the free embedding service.
        
        Args:
            method: "auto", "sentence-transformers", "tfidf", or "simple"
        """
        self.method = method
        self.model = None
        self.vectorizer = None
        self.dimension = 384  # Default dimension
        
        # Auto-select best available method
        if method == "auto":
            if SENTENCE_TRANSFORMERS_AVAILABLE and TORCH_AVAILABLE:
                self.method = "sentence-transformers"
            else:
                self.method = "tfidf"
        
        self._initialize_model()
    
    def _initialize_model(self):
        """Initialize the selected embedding model."""
        try:
            if self.method == "sentence-transformers":
                logger.info("Loading SentenceTransformers model (this may take a moment on first run)...")
                # Use a lightweight, multilingual model
                self.model = SentenceTransformer('all-MiniLM-L6-v2')
                self.dimension = 384
                logger.info("✅ SentenceTransformers model loaded successfully")
                
            elif self.method == "tfidf":
                # TF-IDF with optimized parameters
                self.vectorizer = TfidfVectorizer(
                    max_features=1000,  # Limit to 1000 features for speed
                    stop_words='english',
                    ngram_range=(1, 2),  # Include bigrams
                    min_df=1,
                    max_df=0.95
                )
                self.dimension = 1000
                logger.info("✅ TF-IDF vectorizer initialized")
                
            else:  # simple method
                self.dimension = 100  # Smaller dimension for simple matching
                logger.info("✅ Simple word matching initialized")
                
        except Exception as e:
            logger.error(f"Error initializing {self.method} model: {e}")
            # Fallback to TF-IDF
            if self.method != "tfidf":
                logger.info("Falling back to TF-IDF method...")
                self.method = "tfidf"
                self._initialize_model()
    
    def get_embedding(self, text: str) -> List[float]:
        """Get embedding for a single text."""
        try:
            cleaned_text = self._preprocess_text(text)
            
            if self.method == "sentence-transformers" and self.model:
                embedding = self.model.encode(cleaned_text)
                return embedding.tolist()
                
            elif self.method == "tfidf":
                # For single text with TF-IDF, we need to fit or use pre-fitted
                if not hasattr(self.vectorizer, 'vocabulary_'):
                    # If not fitted, return zero vector (will be handled in batch)
                    return [0.0] * self.dimension
                
                tfidf_matrix = self.vectorizer.transform([cleaned_text])
                dense_vector = tfidf_matrix.toarray()[0]
                return dense_vector.tolist()
                
            else:  # simple method
                return self._simple_text_embedding(cleaned_text)
                
        except Exception as e:
            logger.error(f"Error generating embedding: {e}")
            return [0.0] * self.dimension
    
    def get_embeddings_batch(self, texts: List[str]) -> List[List[float]]:
        """Get embeddings for multiple texts."""
        try:
            cleaned_texts = [self._preprocess_text(text) for text in texts]
            
            if self.method == "sentence-transformers" and self.model:
                embeddings = self.model.encode(cleaned_texts)
                return [emb.tolist() for emb in embeddings]
                
            elif self.method == "tfidf":
                # Fit TF-IDF on all texts
                tfidf_matrix = self.vectorizer.fit_transform(cleaned_texts)
                dense_matrix = tfidf_matrix.toarray()
                return [row.tolist() for row in dense_matrix]
                
            else:  # simple method
                return [self._simple_text_embedding(text) for text in cleaned_texts]
                
        except Exception as e:
            logger.error(f"Error generating batch embeddings: {e}")
            return [[0.0] * self.dimension for _ in texts]
    
    def _preprocess_text(self, text: str) -> str:
        """Preprocess text before embedding."""
        # Basic cleaning
        cleaned = " ".join(text.strip().split())
        
        # Truncate if too long
        max_chars = 5000  # Reasonable limit for free models
        if len(cleaned) > max_chars:
            cleaned = cleaned[:max_chars]
        
        return cleaned
    
    def _simple_text_embedding(self, text: str) -> List[float]:
        """
        Simple text embedding using basic NLP techniques.
        Creates a feature vector based on:
        - Word counts of important keywords
        - Text statistics
        - Simple patterns
        """
        text_lower = text.lower()
        
        # Technical skills keywords
        tech_keywords = [
            'python', 'java', 'javascript', 'react', 'node', 'sql', 'aws', 'docker',
            'kubernetes', 'machine learning', 'ai', 'data science', 'analytics',
            'frontend', 'backend', 'full stack', 'api', 'database', 'cloud',
            'agile', 'scrum', 'git', 'ci/cd', 'devops', 'linux', 'windows'
        ]
        
        # Experience indicators
        experience_keywords = [
            'years', 'experience', 'senior', 'lead', 'manager', 'director',
            'junior', 'intern', 'entry level', 'beginner', 'expert', 'specialist'
        ]
        
        # Education keywords
        education_keywords = [
            'degree', 'bachelor', 'master', 'phd', 'university', 'college',
            'certification', 'certified', 'course', 'training', 'bootcamp'
        ]
        
        # Create feature vector
        features = []
        
        # Tech skills features (50 dimensions)
        for keyword in tech_keywords[:50]:
            count = text_lower.count(keyword)
            features.append(min(count / 10.0, 1.0))  # Normalize
        
        # Pad to 50 if needed
        while len(features) < 50:
            features.append(0.0)
        
        # Experience features (25 dimensions)
        for keyword in experience_keywords[:25]:
            count = text_lower.count(keyword)
            features.append(min(count / 5.0, 1.0))
        
        # Education features (25 dimensions)
        for keyword in education_keywords[:25]:
            count = text_lower.count(keyword)
            features.append(min(count / 3.0, 1.0))
        
        # Ensure exactly 100 dimensions
        features = features[:100]
        while len(features) < 100:
            features.append(0.0)
        
        return features
    
    def compute_similarity(self, embedding1: List[float], embedding2: List[float]) -> float:
        """Compute cosine similarity between two embeddings."""
        try:
            vec1 = np.array(embedding1).reshape(1, -1)
            vec2 = np.array(embedding2).reshape(1, -1)
            
            similarity = cosine_similarity(vec1, vec2)[0][0]
            return max(0.0, min(1.0, float(similarity)))
            
        except Exception as e:
            logger.error(f"Error computing similarity: {e}")
            return 0.0
    
    def get_info(self) -> dict:
        """Get information about the current embedding method."""
        return {
            "method": self.method,
            "dimension": self.dimension,
            "description": {
                "sentence-transformers": "High-quality semantic embeddings using pre-trained models",
                "tfidf": "TF-IDF based embeddings with n-grams",
                "simple": "Basic keyword-based feature extraction"
            }.get(self.method, "Unknown method"),
            "pros": {
                "sentence-transformers": "Best quality, understands context and semantics",
                "tfidf": "Fast, works offline, good for keyword matching",
                "simple": "Fastest, works everywhere, no dependencies"
            }.get(self.method, ""),
            "requirements": {
                "sentence-transformers": "sentence-transformers library (~500MB download)",
                "tfidf": "scikit-learn (already included)",
                "simple": "No additional requirements"
            }.get(self.method, "")
        }


class FreeAIService:
    """
    Free AI service for generating summaries without OpenAI.
    Uses template-based generation and keyword analysis.
    """
    
    def __init__(self):
        """Initialize the free AI service."""
        pass
    
    def generate_fit_summary(self, job_description: str, resume_text: str, candidate_name: str) -> str:
        """
        Generate a candidate fit summary using keyword analysis and templates.
        """
        try:
            # Extract key information
            job_skills = self._extract_skills(job_description)
            candidate_skills = self._extract_skills(resume_text)
            experience_level = self._estimate_experience(resume_text)
            
            # Find matching skills
            matching_skills = set(job_skills) & set(candidate_skills)
            
            # Generate summary using templates
            summary = self._generate_template_summary(
                candidate_name, 
                matching_skills, 
                experience_level,
                job_skills,
                candidate_skills
            )
            
            return summary
            
        except Exception as e:
            logger.error(f"Error generating summary: {e}")
            return f"{candidate_name} appears to have relevant experience for this role. Manual review recommended to assess detailed qualifications and fit."
    
    def _extract_skills(self, text: str) -> List[str]:
        """Extract technical skills from text."""
        text_lower = text.lower()
        
        skills_db = [
            # Programming languages
            'python', 'java', 'javascript', 'typescript', 'c++', 'c#', 'php', 'ruby', 'go', 'rust',
            'swift', 'kotlin', 'scala', 'r', 'matlab', 'sql', 'html', 'css',
            
            # Frameworks & libraries
            'react', 'angular', 'vue', 'node.js', 'express', 'django', 'flask', 'spring',
            'laravel', 'rails', 'tensorflow', 'pytorch', 'scikit-learn', 'pandas', 'numpy',
            
            # Tools & platforms
            'aws', 'azure', 'gcp', 'docker', 'kubernetes', 'jenkins', 'git', 'jira',
            'linux', 'windows', 'macos', 'mysql', 'postgresql', 'mongodb', 'redis',
            
            # Methodologies
            'agile', 'scrum', 'kanban', 'devops', 'ci/cd', 'tdd', 'microservices',
            
            # Domains
            'machine learning', 'data science', 'web development', 'mobile development',
            'cloud computing', 'cybersecurity', 'blockchain', 'iot'
        ]
        
        found_skills = []
        for skill in skills_db:
            if skill in text_lower:
                found_skills.append(skill)
        
        return found_skills
    
    def _estimate_experience(self, resume_text: str) -> str:
        """Estimate experience level from resume."""
        text_lower = resume_text.lower()
        
        # Look for experience indicators
        senior_indicators = ['senior', 'lead', 'principal', 'architect', 'manager', 'director']
        junior_indicators = ['junior', 'intern', 'entry', 'associate', 'graduate']
        
        # Count years mentioned
        years_mentioned = []
        words = text_lower.split()
        for i, word in enumerate(words):
            if word in ['years', 'year'] and i > 0:
                prev_word = words[i-1]
                try:
                    years = int(prev_word)
                    years_mentioned.append(years)
                except:
                    pass
        
        # Determine level
        if any(indicator in text_lower for indicator in senior_indicators):
            return "Senior"
        elif any(indicator in text_lower for indicator in junior_indicators):
            return "Junior"
        elif years_mentioned and max(years_mentioned) >= 5:
            return "Experienced"
        elif years_mentioned and max(years_mentioned) >= 2:
            return "Mid-level"
        else:
            return "Entry-level"
    
    def _generate_template_summary(self, name: str, matching_skills: set, 
                                 experience_level: str, job_skills: List[str], 
                                 candidate_skills: List[str]) -> str:
        """Generate summary using templates."""
        
        # Calculate match percentage
        if job_skills:
            match_percentage = len(matching_skills) / len(set(job_skills)) * 100
        else:
            match_percentage = 0
        
        # Select template based on match quality
        if match_percentage >= 70:
            template = "excellent"
        elif match_percentage >= 50:
            template = "good"
        elif match_percentage >= 30:
            template = "moderate"
        else:
            template = "basic"
        
        # Generate summary
        templates = {
            "excellent": f"{name} is an excellent match for this position. As a {experience_level.lower()} professional, they demonstrate strong alignment with the required skills, particularly in {', '.join(list(matching_skills)[:3])}. Their background shows {match_percentage:.0f}% skill overlap with the job requirements, indicating they could contribute immediately to the team.",
            
            "good": f"{name} presents a solid candidacy for this role. With {experience_level.lower()} experience and skills in {', '.join(list(matching_skills)[:3])}, they show {match_percentage:.0f}% alignment with the position requirements. Their technical background appears well-suited for the responsibilities outlined.",
            
            "moderate": f"{name} shows potential for this position. Their {experience_level.lower()} background includes relevant experience in {', '.join(list(matching_skills)[:2]) if matching_skills else 'several technical areas'}. With {match_percentage:.0f}% skill overlap, they may require some additional training but could grow into the role effectively.",
            
            "basic": f"{name} has foundational qualifications that could be developed for this role. As a {experience_level.lower()} candidate, they demonstrate some relevant skills and show potential for growth. Additional assessment would help determine training needs and long-term fit."
        }
        
        summary = templates.get(template, templates["basic"])
        
        # Add specific skill mentions if available
        if matching_skills:
            key_skills = list(matching_skills)[:3]
            summary += f" Key strengths include {', '.join(key_skills)}."
        
        return summary


# Global free service instances
free_embedding_service = FreeEmbeddingService()
free_ai_service = FreeAIService()
