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


class FreeEmbeddingService:
    """Free embedding service using TF-IDF and keyword matching."""
    
    def __init__(self):
        self.vectorizer = None
        self.method = "tfidf"
        self.dimension = 2000  # Match max_features
    
    def get_embedding(self, text: str) -> List[float]:
        """Get embedding for single text."""
        if not hasattr(self, '_fitted') or not self._fitted:
            return [0.0] * self.dimension
        
        try:
            tfidf_matrix = self.vectorizer.transform([text])
            return tfidf_matrix.toarray()[0].tolist()
        except:
            return [0.0] * self.dimension
    
    def get_embeddings_batch(self, texts: List[str]) -> List[List[float]]:
        """Get embeddings for multiple texts using TF-IDF."""
        try:
            from sklearn.feature_extraction.text import TfidfVectorizer
            
            # Enhanced TF-IDF with better parameters for resumes
            self.vectorizer = TfidfVectorizer(
                max_features=2000,  # More features for better granularity
                stop_words='english',
                ngram_range=(1, 3),  # Include trigrams for better context
                min_df=1,
                max_df=0.85,  # More restrictive to filter common words
                sublinear_tf=True,  # Use log scaling
                norm='l2'  # L2 normalization for better cosine similarity
            )
            
            # Fit and transform
            tfidf_matrix = self.vectorizer.fit_transform(texts)
            self._fitted = True
            
            # Convert to list of lists
            dense_matrix = tfidf_matrix.toarray()
            return [row.tolist() for row in dense_matrix]
            
        except Exception as e:
            st.error(f"Error generating embeddings: {e}")
            return [[0.0] * self.dimension for _ in texts]
    
    def get_job_and_candidate_embeddings(self, job_text: str, candidate_texts: List[str]) -> tuple[List[float], List[List[float]]]:
        """
        Get embeddings for job and candidates together (required for TF-IDF).
        This ensures they're in the same vector space.
        """
        try:
            # Combine job + candidates for fitting
            all_texts = [job_text] + candidate_texts
            all_embeddings = self.get_embeddings_batch(all_texts)
            
            # Split back into job and candidates
            job_embedding = all_embeddings[0]
            candidate_embeddings = all_embeddings[1:]
            
            return job_embedding, candidate_embeddings
            
        except Exception as e:
            st.error(f"Error generating embeddings: {e}")
            # Fallback to simple method
            job_embedding = [0.1] * self.dimension  # Small positive values
            candidate_embeddings = [[0.1] * self.dimension for _ in candidate_texts]
            return job_embedding, candidate_embeddings

    def get_info(self) -> dict:
        return {
            "method": "TF-IDF + N-grams",
            "description": "Fast keyword-based similarity using TF-IDF vectorization"
        }


class FreeAIService:
    """Free AI service using template-based summaries."""
    
    def generate_fit_summary(self, job_description: str, resume_text: str, candidate_name: str) -> str:
        """Generate template-based summary."""
        try:
            # Extract skills
            job_skills = self._extract_skills(job_description.lower())
            candidate_skills = self._extract_skills(resume_text.lower())
            
            # Find matches
            matching_skills = set(job_skills) & set(candidate_skills)
            match_percentage = len(matching_skills) / max(len(job_skills), 1) * 100
            
            # Generate summary
            if match_percentage >= 60:
                quality = "excellent"
            elif match_percentage >= 40:
                quality = "good"
            elif match_percentage >= 20:
                quality = "moderate"
            else:
                quality = "basic"
            
            templates = {
                "excellent": f"{candidate_name} shows excellent alignment with this position, with strong skills in {', '.join(list(matching_skills)[:3])}. Their background demonstrates {match_percentage:.0f}% skill overlap with the requirements.",
                "good": f"{candidate_name} presents a solid match for this role, with relevant experience in {', '.join(list(matching_skills)[:2])}. Shows {match_percentage:.0f}% alignment with the position requirements.",
                "moderate": f"{candidate_name} has foundational qualifications with some overlap in {', '.join(list(matching_skills)[:2]) if matching_skills else 'technical areas'}. May require additional training but shows potential.",
                "basic": f"{candidate_name} demonstrates basic qualifications for this role. Further assessment recommended to determine specific training needs and development opportunities."
            }
            
            return templates[quality]
            
        except Exception as e:
            return f"{candidate_name} appears to have relevant experience. Manual review recommended."
    
    def _extract_skills(self, text: str) -> List[str]:
        """Extract technical skills."""
        skills = [
            'python', 'java', 'javascript', 'react', 'node', 'sql', 'aws', 'docker',
            'machine learning', 'data science', 'ai', 'analytics', 'api', 'database',
            'cloud', 'agile', 'scrum', 'git', 'linux', 'frontend', 'backend'
        ]
        
        found = []
        for skill in skills:
            if skill in text:
                found.append(skill)
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
    
    # Choose AI service
    ai_service_option = st.sidebar.selectbox(
        "AI Service",
        ["OpenAI (Recommended - Best Quality)", "Free Mode (Good Alternative)"],
        help="OpenAI provides superior semantic understanding and professional summaries. Free mode is a solid backup option."
    )
    
    use_openai = "OpenAI" in ai_service_option
    api_key = None
    
    if use_openai:
        api_key = st.sidebar.text_input(
            "OpenAI API Key",
            type="password",
            help="Enter your OpenAI API key. Get one at https://platform.openai.com/api-keys"
        )
        
        if not api_key:
            st.warning("⚠️ Please enter your OpenAI API key in the sidebar to use the recommended OpenAI service.")
            st.info("""
            **Why OpenAI is Recommended:**
            - **Superior Quality**: Best-in-class semantic understanding
            - **Professional Summaries**: Human-like analysis of candidate fit
            - **Proven Accuracy**: Industry-leading AI for HR applications
            - **Cost**: Only ~$0.002 per candidate analysis
            
            **🔗 Get Started:**
            1. Get an OpenAI API key from https://platform.openai.com/api-keys
            2. Enter it in the sidebar above
            
            **Alternative**: Switch to "Free Mode" above if you prefer no-cost operation (good quality, but not as sophisticated)
            """)
            return
    else:
        # Free mode selected
        st.sidebar.success("Free Mode Selected")
        st.sidebar.info("""
        **Free Mode Features:**
        - No API key required
        - Uses TF-IDF + keyword matching
        - Works completely offline
        - Good quality for basic screening
        
        **Recommendation**: For professional use, 
        switch to OpenAI mode above for significantly 
        better semantic understanding and summaries.
        """)
        
        # Show which free method is being used
        free_embedding_service = FreeEmbeddingService()
        method_info = free_embedding_service.get_info()
        st.sidebar.write(f"**Method:** {method_info['method']}")
        st.sidebar.write(f"**Quality:** {method_info['description']}")
    
    # Initialize services based on mode
    try:
        if use_openai:
            embedding_service = EmbeddingService(api_key)
            ai_service = AIService(api_key)
            st.sidebar.success("OpenAI connection ready!")
        else:
            # Use free services
            embedding_service = FreeEmbeddingService()
            ai_service = FreeAIService()
            st.sidebar.success("Free AI services ready!")
    except Exception as e:
        if use_openai:
            st.sidebar.error(f"OpenAI connection failed: {e}")
        else:
            st.sidebar.error(f"Free services failed: {e}")
        return
    
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
                
                # For free mode, use special method that fits TF-IDF on all texts together
                if not use_openai and hasattr(embedding_service, 'get_job_and_candidate_embeddings'):
                    progress_bar.progress(20)
                    job_embedding, candidate_embeddings = embedding_service.get_job_and_candidate_embeddings(
                        job_text, candidate_texts
                    )
                else:
                    # OpenAI mode - get embeddings separately
                    progress_bar.progress(15)
                    job_embedding = embedding_service.get_embedding(job_text)
                    progress_bar.progress(25)
                    candidate_embeddings = embedding_service.get_embeddings_batch(candidate_texts)
                
                # Compute similarities
                progress_bar.progress(60)
                similarities = compute_similarity_scores(job_embedding, candidate_embeddings, is_free_mode=not use_openai)
                
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
