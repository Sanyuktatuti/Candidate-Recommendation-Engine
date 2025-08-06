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
import faiss
from sklearn.metrics.pairwise import cosine_similarity

# Set page config
st.set_page_config(
    page_title="Candidate Recommendation Engine",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: 700;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .similarity-score {
        font-size: 1.2rem;
        font-weight: bold;
        color: #2e8b57;
    }
    .candidate-name {
        font-size: 1.1rem;
        font-weight: 600;
        color: #333;
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
    st.markdown('<div class="main-header">Candidate Recommendation Engine</div>', unsafe_allow_html=True)
    
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
        - Uses FastAPI backend with TF-IDF
        - Works completely offline
        - Good quality for basic screening
        
        **Recommendation**: For professional use, 
        switch to OpenAI mode above for significantly 
        better semantic understanding and summaries.
        """)
    
    # Initialize services based on mode
    try:
        if use_openai:
            embedding_service = EmbeddingService(api_key)
            ai_service = AIService(api_key)
            st.sidebar.success("OpenAI connection ready!")
        else:
            # Use free services
            from app.services.free_embedding_service import free_embedding_service, free_ai_service
            embedding_service = free_embedding_service
            ai_service = free_ai_service
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
    st.header("Job Description")
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        job_title = st.text_input("Job Title", placeholder="e.g., Senior Software Engineer")
        job_description = st.text_area(
            "Job Description",
            height=150,
            placeholder="Describe the role, responsibilities, and what you're looking for..."
        )
    
    with col2:
        job_requirements = st.text_area(
            "Requirements (Optional)",
            height=150,
            placeholder="Specific skills, experience, or qualifications required..."
        )
    
    # Input validation
    if not job_title or not job_description:
        st.warning("Please provide both job title and description to proceed.")
        return
    
    # Method-specific input
    candidates = []
    
    if method == "File Upload":
        st.header("📁 Upload Candidate Resumes")
        
        uploaded_files = st.file_uploader(
            "Choose resume files",
            type=['pdf', 'docx', 'txt'],
            accept_multiple_files=True,
            help="Upload PDF, DOCX, or TXT files containing candidate resumes"
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
        st.header("✏️ Enter Candidate Information")
        
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
                st.success(f"⚡ Processing completed in {processing_time:.2f} seconds")
    
    else:
        if method == "File Upload":
            st.info("👆 Please upload resume files using the file uploader above")
        else:
            st.info("👆 Please add at least one candidate with both name and resume content")

if __name__ == "__main__":
    main()
