"""
Streamlit frontend for the Candidate Recommendation Engine.
"""
import os
import streamlit as st
import requests
import json
import time
import pandas as pd
from typing import List, Dict, Any
import plotly.express as px
import plotly.graph_objects as go
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent))
from config import settings

# Page configuration
st.set_page_config(
    page_title="Candidate Recommendation Engine",
    page_icon="🎯",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: 700;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .candidate-card {
        border: 1px solid #e0e0e0;
        border-radius: 10px;
        padding: 1rem;
        margin: 1rem 0;
        background-color: #f9f9f9;
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
    .ai-summary {
        font-style: italic;
        color: #555;
        margin-top: 0.5rem;
    }
</style>
""", unsafe_allow_html=True)


def get_api_url() -> str:
    """Get the API base URL."""
    # Check if we're in a Docker environment
    api_host = "api" if "API_HOST" in os.environ and os.environ.get("API_HOST") == "api" else "localhost"
    return f"http://{api_host}:{settings.api_port}"


def check_api_health() -> bool:
    """Check if the API is healthy."""
    try:
        response = requests.get(f"{get_api_url()}/health", timeout=5)
        return response.status_code == 200
    except Exception:
        return False


def upload_files_search(
    job_title: str,
    job_description: str,
    job_requirements: str,
    files: List[Any],
    top_k: int,
    include_summary: bool
) -> Dict[str, Any]:
    """Send files to the API for processing."""
    try:
        # Prepare files for upload
        files_data = []
        for file in files:
            files_data.append(("files", (file.name, file.getvalue(), file.type)))
        
        # Prepare form data
        data = {
            "job_title": job_title,
            "job_description": job_description,
            "job_requirements": job_requirements,
            "top_k": top_k,
            "include_summary": include_summary
        }
        
        # Send request
        response = requests.post(
            f"{get_api_url()}/upload-search",
            data=data,
            files=files_data,
            timeout=120  # 2 minutes timeout for processing
        )
        
        if response.status_code == 200:
            return response.json()
        else:
            st.error(f"API Error: {response.status_code} - {response.text}")
            return None
            
    except requests.exceptions.Timeout:
        st.error("Request timed out. Please try again with fewer files or simpler content.")
        return None
    except Exception as e:
        st.error(f"Error communicating with API: {str(e)}")
        return None


def text_input_search(
    job_description: Dict[str, str],
    candidates: List[Dict[str, str]],
    top_k: int,
    include_summary: bool
) -> Dict[str, Any]:
    """Send text input to the API for processing."""
    try:
        payload = {
            "job_description": job_description,
            "candidates": candidates,
            "top_k": top_k,
            "include_summary": include_summary
        }
        
        response = requests.post(
            f"{get_api_url()}/search",
            json=payload,
            timeout=120
        )
        
        if response.status_code == 200:
            return response.json()
        else:
            st.error(f"API Error: {response.status_code} - {response.text}")
            return None
            
    except requests.exceptions.Timeout:
        st.error("Request timed out. Please try again.")
        return None
    except Exception as e:
        st.error(f"Error communicating with API: {str(e)}")
        return None


def display_results(results: Dict[str, Any]):
    """Display search results in an attractive format."""
    if not results or "matches" not in results:
        st.error("No results to display")
        return
    
    matches = results["matches"]
    total_candidates = results["total_candidates"]
    processing_time = results["processing_time"]
    
    # Results summary
    st.success(f"✅ Processed {total_candidates} candidates in {processing_time:.2f} seconds")
    
    if not matches:
        st.warning("No matching candidates found.")
        return
    
    # Create tabs for different views
    tab1, tab2, tab3 = st.tabs(["📋 Detailed Results", "📊 Similarity Scores", "📈 Analysis"])
    
    with tab1:
        st.subheader(f"🏆 Top {len(matches)} Candidates")
        
        for i, match in enumerate(matches, 1):
            with st.expander(f"#{i} {match['candidate_name']} - Similarity: {match['similarity_score']:.1%}", expanded=i <= 3):
                col1, col2 = st.columns([2, 1])
                
                with col1:
                    st.markdown(f"**Candidate:** {match['candidate_name']}")
                    st.markdown(f"**Similarity Score:** {match['similarity_score']:.1%}")
                    
                    if match.get('ai_summary'):
                        st.markdown("**Why this candidate is a great fit:**")
                        st.markdown(f"*{match['ai_summary']}*")
                    
                    # Resume preview - using a button instead of nested expander
                    if st.button(f"📄 View Resume", key=f"view_resume_{i}"):
                        st.text_area(
                            "Resume Content",
                            value=match['resume_text'][:1000] + "..." if len(match['resume_text']) > 1000 else match['resume_text'],
                            height=200,
                            disabled=True,
                            key=f"resume_{i}"
                        )
                
                with col2:
                    # Similarity score gauge
                    fig = go.Figure(go.Indicator(
                        mode = "gauge+number",
                        value = match['similarity_score'] * 100,
                        domain = {'x': [0, 1], 'y': [0, 1]},
                        title = {'text': "Match %"},
                        gauge = {
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
        
        # Prepare data for chart
        df = pd.DataFrame([
            {
                "Candidate": match["candidate_name"],
                "Similarity Score": match["similarity_score"] * 100,
                "Rank": i
            }
            for i, match in enumerate(matches, 1)
        ])
        
        # Horizontal bar chart
        fig = px.bar(
            df,
            x="Similarity Score",
            y="Candidate",
            orientation="h",
            title="Candidate Similarity Scores (%)",
            color="Similarity Score",
            color_continuous_scale="Viridis"
        )
        fig.update_layout(height=max(400, len(matches) * 50))
        st.plotly_chart(fig, use_container_width=True)
        
        # Display data table
        st.subheader("📋 Results Table")
        display_df = df[["Rank", "Candidate", "Similarity Score"]].copy()
        display_df["Similarity Score"] = display_df["Similarity Score"].round(1).astype(str) + "%"
        st.dataframe(display_df, use_container_width=True, hide_index=True)
    
    with tab3:
        # Analysis and insights
        st.subheader("📈 Analysis & Insights")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            avg_score = sum(m["similarity_score"] for m in matches) / len(matches)
            st.metric("Average Similarity", f"{avg_score:.1%}")
        
        with col2:
            top_score = matches[0]["similarity_score"] if matches else 0
            st.metric("Best Match", f"{top_score:.1%}")
        
        with col3:
            candidates_above_70 = sum(1 for m in matches if m["similarity_score"] >= 0.7)
            st.metric("Strong Matches (≥70%)", candidates_above_70)
        
        # Score distribution
        st.subheader("Score Distribution")
        scores = [m["similarity_score"] * 100 for m in matches]
        fig = px.histogram(
            x=scores,
            nbins=10,
            title="Distribution of Similarity Scores",
            labels={"x": "Similarity Score (%)", "y": "Number of Candidates"}
        )
        st.plotly_chart(fig, use_container_width=True)


def main():
    """Main Streamlit application."""
    
    # Header
    st.markdown('<div class="main-header">🎯 Candidate Recommendation Engine</div>', unsafe_allow_html=True)
    
    # Check API health
    if not check_api_health():
        st.error("⚠️ API is not available. Please ensure the FastAPI server is running.")
        st.info(f"Expected API URL: {get_api_url()}")
        st.stop()
    
    st.success("✅ Connected to API successfully!")
    
    # Sidebar for configuration
    with st.sidebar:
        st.header("⚙️ Settings")
        
        # Method selection
        method = st.selectbox(
            "Input Method",
            ["📁 File Upload", "✏️ Text Input"],
            help="Choose how to provide candidate information"
        )
        
        # Search parameters
        st.subheader("Search Parameters")
        top_k = st.slider("Number of top candidates", 1, 20, 10)
        include_summary = st.checkbox("Include AI summaries", value=True, help="Generate AI explanations for why candidates are good fits")
        
        # API stats
        if st.button("📊 View API Stats"):
            try:
                response = requests.get(f"{get_api_url()}/stats")
                if response.status_code == 200:
                    stats = response.json()
                    st.json(stats)
            except Exception as e:
                st.error(f"Failed to get stats: {e}")
    
    # Main content area
    st.header("📝 Job Description")
    
    # Job information input
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
        st.stop()
    
    # Method-specific input
    if method == "📁 File Upload":
        st.header("📁 Upload Candidate Resumes")
        
        uploaded_files = st.file_uploader(
            "Choose resume files",
            type=['pdf', 'docx', 'txt'],
            accept_multiple_files=True,
            help="Upload PDF, DOCX, or TXT files containing candidate resumes"
        )
        
        if uploaded_files:
            st.info(f"📄 {len(uploaded_files)} files uploaded")
            
            # Show file preview
            with st.expander("📋 Uploaded Files Preview"):
                for file in uploaded_files:
                    st.write(f"• {file.name} ({file.size} bytes)")
            
            if st.button("🔍 Analyze Candidates", type="primary"):
                if len(uploaded_files) > settings.max_candidates:
                    st.error(f"Too many files. Maximum allowed: {settings.max_candidates}")
                    st.stop()
                
                with st.spinner("🔄 Processing resumes and generating recommendations..."):
                    results = upload_files_search(
                        job_title,
                        job_description,
                        job_requirements,
                        uploaded_files,
                        top_k,
                        include_summary
                    )
                
                if results:
                    display_results(results)
    
    else:  # Text Input
        st.header("✏️ Enter Candidate Information")
        
        # Dynamic candidate input
        if "candidates" not in st.session_state:
            st.session_state.candidates = [{"name": "", "resume": ""}]
        
        # Add/Remove candidate buttons
        col1, col2 = st.columns([1, 1])
        with col1:
            if st.button("➕ Add Candidate"):
                st.session_state.candidates.append({"name": "", "resume": ""})
        with col2:
            if st.button("➖ Remove Last") and len(st.session_state.candidates) > 1:
                st.session_state.candidates.pop()
        
        # Candidate input forms
        valid_candidates = []
        for i, candidate in enumerate(st.session_state.candidates):
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
                st.session_state.candidates[i] = {"name": name, "resume": resume}
                
                # Add to valid candidates if both fields are filled
                if name.strip() and resume.strip():
                    valid_candidates.append({"name": name.strip(), "resume_text": resume.strip()})
        
        if valid_candidates:
            st.info(f"✅ {len(valid_candidates)} candidates ready for analysis")
            
            if st.button("🔍 Analyze Candidates", type="primary"):
                with st.spinner("🔄 Generating embeddings and computing similarities..."):
                    job_desc = {
                        "title": job_title,
                        "description": job_description,
                        "requirements": job_requirements if job_requirements else None
                    }
                    
                    results = text_input_search(
                        job_desc,
                        valid_candidates,
                        top_k,
                        include_summary
                    )
                
                if results:
                    display_results(results)
        else:
            st.warning("Please add at least one candidate with both name and resume content.")


if __name__ == "__main__":
    main()
