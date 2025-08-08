"""
UI components for the Streamlit application.
"""

from typing import List, Tuple, Optional
import streamlit as st
from ..models.candidate import Candidate, JobDescription
from ..services.document_processor import DocumentProcessor


class UIComponents:
    """Reusable UI components for the Streamlit application."""
    
    def __init__(self):
        """Initialize UI components."""
        self.document_processor = DocumentProcessor()
    
    def render_header(self) -> None:
        """Render the main application header."""
        st.markdown("""
        <div class="main-header">
            Candidate Recommendation Engine
            <div style="font-size: 1.2rem; font-weight: 400; margin-top: 0.5rem; opacity: 0.9;">
                AI-powered semantic matching for intelligent hiring decisions
            </div>
        </div>
        """, unsafe_allow_html=True)
    
    def render_sidebar_settings(self) -> Tuple[str, int, bool]:
        """Render sidebar settings and return configuration values.
        
        Returns:
            Tuple of (method, top_k, include_summary)
        """
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
            
            return method, top_k, include_summary
    
    def render_job_description_form(self) -> Tuple[Optional[JobDescription], bool]:
        """Render job description form.
        
        Returns:
            Tuple of (JobDescription if complete, is_complete)
        """
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
            # Success indicator
            st.markdown("""
            <div class="status-success">
                <strong>Job description complete</strong> • Ready to upload candidate resumes
            </div>
            """, unsafe_allow_html=True)
            
            # Continue button
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
            
            # Set session state
            if continue_clicked:
                st.session_state.show_upload_section = True
            
            job_desc = JobDescription(
                title=job_title.strip(),
                description=job_description.strip(),
                requirements=job_requirements.strip() if job_requirements.strip() else None
            )
            
            return job_desc, job_complete
        else:
            # Warning indicator
            st.markdown("""
            <div class="status-warning">
                Please provide both job title and description to continue
            </div>
            """, unsafe_allow_html=True)
            
            return None, False
    
    def render_file_upload_section(self) -> List[Candidate]:
        """Render file upload section and return candidates.
        
        Returns:
            List of candidates from uploaded files
        """
        st.markdown('<div class="clean-header">Upload Candidate Resumes</div>', unsafe_allow_html=True)
        
        uploaded_files = st.file_uploader(
            "Drop resume files here or click to browse",
            type=self.document_processor.get_supported_extensions(),
            accept_multiple_files=True,
            help="Supported formats: PDF, DOCX, TXT • Upload multiple files at once"
        )
        
        candidates = []
        
        if uploaded_files:
            if len(uploaded_files) > 20:  # MAX_CANDIDATES
                st.error(f"Too many files. Maximum allowed: 20")
                return candidates
            
            st.info(f"📄 {len(uploaded_files)} files uploaded")
            
            # Process files
            for file in uploaded_files:
                if file.size > 10 * 1024 * 1024:  # MAX_FILE_SIZE
                    st.warning(f"File {file.name} is too large (max 10MB)")
                    continue
                
                resume_text = self.document_processor.process_uploaded_file(file)
                if resume_text.strip():
                    candidate_name = file.name.rsplit('.', 1)[0] if file.name else "Unknown"
                    try:
                        candidate = Candidate(
                            name=candidate_name,
                            resume_text=resume_text.strip()
                        )
                        candidates.append(candidate)
                    except ValueError as e:
                        st.error(f"Error processing {file.name}: {e}")
        
        return candidates
    
    def render_text_input_section(self) -> List[Candidate]:
        """Render text input section and return candidates.
        
        Returns:
            List of candidates from text input
        """
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
        
        candidates = []
        
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
                    try:
                        candidate_obj = Candidate(
                            name=name.strip(),
                            resume_text=resume.strip()
                        )
                        candidates.append(candidate_obj)
                    except ValueError as e:
                        st.error(f"Error with candidate {i+1}: {e}")
        
        return candidates
    
    def render_progress_indicator(self, current_step: int, total_steps: int, message: str) -> None:
        """Render a progress indicator.
        
        Args:
            current_step: Current step number
            total_steps: Total number of steps
            message: Progress message to display
        """
        progress = current_step / total_steps
        st.progress(progress)
        st.text(f"Step {current_step}/{total_steps}: {message}")
    
    def render_status_message(self, message: str, status_type: str = "info") -> None:
        """Render a status message.
        
        Args:
            message: Message to display
            status_type: Type of status (success, warning, error, info)
        """
        if status_type == "success":
            st.success(message)
        elif status_type == "warning":
            st.warning(message)
        elif status_type == "error":
            st.error(message)
        else:
            st.info(message)
