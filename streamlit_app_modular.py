"""
Modular Streamlit app for Streamlit Community Cloud deployment.
Clean, maintainable architecture with separated concerns.
"""

import sys
import os

# Add src directory to Python path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

import streamlit as st
from typing import List

# Import our modular components
from src.models.candidate import Candidate, JobDescription
from src.services.embedding_service import UnifiedEmbeddingService
from src.services.ai_service import UnifiedAIService
from src.utils.result_processor import ResultProcessor
from src.ui.styles import get_custom_css
from src.ui.components import UIComponents
from src.ui.display import ResultDisplay

# Configuration
MAX_CANDIDATES = 20
MAX_FILE_SIZE = 10 * 1024 * 1024  # 10MB

# Set page config
st.set_page_config(
    page_title="Candidate Recommendation Engine",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Apply custom CSS
st.markdown(get_custom_css(), unsafe_allow_html=True)


class CandidateRecommendationApp:
    """Main application class for the Candidate Recommendation Engine."""
    
    def __init__(self):
        """Initialize the application."""
        self.ui_components = UIComponents()
        self.result_display = ResultDisplay()
        
        # Initialize services
        self.embedding_service = UnifiedEmbeddingService()
        self.ai_service = UnifiedAIService()
        self.result_processor = ResultProcessor(
            embedding_service=self.embedding_service,
            ai_service=self.ai_service
        )
    
    def run(self) -> None:
        """Run the main application."""
        # Render header
        self.ui_components.render_header()
        
        # Render sidebar settings
        method, top_k, include_summary = self.ui_components.render_sidebar_settings()
        
        # Job description form
        job, job_complete = self.ui_components.render_job_description_form()
        
        if not job_complete:
            st.stop()  # Don't show anything below until requirements are met
        
        # Only show upload section if continue button is clicked or if we're in session state
        if not st.session_state.get('show_upload_section', False):
            st.stop()  # Don't show anything below until button is clicked
        
        # Section break
        st.markdown("<div style='margin: 2rem 0;'></div>", unsafe_allow_html=True)
        
        # Candidate input based on method
        candidates = self._get_candidates_by_method(method)
        
        # Process candidates if available
        if candidates:
            self._process_and_display_results(job, candidates, include_summary, top_k)
        else:
            self._show_input_instructions(method)
    
    def _get_candidates_by_method(self, method: str) -> List[Candidate]:
        """Get candidates based on the selected input method.
        
        Args:
            method: Input method ("File Upload" or "Text Input")
            
        Returns:
            List of candidates
        """
        if method == "File Upload":
            return self.ui_components.render_file_upload_section()
        else:
            return self.ui_components.render_text_input_section()
    
    def _process_and_display_results(
        self, 
        job: JobDescription, 
        candidates: List[Candidate], 
        include_summary: bool, 
        top_k: int
    ) -> None:
        """Process candidates and display results.
        
        Args:
            job: Job description
            candidates: List of candidates
            include_summary: Whether to include AI summaries
            top_k: Number of top candidates to show
        """
        self.ui_components.render_status_message(
            f"{len(candidates)} candidates ready for analysis", 
            "info"
        )
        
        if st.button("Analyze Candidates", type="primary"):
            with st.spinner("Generating embeddings and computing similarities..."):
                try:
                    # Process candidates
                    search_result = self.result_processor.process_candidates(
                        job=job,
                        candidates=candidates,
                        include_summaries=include_summary,
                        top_k=top_k
                    )
                    
                    # Display results
                    st.balloons()
                    self.result_display.display_search_results(search_result, top_k)
                    
                except Exception as e:
                    st.error(f"An error occurred during processing: {str(e)}")
    
    def _show_input_instructions(self, method: str) -> None:
        """Show instructions for input method.
        
        Args:
            method: Input method
        """
        if method == "File Upload":
            self.ui_components.render_status_message(
                "Please upload resume files using the file uploader above",
                "info"
            )
        else:
            self.ui_components.render_status_message(
                "Please add at least one candidate with both name and resume content",
                "info"
            )


def main():
    """Main application entry point."""
    app = CandidateRecommendationApp()
    app.run()


if __name__ == "__main__":
    main()
