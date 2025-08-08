"""
Result display components for the Streamlit application.
"""

import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from typing import List
from ..models.candidate import SearchResult


class ResultDisplay:
    """Handles display of search results and analytics."""
    
    @staticmethod
    def display_search_results(search_result: SearchResult, top_k: int = 10) -> None:
        """Display comprehensive search results.
        
        Args:
            search_result: The search result to display
            top_k: Number of top candidates to show
        """
        ranked_candidates = search_result.ranked_candidates[:top_k]
        
        if not ranked_candidates:
            st.warning("No matching candidates found.")
            return
        
        st.success(f"Processed {len(search_result.candidates)} candidates")
        
        # Create tabs for different views
        tab1, tab2, tab3 = st.tabs(["📋 Detailed Results", "📊 Similarity Scores", "📈 Analysis"])
        
        with tab1:
            ResultDisplay._display_detailed_results(ranked_candidates)
        
        with tab2:
            ResultDisplay._display_similarity_scores(ranked_candidates)
        
        with tab3:
            ResultDisplay._display_analysis(search_result)
        
        # Show processing time
        st.success(f"Processing completed in {search_result.processing_time:.2f} seconds")
    
    @staticmethod
    def _display_detailed_results(candidates: List) -> None:
        """Display detailed candidate results."""
        st.subheader(f"Top {len(candidates)} Candidates")
        
        for i, candidate in enumerate(candidates, 1):
            similarity_pct = candidate.similarity_score * 100 if candidate.similarity_score else 0
            
            with st.expander(
                f"#{i} {candidate.name} - Similarity: {similarity_pct:.1f}%", 
                expanded=i <= 3
            ):
                col1, col2 = st.columns([2, 1])
                
                with col1:
                    st.markdown(f"**Candidate:** {candidate.name}")
                    st.markdown(f"**Similarity Score:** {similarity_pct:.1f}%")
                    
                    if candidate.fit_summary:
                        st.markdown("**Why this candidate is a great fit:**")
                        st.markdown(f"*{candidate.fit_summary}*")
                    
                    # Resume preview button
                    if st.button(f"📄 View Resume", key=f"view_resume_{i}"):
                        resume_preview = (candidate.resume_text[:1000] + "..." 
                                        if len(candidate.resume_text) > 1000 
                                        else candidate.resume_text)
                        st.text_area(
                            "Resume Content",
                            value=resume_preview,
                            height=200,
                            disabled=True,
                            key=f"resume_{i}"
                        )
                
                with col2:
                    # Similarity score gauge
                    fig = go.Figure(go.Indicator(
                        mode="gauge+number",
                        value=similarity_pct,
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
    
    @staticmethod
    def _display_similarity_scores(candidates: List) -> None:
        """Display similarity scores chart and table."""
        st.subheader("📊 Candidate Similarity Scores")
        
        # Create DataFrame for visualization
        df = pd.DataFrame([
            {
                "Candidate": candidate.name,
                "Similarity Score": candidate.similarity_score * 100 if candidate.similarity_score else 0,
                "Rank": i + 1
            }
            for i, candidate in enumerate(candidates)
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
        fig.update_layout(height=max(400, len(candidates) * 50))
        st.plotly_chart(fig, use_container_width=True)
        
        # Data table
        st.subheader("📋 Results Table")
        display_df = df[["Rank", "Candidate", "Similarity Score"]].copy()
        display_df["Similarity Score"] = display_df["Similarity Score"].round(1).astype(str) + "%"
        st.dataframe(display_df, use_container_width=True, hide_index=True)
    
    @staticmethod
    def _display_analysis(search_result: SearchResult) -> None:
        """Display analysis and insights."""
        st.subheader("📈 Analysis & Insights")
        
        ranked_candidates = search_result.ranked_candidates
        
        if not ranked_candidates:
            st.info("No candidates to analyze.")
            return
        
        # Metrics
        col1, col2, col3 = st.columns(3)
        
        with col1:
            avg_score = search_result.average_similarity
            st.metric("Average Similarity", f"{avg_score:.1%}")
        
        with col2:
            top_score = ranked_candidates[0].similarity_score if ranked_candidates else 0
            st.metric("Best Match", f"{top_score:.1%}")
        
        with col3:
            strong_matches = search_result.strong_matches_count
            st.metric("Strong Matches (≥70%)", strong_matches)
        
        # Score distribution
        if len(ranked_candidates) > 1:
            st.subheader("Score Distribution")
            scores = [c.similarity_score * 100 for c in ranked_candidates if c.similarity_score]
            
            if scores:
                fig = px.histogram(
                    x=scores,
                    nbins=min(10, len(scores)),
                    title="Distribution of Similarity Scores",
                    labels={"x": "Similarity Score (%)", "y": "Number of Candidates"}
                )
                st.plotly_chart(fig, use_container_width=True)
        
        # Service information
        st.subheader("Service Information")
        service_info = search_result.service_info
        st.info(f"**{service_info['method']}**: {service_info['description']}")
    
    @staticmethod
    def display_loading_message(message: str) -> None:
        """Display a loading message with spinner.
        
        Args:
            message: Loading message to display
        """
        with st.spinner(message):
            st.empty()  # Placeholder for the spinner
