"""
CSS styles for the Streamlit application.
"""


def get_custom_css() -> str:
    """Get custom CSS for the application.
    
    Returns:
        CSS string for styling the Streamlit app
    """
    return """
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
    
    /* Status indicators */
    .status-success {
        background: #F8FFF8; 
        border-left: 4px solid #27AE60; 
        border-radius: 0 8px 8px 0; 
        padding: 1rem 1.5rem; 
        margin: 1rem 0; 
        color: #1E8449;
    }
    
    .status-warning {
        background: #FFFAF0; 
        border-left: 4px solid #F39C12; 
        border-radius: 0 8px 8px 0; 
        padding: 1rem 1.5rem; 
        margin: 1rem 0; 
        color: #D68910;
    }
</style>
"""
