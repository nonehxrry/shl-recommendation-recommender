# -*- coding: utf-8 -*-
"""
SHL Assessment Recommendation Engine
Streamlit Web Application
"""

# Standard library imports
import os
import sys
import traceback
from datetime import datetime
from pathlib import Path
from typing import Optional, List, Dict
import requests  # NEW IMPORT

# Third-party imports
import pandas as pd
import streamlit as st
from streamlit.runtime.uploaded_file_manager import UploadedFile
from recommend import initialize_model

# Local imports
try:
    from .recommend import get_top_k
except ImportError as e:
    from recommend import get_top_k

# ------------------------------
# Constants & Configuration
# ------------------------------
DATA_DIR = "data"
DEFAULT_CATALOG_PATH = os.path.join(DATA_DIR, "catalog.csv")
MAX_FILE_SIZE_MB = 5
ALLOWED_FILE_TYPES = ["csv"]
DEFAULT_RECOMMENDATIONS = 5
MAX_RECOMMENDATIONS = 20
API_BASE_URL = "http://localhost:8000"  # NEW CONSTANT

# ------------------------------
# Initial Configuration (EXACTLY THE SAME)
# ------------------------------
st.set_page_config(
    page_title="SHL Assessment Recommendation Engine",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded",
    menu_items={
        'Get Help': 'https://shl.com/support',
        'Report a bug': "https://shl.com/bug-report",
        'About': "### SHL AI Recommendation Engine v2.1"
    }
)

# ------------------------------
# Custom Styling (EXACTLY THE SAME)
# ------------------------------
def apply_custom_styles():
    """Preserves your original styling with visible text"""
    st.markdown("""
    <style>
        :root {
            --primary-color: #1f4e79;
            --secondary-color: #4b86b4;
            --accent-color: #63ace5;
        }
        
        .reportview-container {
            background-color: #f8f9fa;
        }
        
        .sidebar .sidebar-content {
            background: linear-gradient(180deg, var(--primary-color), var(--secondary-color));
            color: white;
            padding: 2rem 1rem;
        }
        
        .title {
            font-size: 2.5rem;
            font-weight: 700;
            color: var(--primary-color);
            margin-bottom: 0.5rem;
        }
        
        .subtitle {
            font-size: 1.1rem;
            color: #555;
            margin-bottom: 2rem;
        }
        
        .recommendation-card {
            border-left: 4px solid var(--accent-color);
            padding: 1.5rem;
            margin-bottom: 1.5rem;
            background-color: inherit;
            border-radius: 0 8px 8px 0;
            box-shadow: 0 2px 8px rgba(0,0,0,0.1);
            color: #333 !important;
        }
        
        .stButton>button {
            background-color: var(--accent-color);
            color: white;
            font-weight: 600;
            padding: 0.75rem 1.5rem;
            border-radius: 8px;
            border: none;
            transition: all 0.3s ease;
        }
        
        .stButton>button:hover {
            background-color: var(--secondary-color);
            transform: translateY(-2px);
        }
        
        footer {
            position: fixed;
            bottom: 0;
            width: 100%;
            background-color: var(--primary-color);
            color: white;
            padding: 1rem;
            text-align: center;
        }
        
        body, .stMarkdown {
            color: #333;
        }

        /* NEW: Style for evaluation metrics */
        .metric-box {
            background-color: white;
            border-radius: 8px;
            padding: 1rem;
            margin: 0.5rem 0;
            box-shadow: 0 1px 4px rgba(0,0,0,0.1);
        }
    </style>
    """, unsafe_allow_html=True)

# ------------------------------
# Data Handling (EXACTLY THE SAME)
# ------------------------------
@st.cache_data(show_spinner="Loading catalog data...")
def load_catalog(file_path: str = DEFAULT_CATALOG_PATH) -> pd.DataFrame:
    """[Previous implementation remains identical]"""
    # [Keep all existing code exactly the same]

# ------------------------------
# File Upload Handling (EXACTLY THE SAME)
# ------------------------------
def handle_file_upload() -> Optional[str]:
    """[Previous implementation remains identical]"""
    # [Keep all existing code exactly the same]

# ------------------------------
# Recommendation Display (EXACTLY THE SAME)
# ------------------------------
def display_recommendations(recommendations: List[Dict], query: str):
    """[Previous implementation remains identical]"""
    # [Keep all existing code exactly the same]

# ------------------------------
# NEW: Evaluation Metrics Display
# ------------------------------
def display_evaluation_metrics(recommendations: List[Dict]):
    """NEW: Shows evaluation metrics in a collapsible section"""
    with st.expander("📊 Evaluation Metrics", expanded=False):
        st.markdown("""
        <div class="metric-box">
            <p><strong>Note:</strong> Configure your test cases in the code</p>
        </div>
        """, unsafe_allow_html=True)
        
        if st.button("Calculate Sample Metrics"):
            # Example test case - replace with your actual ground truth
            sample_ground_truth = ['Communication Skills']
            
            if recommendations:
                # Calculate simple metrics
                relevant_count = sum(1 for r in recommendations 
                                   if r['assessment_name'] in sample_ground_truth)
                
                st.markdown(f"""
                <div class="metric-box">
                    <p><strong>Relevant Assessments Found:</strong> {relevant_count}/{len(sample_ground_truth)}</p>
                    <p><strong>Average Score:</strong> {sum(r['score'] for r in recommendations)/len(recommendations):.2f}</p>
                </div>
                """, unsafe_allow_html=True)
            else:
                st.warning("No recommendations to evaluate")

# ------------------------------
# Main App Functionality (CAREFULLY MODIFIED)
# ------------------------------
def main():
    """Main application function with new features added without changing existing UI"""
    # Apply styles and setup (unchanged)
    apply_custom_styles()
    
    # Initialize model once at startup (unchanged)
    model = initialize_model()
    
    # Header Section (EXACTLY THE SAME)
    st.markdown('<div class="title">🔍 SHL Assessment Recommendation Engine</div>', 
                unsafe_allow_html=True)
    st.markdown("""
    <div class="subtitle">
        AI-powered tool to match job descriptions with the most relevant SHL assessments.
        <br>Enter a job description below to get started.
    </div>
    """, unsafe_allow_html=True)
    
    # Sidebar Configuration (EXACTLY THE SAME)
    st.sidebar.title("⚙️ Configuration")
    
    # NEW: API Health Check (added to existing sidebar)
    with st.sidebar.expander("🔌 API Status", expanded=False):
        if st.button("Check Health"):
            try:
                response = requests.get(f"{API_BASE_URL}/health", timeout=3)
                st.success(f"API Status: {response.json().get('status', 'unknown')}")
            except Exception as e:
                st.error(f"API Error: {str(e)}")
    
    # File upload handling (EXACTLY THE SAME)
    custom_catalog_path = handle_file_upload()
    catalog_path = custom_catalog_path if custom_catalog_path else DEFAULT_CATALOG_PATH
    df = load_catalog(catalog_path)
    
    # Query Input Section (EXACTLY THE SAME)
    st.header("📝 Enter Job Description")
    query = st.text_area(
        "Describe the role, skills, or candidate profile:",
        height=200,
        placeholder="Example: Looking for a sales manager with strong communication skills...",
        help="Be as specific as possible for better recommendations"
    )
    
    # Recommendation Settings (MODIFIED TO ADD NEW CONTROLS)
    with st.expander("⚙️ Advanced Settings"):
        col1, col2 = st.columns(2)
        
        # Left column (existing controls remain first)
        with col1:
            # Existing slider stays exactly the same
            num_recommendations = st.slider(
                "Number of recommendations",
                1, MAX_RECOMMENDATIONS, DEFAULT_RECOMMENDATIONS,
                help="Adjust how many results to display"
            )
            
            # NEW: Duration filter added below existing control
            max_duration = st.slider(
                "Max duration (minutes)",
                0, 240, 120,
                help="Filter by assessment duration"
            )
        
        # Right column (existing checkbox remains first)
        with col2:
            # Existing checkbox stays exactly the same
            show_technical = st.checkbox(
                "Show technical details",
                False,
                help="Display embedding and scoring details"
            )
            
            # NEW: Confidence threshold added below existing control
            min_score = st.slider(
                "Minimum confidence score",
                0.0, 1.0, 0.3, 0.05,
                help="Filter by minimum match quality"
            )
    
    # Generate Recommendations (MODIFIED TO USE NEW PARAMETERS)
    if st.button("🚀 Generate Recommendations", type="primary"):
        if not query.strip():
            st.warning("Please enter a job description to continue")
        else:
            try:
                with st.spinner("🔍 Analyzing job description and finding best matches..."):
                    recommendations = get_top_k(
                        query=query,
                        df=df,
                        k=num_recommendations,
                        model=model,
                        max_duration=max_duration,
                        min_score=min_score
                    )
                
                # Existing display function called exactly as before
                display_recommendations(recommendations, query)
                
                # NEW: Evaluation metrics section added AFTER existing results
                display_evaluation_metrics(recommendations)
                    
            except Exception as e:
                # Enhanced error handling
                error_msg = str(e)
                if "duration" in error_msg.lower():
                    st.warning("⚠️ " + error_msg)
                else:
                    st.error(f"""
                    ❌ An error occurred:
                    {error_msg}
                    """)
                
                if show_technical:
                    with st.expander("Technical Details"):
                        st.code(traceback.format_exc())
    
    # Footer (EXACTLY THE SAME)
    st.markdown("---")
    st.markdown("""
    <footer>
        © 2025 SHL AI Intern Project | Version 2.1 | Built by Harjit Singh Bhadauriya
    </footer>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    os.makedirs(DATA_DIR, exist_ok=True)
    main()
