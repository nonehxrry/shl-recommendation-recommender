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
from typing import List, Dict, Optional

# Third-party imports
import pandas as pd
import streamlit as st
from streamlit.runtime.uploaded_file_manager import UploadedFile

# Local imports
try:
    from recommend import get_top_k
except ImportError as e:
    st.error("""
    ❌ Critical Error: Failed to import recommendation module.
    Please ensure all dependencies are installed correctly.
    """)
    st.stop()

# ------------------------------
# Constants & Configuration
# ------------------------------
DEFAULT_CATALOG_PATH = "data/catalogue.csv"
MAX_FILE_SIZE_MB = 5
ALLOWED_FILE_TYPES = ["csv"]
DEFAULT_RECOMMENDATIONS = 5
MAX_RECOMMENDATIONS = 20

# ------------------------------
# Page Configuration
# ------------------------------
def configure_page():
    """Set up Streamlit page configuration."""
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
# Custom Styling
# ------------------------------
def apply_custom_styles():
    """Inject custom CSS styles."""
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
        
        .sidebar .sidebar-content .stMarkdown h1 {
            color: white;
        }
        
        .main .block-container {
            padding: 3rem 1rem 10rem;
            max-width: 1200px;
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
            background-color: white;
            border-radius: 0 8px 8px 0;
            box-shadow: 0 2px 8px rgba(0,0,0,0.1);
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
    </style>
    """, unsafe_allow_html=True)

# ------------------------------
# Data Handling
# ------------------------------
@st.cache_data(show_spinner="Loading catalog data...")
def load_catalog(file_path: str = DEFAULT_CATALOG_PATH) -> pd.DataFrame:
    """
    Load and validate the product catalog.
    
    Args:
        file_path: Path to the catalog CSV file
        
    Returns:
        pandas.DataFrame: Loaded catalog data
        
    Raises:
        ValueError: If required columns are missing
    """
    try:
        df = pd.read_csv(file_path)
        
        # Validate required columns
        required_cols = [
            'assessment_name', 
            'description', 
            'skills_measured', 
            'job_roles', 
            'duration_minutes', 
            'assessment_id'
        ]
        
        missing_cols = [col for col in required_cols if col not in df.columns]
        if missing_cols:
            raise ValueError(f"Missing required columns: {', '.join(missing_cols)}")
            
        # Clean data
        df.fillna('', inplace=True)
        
        # Create searchable text blob
        df['text_blob'] = (
            df['assessment_name'] + ". " + 
            df['description'] + ". " + 
            df['skills_measured'] + ". " + 
            df['job_roles']
        )
        
        return df
        
    except Exception as e:
        st.error(f"❌ Failed to load catalog: {str(e)}")
        st.stop()

# ------------------------------
# File Upload Handling
# ------------------------------
def handle_file_upload() -> Optional[str]:
    """
    Handle catalog file upload from user.
    
    Returns:
        str or None: Path to temporary file if uploaded, None otherwise
    """
    st.sidebar.markdown("## 📁 Upload Custom Catalog")
    uploaded_file = st.sidebar.file_uploader(
        "Upload CSV (max 5MB)",
        type=ALLOWED_FILE_TYPES,
        accept_multiple_files=False,
        help="Upload a custom SHL catalog CSV file"
    )
    
    if uploaded_file is not None:
        if uploaded_file.size > MAX_FILE_SIZE_MB * 1024 * 1024:
            st.sidebar.error(f"File too large. Max size: {MAX_FILE_SIZE_MB}MB")
            return None
            
        # Save to temp file
        temp_path = os.path.join("data", "temp_catalog.csv")
        with open(temp_path, "wb") as f:
            f.write(uploaded_file.getbuffer())
            
        st.sidebar.success("File uploaded successfully!")
        return temp_path
        
    return None

# ------------------------------
# Recommendation Display
# ------------------------------
def display_recommendations(recommendations: List[Dict], query: str):
    """
    Display recommendations in an attractive format.
    
    Args:
        recommendations: List of recommendation dictionaries
        query: Original user query
    """
    st.success(f"✅ Found {len(recommendations)} recommendations for: '{query}'")
    st.markdown("---")
    
    for i, rec in enumerate(recommendations, 1):
        with st.container():
            st.markdown(f"""
            <div class="recommendation-card">
                <h3>#{i}: {rec['assessment_name']}</h3>
                <p><strong>🔍 Match Score:</strong> {rec.get('score', 0):.2f}/1.00</p>
                <p><strong>📝 Description:</strong> {rec['description']}</p>
                <p><strong>⏱ Duration:</strong> {rec['duration']} minutes</p>
                <p><strong>🔗</strong> <a href="{rec['url']}" target="_blank">View Assessment Details</a></p>
            </div>
            """, unsafe_allow_html=True)
            
            # Add expandable technical details
            with st.expander("ℹ️ Technical Details"):
                st.json({
                    "assessment_id": rec.get('assessment_id', ''),
                    "embedding_model": "all-MiniLM-L6-v2",
                    "similarity_metric": "cosine_similarity"
                })

# ------------------------------
# Main App Functionality
# ------------------------------
def main():
    """Main application function."""
    configure_page()
    apply_custom_styles()
    
    # Header Section
    st.markdown('<div class="title">🔍 SHL Assessment Recommendation Engine</div>', 
                unsafe_allow_html=True)
    st.markdown("""
    <div class="subtitle">
        AI-powered tool to match job descriptions with the most relevant SHL assessments.
        <br>Enter a job description below to get started.
    </div>
    """, unsafe_allow_html=True)
    
    # Sidebar Configuration
    st.sidebar.title("⚙️ Configuration")
    
    # File upload handling
    custom_catalog_path = handle_file_upload()
    catalog_path = custom_catalog_path if custom_catalog_path else DEFAULT_CATALOG_PATH
    
    # Load data
    try:
        df = load_catalog(catalog_path)
    except Exception as e:
        st.error(f"Data loading error: {str(e)}")
        st.stop()
    
    # Query Input Section
    st.header("📝 Enter Job Description")
    query = st.text_area(
        "Describe the role, skills, or candidate profile:",
        height=200,
        placeholder="Example: Looking for a sales manager with strong communication skills, analytical ability, and team leadership experience...",
        help="Be as specific as possible for better recommendations"
    )
    
    # Recommendation Settings
    with st.expander("⚙️ Advanced Settings"):
        col1, col2 = st.columns(2)
        with col1:
            num_recommendations = st.slider(
                "Number of recommendations",
                1,
                MAX_RECOMMENDATIONS,
                DEFAULT_RECOMMENDATIONS,
                help="Adjust how many results to display"
            )
        with col2:
            show_technical = st.checkbox(
                "Show technical details",
                False,
                help="Display embedding and scoring details"
            )
    
    # Generate Recommendations
    if st.button("🚀 Generate Recommendations", type="primary"):
        if not query.strip():
            st.warning("Please enter a job description to continue")
        else:
            try:
                with st.spinner("🔍 Analyzing job description and finding best matches..."):
                    recommendations = get_top_k(query, k=num_recommendations)
                
                if not recommendations:
                    st.warning("No matching assessments found. Try broadening your search.")
                else:
                    display_recommendations(recommendations, query)
                    
            except Exception as e:
                st.error("""
                ❌ An error occurred during processing.
                Please try again or contact support if the problem persists.
                """)
                if show_technical:
                    with st.expander("Technical Details"):
                        st.code(traceback.format_exc())
    
    # Footer
    st.markdown("---")
    st.markdown("""
    <footer>
        © 2025 SHL AI Intern Project | Version 2.1 | Built by Harjit Singh Bhadauriya
    </footer>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()
