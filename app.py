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
DEFAULT_CATALOG_PATH = os.path.join(DATA_DIR, "catalog.csv")  # Changed from catalogue.csv to catalog.csv
MAX_FILE_SIZE_MB = 5
ALLOWED_FILE_TYPES = ["csv"]
DEFAULT_RECOMMENDATIONS = 5
MAX_RECOMMENDATIONS = 20

# ------------------------------
# Initial Configuration (MUST BE FIRST STREAMLIT COMMAND)
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
# Custom Styling
# ------------------------------
def apply_custom_styles():
    """Only the essential fixes for text visibility"""
    st.markdown("""
    <style>
        /* ONLY THE ABSOLUTELY NECESSARY FIXES */
        /* Make sure text is visible in recommendation cards */
        .recommendation-card {
            background-color: white;
            color: black;
        }
        
        /* Ensure text areas have visible text */
        .stTextArea textarea {
            color: black;
        }
        
        /* Make sure form labels are visible */
        .stTextArea label, .stSlider label, .stCheckbox label {
            color: black;
        }
    </style>
    """, unsafe_allow_html=True)
# ------------------------------
# Data Handling
# ------------------------------
@st.cache_data(show_spinner="Loading catalog data...")
def load_catalog(file_path: str = DEFAULT_CATALOG_PATH) -> pd.DataFrame:
    """
    Load and validate the product catalog with fallback to sample data.
    """
    try:
        # Ensure data directory exists
        os.makedirs(DATA_DIR, exist_ok=True)
        
        # Check if file exists
        if not os.path.exists(file_path):
            st.warning(f"Catalog not found at {file_path} - using sample data. Please upload your catalog.")
            
            # Create sample data
            sample_data = {
                'assessment_name': ['Communication Skills', 'Leadership Assessment'],
                'description': ['Measures verbal and written communication', 'Evaluates leadership potential'],
                'skills_measured': ['Verbal communication, Writing', 'Decision making, Team management'],
                'job_roles': ['All roles', 'Managerial roles'],
                'duration_minutes': [30, 45],
                'assessment_id': [101, 102]
            }
            df = pd.DataFrame(sample_data)
            
            # Try to save sample data for future use
            try:
                df.to_csv(file_path, index=False)
                st.info(f"Created sample catalog at {file_path}")
            except Exception as save_error:
                st.warning(f"Couldn't save sample catalog: {save_error}")
            
            return df
            
        # Load existing catalog
        df = pd.read_csv(file_path)
        
        # Validate required columns
        required_cols = [
            'assessment_name', 'description', 'skills_measured',
            'job_roles', 'duration_minutes', 'assessment_id'
        ]
        
        missing_cols = [col for col in required_cols if col not in df.columns]
        if missing_cols:
            raise ValueError(f"Missing required columns: {', '.join(missing_cols)}")
            
        # Clean data
        df.fillna('', inplace=True)
        df['text_blob'] = (
            df['assessment_name'] + ". " + 
            df['description'] + ". " + 
            df['skills_measured'] + ". " + 
            df['job_roles']
        )
        
        return df
        
    except Exception as e:
        st.error(f"❌ Failed to load catalog: {str(e)}")
        st.error(f"Please ensure you have a valid 'catalog.csv' file in the 'data' directory.")
        st.error(f"Current working directory: {os.getcwd()}")
        st.error(f"Looking for catalog at: {os.path.abspath(file_path)}")
        st.stop()

# ------------------------------
# File Upload Handling
# ------------------------------
def handle_file_upload() -> Optional[str]:
    """
    Handle catalog file upload from user.
    Returns path to uploaded file if successful, None otherwise.
    """
    st.sidebar.markdown("## 📁 Upload Custom Catalog")
    
    with st.sidebar:
        uploaded_file = st.file_uploader(
            "Upload CSV (max 5MB)",
            type=ALLOWED_FILE_TYPES,
            accept_multiple_files=False,
            help="Upload a custom SHL catalog CSV file"
        )
        
        if uploaded_file is not None:
            # Validate file size
            if uploaded_file.size > MAX_FILE_SIZE_MB * 1024 * 1024:
                st.error(f"File too large. Max size: {MAX_FILE_SIZE_MB}MB")
                return None
                
            # Ensure data directory exists
            os.makedirs(DATA_DIR, exist_ok=True)
                
            # Save to data directory
            upload_path = os.path.join(DATA_DIR, "uploaded_catalog.csv")
            try:
                with open(upload_path, "wb") as f:
                    f.write(uploaded_file.getbuffer())
                st.success("File uploaded successfully!")
                return upload_path
            except Exception as e:
                st.error(f"Failed to save uploaded file: {e}")
                return None
                
    return None

# ------------------------------
# Recommendation Display
# ------------------------------
def display_recommendations(recommendations: List[Dict], query: str):
    """
    Display recommendations in an attractive format.
    """
    if not recommendations:
        st.warning("No recommendations found. Try broadening your search criteria.")
        return
        
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
                <p><strong>🔗</strong> <a href="{rec.get('url', '#')}" target="_blank">View Assessment Details</a></p>
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
    # Apply styles and setup
    apply_custom_styles()
    
    # Initialize model once at startup (using cache_resource)
    model = initialize_model()
    
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
    
    # File upload handling - returns None if no file uploaded
    custom_catalog_path = handle_file_upload()
    
    # Load data - uses uploaded file if available, otherwise default
    catalog_path = custom_catalog_path if custom_catalog_path else DEFAULT_CATALOG_PATH
    df = load_catalog(catalog_path)
    
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
                    # Pass the pre-loaded model to get_top_k
                    recommendations = get_top_k(
                        query=query,
                        df=df,
                        k=num_recommendations,
                        model=model  # Pass the initialized model
                    )
                
                display_recommendations(recommendations, query)
                    
            except Exception as e:
                st.error(f"""
                ❌ An error occurred during processing: {str(e)}
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
    # Ensure data directory exists before starting
    os.makedirs(DATA_DIR, exist_ok=True)
    main()
