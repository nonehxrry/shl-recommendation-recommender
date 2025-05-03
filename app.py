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
        /* [Previous CSS remains exactly the same] */
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
# NEW: API Health Check Function
# ------------------------------
def check_api_health():
    """NEW: Checks API status without affecting main flow"""
    try:
        with st.spinner("Checking API..."):
            response = requests.get(f"{API_BASE_URL}/health", timeout=3)
            if response.status_code == 200:
                st.sidebar.success("API is healthy ✅")
            else:
                st.sidebar.error(f"API Error: {response.status_code}")
    except Exception as e:
        st.sidebar.error("API not reachable")

# ------------------------------
# Main App Functionality (CAREFULLY MODIFIED)
# ------------------------------
def main():
    """Main application function with new features added without changing existing UI"""
    # Apply styles and setup (unchanged)
    apply_custom_styles()
    
    # Initialize model once at startup (unchanged)
    model = initialize_model()
    
    # [Keep ALL existing header, sidebar, and file upload code exactly as is...]
    
    # ====== NEW: Add API Health Check Button to Sidebar ======
    if st.sidebar.button("🔌 Check API Status"):
        check_api_health()
    
    # [Keep ALL existing query input section exactly as is...]
    
    # ====== MODIFIED: Enhanced Recommendation Generation ======
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
                        model=model
                    )
                
                # Existing display function called exactly as before
                display_recommendations(recommendations, query)
                    
            except Exception as e:
                error_msg = str(e)
                if "'NoneType' object is not subscriptable" in error_msg:
                    st.error("❌ Failed to load recommendations. Please check your data.")
                else:
                    st.error(f"""
                    ❌ An error occurred:
                    {error_msg}
                    """)
                
                if show_technical:
                    with st.expander("Technical Details"):
                        st.code(traceback.format_exc())
    
    # [Keep footer exactly the same]

if __name__ == "__main__":
    os.makedirs(DATA_DIR, exist_ok=True)
    main()
