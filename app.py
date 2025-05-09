





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
from recommend import initialize_model, get_top_k
import requests  # For fetching from URLs
from bs4 import BeautifulSoup  # For parsing HTML (optional, for better URL handling)
import spacy  # For keyword/skill extraction (if implemented)

# ------------------------------
# Constants & Configuration
# ------------------------------
DATA_DIR = "data"
DEFAULT_CATALOG_PATH = os.path.join(DATA_DIR, "catalog.csv")  # Changed from catalogue.csv to catalog.csv
MAX_FILE_SIZE_MB = 5
ALLOWED_FILE_TYPES = ["csv"]
DEFAULT_RECOMMENDATIONS = 5
MAX_RECOMMENDATIONS = 10  # Updated to match the PDF requirement
API_ENDPOINT = "/recommendations"  # Define API endpoint

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
    """Preserves your original styling with visible text"""
    st.markdown("""
    <style>
        :root {
            --primary-color: #1f4e79;
            --secondary-color: #4b86b4;
            --accent-color: #63ace5;
        }

        /* MAIN CONTAINER (unchanged) */
        .reportview-container {
            background-color: #f8f9fa;
        }

        /* SIDEBAR (unchanged) */
        .sidebar .sidebar-content {
            background: linear-gradient(180deg, var(--primary-color), var(--secondary-color));
            color: white;
            padding: 2rem 1rem;
        }

        /* HEADERS (unchanged) */
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

        /* RECOMMENDATION CARDS (text visibility fix only) */
        .recommendation-card {
            border-left: 4px solid var(--accent-color);
            padding: 1.5rem;
            margin-bottom: 1.5rem;
            background-color: inherit;  /* Changed from white to inherit */
            border-radius: 0 8px 8px 0;
            box-shadow: 0 2px 8px rgba(0,0,0,0.1);
            color: #333 !important;  /* Ensures dark text */
        }

        /* BUTTONS (unchanged) */
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

        /* FOOTER (unchanged) */
        footer {
            position: fixed;
            bottom: 0;
            width: 100%;
            background-color: var(--primary-color);
            color: white;
            padding: 1rem;
            text-align: center;
        }

        /* ADDED: Ensure all text is visible */
        body, .stMarkdown {
            color: #333;
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

            # Create sample data with Remote Testing and Adaptive/IRT support
            sample_data = {
                'assessment_name': ['Communication Skills', 'Leadership Potential', 'Numerical Reasoning', 'OPQ32r', 'Motivation Questionnaire'],
                'description': ['Measures verbal and written communication', 'Evaluates leadership potential', 'Assesses ability to understand numerical data', 'Occupational Personality Questionnaire', 'Evaluates workplace motivation'],
                'skills_measured': ['Verbal communication, Writing', 'Decision making, Team management', 'Data interpretation, Calculation', 'Personality traits', 'Motivators, Values'],
                'job_roles': ['All roles', 'Managerial roles', 'Roles requiring data analysis', 'All roles', 'All roles'],
                'duration_minutes': [30, 45, 60, 60, 20],
                'assessment_id': [101, 102, 103, 104, 105],
                'remote_testing': ['Yes', 'Yes', 'Yes', 'Yes', 'Yes'],
                'adaptive_irt': ['No', 'Yes', 'No', 'No', 'No'],
                'test_type': ['Skills Test', 'Personality Assessment', 'Aptitude Test', 'Personality Assessment', 'Motivation Assessment'],
                'url': ['https://www.shl.com/communication-skills', 'https://www.shl.com/leadership-assessment', 'https://www.shl.com/numerical-reasoning', 'https://www.shl.com/solutions/products/product-catalog/view/occupational-personality-questionnaire-opq32r/', 'https://www.shl.com/motivation-questionnaire/']  # Example URLs
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

        # Validate required columns (including new ones from PDF)
        required_cols = [
            'assessment_name', 'description', 'skills_measured',
            'job_roles', 'duration_minutes', 'assessment_id',
            'remote_testing', 'adaptive_irt', 'test_type', 'url'
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
        st.error(f"Please ensure you have a valid 'catalog.csv' file in the 'data' directory with the required columns.")
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
            help="Upload a custom SHL catalog CSV file with columns: assessment_name, description, skills_measured, job_roles, duration_minutes, assessment_id, remote_testing, adaptive_irt, test_type, url"
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
# Text Extraction from URL (Enhanced)
# ------------------------------
def extract_text_from_url(url: str) -> str:
    """
    Extracts relevant text from a given URL.
    This is a simplified version and might need adjustments for different website structures.
    """
    try:
        response = requests.get(url, timeout=10)  # Add timeout
        response.raise_for_status()  # Raise HTTPError for bad responses (4xx or 5xx)

        soup = BeautifulSoup(response.content, 'html.parser')
        # This is a VERY basic selector; you'll likely need to inspect the target website
        # to find the correct tags/classes that contain the job description.
        job_description_elements = soup.find_all(['p', 'li', 'div'], class_=['job-description', 'description', 'job-detail'])
        text_parts = [element.get_text(strip=True) for element in job_description_elements if element.get_text(strip=True)]
        extracted_text = " ".join(text_parts)

        if not extracted_text:
            # If basic extraction fails, try a more general approach
            extracted_text = soup.get_text(separator=" ", strip=True)
            if len(extracted_text) > 500:  # Limit length for very long pages
                extracted_text = extracted_text[:500] + "..."  # Truncate

        return extracted_text

    except requests.exceptions.RequestException as e:
        st.error(f"Error fetching URL: {e}")
        return ""
    except Exception as e:
        st.error(f"Error processing URL content: {e}")
        return ""

# ------------------------------
# Keyword/Skill Extraction (Basic - Requires spaCy setup)
# ------------------------------
def extract_keywords(text: str) -> List[str]:
    """
    Extracts keywords and skills from the given text (job description).
    Requires spaCy to be installed and a suitable model downloaded (e.g., 'en_core_web_sm').
    This is a basic implementation and can be significantly improved.
    """
    try:
        nlp = spacy.load("en_core_web_sm")  # Load the English model
        doc = nlp(text)
        keywords = [token.text for token in doc if token.pos_ in ['NOUN', 'ADJ'] or token.ent_type_ == 'SKILL']
        return keywords
    except Exception as e:
        st.warning(f"Keyword extraction failed: {e}.  Please ensure spaCy and a model are installed.")
        return []

# ------------------------------
# Recommendation Display
# ------------------------------
def display_recommendations(recommendations: List[Dict], query: str, show_keywords: bool = False):
    """
    Display recommendations in an attractive format, including details from the PDF.
    Optionally displays extracted keywords.
    """
    if not recommendations:
        st.warning("No recommendations found. Try broadening your search criteria.")
        return

    st.success(f"✅ Found {len(recommendations)} relevant assessment solutions for: '{query}'")
    st.markdown("---")

    for i, rec in enumerate(recommendations, 1):
        with st.container():
            st.markdown(f"""
            <div class="recommendation-card">
                <h3>#{i}: <a href="{rec.get('url', '#')}" target="_blank">{rec['assessment_name']}</a></h3>
                <p><strong>🔍 Match Score:</strong> {rec.get('score', 0):.2f}/1.00</p>
                <p><strong>📝 Description:</strong> {rec['description']}</p>
                <p><strong>⏱ Duration:</strong> {rec['duration_minutes']} minutes</p>
                <p><strong>🧪 Test Type:</strong> {rec['test_type']}</p>
                <p><strong>Remote Testing Support:</strong> {rec['remote_testing']}</p>
                <p><strong>Adaptive/IRT Support:</strong> {rec['adaptive_irt']}</p>
            </div>
            """, unsafe_allow_html=True)

            # Add expandable technical details
            with st.expander("ℹ️ Technical Details"):
                st.json({
                    "assessment_id": rec.get('assessment_id', ''),
                    "embedding_model": "all-MiniLM-L6-v2",
                    "similarity_metric": "cosine_similarity"
                })
    if show_keywords:
        extracted_keywords = extract_keywords(query)
        if extracted_keywords:
            st.markdown("---")
            st.subheader("🔑 Extracted Keywords")
            st.write(f"The following keywords were extracted from your input: {', '.join(extracted_keywords)}")
        else:
            st.warning("Could not extract keywords from the input.")

# ------------------------------
# API Endpoint (Conceptual for Demo)
# ------------------------------
def api_recommendations(query: str, df: pd.DataFrame, model, k: int = MAX_RECOMMENDATIONS) -> List[Dict]:
    """
    Conceptual function for the API endpoint to return recommendations in JSON format.
    In a real deployment, this would be part of a separate API framework (e.g., FastAPI).
    """
    if not query.strip():
        return []
    try:
        recommendations = get_top_k(
            query=query,
            df=df,
            k=k,
            model=model
        )
        # Add the required attributes for the API response
        api_response = []
        for rec in recommendations:
            api_response.append({
                "assessment_name": rec.get('assessment_name'),
                "url": rec.get('url', '#'),
                "remote_testing": rec.get('remote_testing'),
                "adaptive_irt": rec.get('adaptive_irt'),
                "duration": f"{rec.get('duration_minutes')} minutes",
                "test_type": rec.get('test_type'),
                "match_score": f"{rec.get('score', 0):.2f}"
            })
        return api_response
    except Exception as e:
        print(f"Error generating API recommendations: {e}")
        return []

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
    st.markdown('<div class="title">🔍 SHL Assessment Recommendation Engine</div>', unsafe_allow_html=True)
    st.markdown("""
    <div class="subtitle">
        AI-powered tool to match job descriptions with the most relevant SHL assessment solutions.
        <br>Enter a natural language query, job description text, or URL below to get started.
    </div>
    """, unsafe_allow_html=True)

    # Sidebar Configuration
    st.sidebar.title("⚙️ Configuration")

    # File upload handling - returns None if no file uploaded
    custom_catalog_path = handle_file_upload()

    # Load data - uses uploaded file if available, otherwise default
    catalog_path = custom_catalog_path if custom_catalog_path else DEFAULT_CATALOG_PATH
    df = load_catalog(catalog_path)

    # Input Method Selection
    input_method = st.radio(
        "Choose Input Method:",
        ["Natural Language Query / Job Description Text", "Job Description URL", "Paste Job Description Text"],
        index=0,
        help="Choose how you want to provide the job description or query."
    )

    query = ""
    if input_method == "Natural Language Query / Job Description Text":
        # Query Input Section
        st.header("📝 Enter Job Description or Query")
        query = st.text_area(
            "Describe the role, skills, or candidate profile:",
            height=200,
            placeholder="Example: Looking for a sales manager with strong communication skills, analytical ability, and team leadership experience...",
            help="Be as specific as possible for better recommendations"
        )
    elif input_method == "Job Description URL":
        st.header("🔗 Enter Job Description URL")
        job_url = st.text_input("Paste the URL of the job description:", help="Enter a valid URL")
        if job_url:
            with st.spinner("Fetching job description from URL..."):
                query = extract_text_from_url(job_url)
            if not query:
                query = job_url # Fallback to URL if extraction fails
                st.info("Failed to extract job description. Using URL as query.")
    elif input_method == "Paste Job Description Text":
        st.header("📄 Paste Job Description Text")
        pasted_text = st.text_area("Paste the job description here:", height=300, placeholder="Paste job description...", help="Paste the job description text directly.")
        query = pasted_text


    # Recommendation Settings
    with st.expander("⚙️ Advanced Settings"):
        col1, col2, col3 = st.columns(3)
        with col1:
            num_recommendations = st.slider(
                "Number of recommendations",
                1,
                MAX_RECOMMENDATIONS,
                DEFAULT_RECOMMENDATIONS,
                help=f"Adjust how many results to display (max {MAX_RECOMMENDATIONS})"
            )
        with col2:
            show_technical = st.checkbox(
                "Show technical details",
                False,
                help="Display embedding and scoring details"
            )
        with col3:
            show_keywords = st.checkbox(
                "Show extracted keywords",
                False,
                help="Display keywords extracted from the input"
            )

    # Generate Recommendations
    if st.button("🚀 Generate Recommendations", type="primary"):
        if not query.strip():
            st.warning("Please enter a job description or query to continue")
        else:
            try:
                with st.spinner("🔍 Analyzing input and finding best matches..."):
                    # Pass the pre-loaded model to get_top_k
                    recommendations = get_top_k(
                        query=query,
                        df=df,
                        k=num_recommendations,
                        model=model
                    )

                    display_recommendations(recommendations, query, show_keywords)

            except Exception as e:
                st.error(f"""
                ❌ An error occurred during processing: {str(e)}
                Please try again or contact support if the problem persists.
                """)
                if show_technical:
                    with st.expander("Technical Details"):
                        st.code(traceback.format_exc())

    # API Endpoint Information (for the submission)
    st.sidebar.markdown("---")
    st.sidebar.markdown(f"## 🌐 API Endpoint (Conceptual)")
    st.sidebar.markdown(f"The API endpoint for querying this recommendation engine would be:")
    # Replace 'your_app_url' with the actual URL of your deployed Streamlit app
    st.sidebar.code(f"POST https://shl-recommendation-recommender-nonehxrry.streamlit.app/recommendations", language="bash")
    st.sidebar.markdown("It would accept a JSON payload with a 'query' field and return a JSON response containing a list of assessment recommendations in the following format:")
    st.sidebar.code(
        """
        [
            {
                "assessment_name": "Assessment Name 1",
                "url": "https://...",
                "remote_testing": "Yes/No",
                "adaptive_irt": "Yes/No",
                "duration": "XX minutes",
                "test_type": "Type",
                "match_score": "0.XX"
            },
            {
                "assessment_name": "Assessment Name 2",
                "url": "https://...",
                "remote_testing": "Yes/No",
                "adaptive_irt": "Yes/No",
                "duration": "YY minutes",
                "test_type": "Type",
                "match_score": "0.YY"
            },
            ...
        ]
        """,
        language="json"
    )
    st.sidebar.markdown("---")
    st.sidebar.markdown("## 💻 GitHub Code")
    st.sidebar.markdown(f"[https://github.com/nonehxrry](https://github.com/nonehxrry)") # Updated with your GitHub link

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

