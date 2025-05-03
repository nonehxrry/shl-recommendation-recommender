# app.py

import streamlit as st
import pandas as pd
import os
import traceback
from recommend import get_top_k  # Recommender logic using Sentence Transformers
from datetime import datetime

# ------------------------------
# Page Setup
# ------------------------------
def load_data():
    # Loading SHL Product Catalogue CSV
    return pd.read_csv("data/catalogue.csv")

shl_catalogue = load_data()

st.set_page_config(
    page_title="SHL Assessment Recommendation Engine",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ------------------------------
# Custom Style (Optional but Elegant)
# ------------------------------
st.markdown("""
    <style>
        .reportview-container {
            background-color: #f9f9f9;
        }
        .sidebar .sidebar-content {
            background-color: #e6f2ff;
        }
        .main .block-container {
            padding: 2rem 1rem;
        }
        .title {
            font-size: 36px;
            font-weight: bold;
            color: #1f4e79;
        }
        .subtitle {
            font-size: 20px;
            color: #555;
        }
    </style>
""", unsafe_allow_html=True)

# ------------------------------
# Title and Introduction
# ------------------------------
st.markdown('<div class="title">🔍 SHL Assessment Recommendation Engine</div>', unsafe_allow_html=True)
st.markdown('<div class="subtitle">Input a job description or candidate role and receive relevant SHL assessments.</div><br>', unsafe_allow_html=True)

# ------------------------------
# Sidebar with Metadata
# ------------------------------
st.sidebar.title("ℹ️ Project Info")
st.sidebar.markdown("""
This tool uses:
- SHL Product Catalogue (.csv)
- Sentence-Transformer Embeddings
- Streamlit Web App Framework

Developed for SHL AI Internship Assignment.
""")
st.sidebar.markdown("---")
st.sidebar.markdown("📅 **Today's Date:** " + datetime.now().strftime("%B %d, %Y"))

# ------------------------------
# Query Input Section
# ------------------------------
st.header("📝 Step 1: Enter Role or Job Description")
st.markdown("Type in the job description or desired skillset. The engine will match the most relevant SHL assessments.")

user_input = st.text_area("Enter job role, description, or required skills below:", height=200, placeholder="e.g., Looking for a sales professional with excellent communication and analytical skills.")

top_k = st.slider("🔢 Number of Recommendations", 1, 20, 10, help="Set how many assessment suggestions you want.")

# ------------------------------
# Trigger Recommendation
# ------------------------------
if st.button("🚀 Generate Recommendations"):
    if user_input.strip() == "":
        st.warning("Please provide a job description or query to proceed.")
    else:
        try:
            with st.spinner("Generating top matches..."):
                results = get_top_k(user_input, k=top_k)

            st.success(f"✅ Top {len(results)} Recommended SHL Assessments")
            st.markdown("---")

            # ------------------------------
            # Display Recommendations
            # ------------------------------
            for i, res in enumerate(results, 1):
                with st.container():
                    st.markdown(f"""
                    ### {i}. {res['assessment_name']}
                    - 🧾 **Description:** {res['description']}
                    - ⏱️ **Duration:** {res['duration']} minutes
                    - 🔗 [**Assessment Link**]({res['url']})
                    """)
                    st.markdown("---")

        except Exception as e:
            st.error("❌ An error occurred during processing. Please check your input or contact support.")
            st.code(traceback.format_exc())

# ------------------------------
# Footer
# ------------------------------
st.markdown("<br><hr>", unsafe_allow_html=True)
st.markdown("© 2025 SHL Intern Project | Built by Harjit Singh Bhadauriya")


