### ui.py (Streamlit UI)

import streamlit as st
import requests
import pandas as pd

# Set page configuration
st.set_page_config(
    page_title="SHL Assessment Recommendation Engine",
    layout="centered",
    initial_sidebar_state="collapsed"
)

# Title and instructions
st.title("🔍 SHL Assessment Recommendation Engine")
st.markdown("""
Welcome to SHL's smart recommendation system! Just enter a job role or required skillset below, and our engine will suggest the most suitable SHL assessments for your needs.

💡 *Examples:*
- "Looking for someone with great analytical and reasoning skills"
- "Need a test to evaluate Java programming and debugging capabilities"
- "Managerial leadership and strategic planning skills"
""")

# Text input from user
user_input = st.text_area("📝 Describe the skillset or job role:", height=150)

# Submit button
if st.button("🎯 Recommend Assessments"):
    if not user_input.strip():
        st.warning("Please enter a description or skillset.")
    else:
        with st.spinner("Fetching personalized recommendations..."):
            try:
                response = requests.post("http://localhost:5000/recommend", json={"text": user_input})
                if response.status_code == 200:
                    data = response.json()
                    recommendations = data.get("recommendations", [])
                    st.success("Top Recommended Assessments:")
                    for i, rec in enumerate(recommendations, start=1):
                        st.markdown(f"### {i}. {rec['Assessment Name']}")
                        st.write(f"**Skills Covered:** {rec['Skills Covered']}")
                        st.write(f"**Suitable For:** {rec['Job Roles']}")
                        st.write(f"**Description:** {rec['Description']}")
                        st.markdown("---")
                else:
                    st.error("Failed to fetch recommendations from the server.")
            except Exception as e:
                st.error(f"An error occurred: {e}")
