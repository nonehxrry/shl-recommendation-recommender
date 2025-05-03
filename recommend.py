import pandas as pd
from sentence_transformers import SentenceTransformer, util
import torch
import streamlit as st
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def load_data():
    """Load and validate the product catalog."""
    try:
        df = pd.read_csv('data/catalogue.csv')
        
        # Ensure required columns are present
        required_cols = ['assessment_name', 'description', 'skills_measured', 
                        'job_roles', 'duration_minutes', 'assessment_id']
        if not all(col in df.columns for col in required_cols):
            raise ValueError("CSV missing required columns")
            
        # Fill missing values
        df.fillna('', inplace=True)
        
        # Combine relevant columns
        df['text_blob'] = (df['assessment_name'] + ". " + 
                          df['description'] + ". " + 
                          df['skills_measured'] + ". " + 
                          df['job_roles'])
        return df
        
    except Exception as e:
        logger.error(f"Error loading data: {str(e)}")
        raise

# Load data
try:
    df = load_data()
except Exception as e:
    st.error(f"Failed to load data: {str(e)}")
    st.stop()

@st.cache_resource(show_spinner="Loading AI model...")
def load_model():
    """Load the SentenceTransformer model with error handling."""
    try:
        model = SentenceTransformer('all-MiniLM-L6-v2')
        return model
    except Exception as e:
        logger.error(f"Error loading model: {str(e)}")
        st.error("Failed to load AI model. Please check the logs.")
        st.stop()

# Load model
model = load_model()

@st.cache_data(show_spinner="Processing catalog...")
def encode_catalog(_model, texts):
    """Encode the catalog with progress tracking."""
    try:
        return _model.encode(texts, convert_to_tensor=True, show_progress_bar=True)
    except Exception as e:
        logger.error(f"Error encoding catalog: {str(e)}")
        st.error("Failed to process catalog. Please check the logs.")
        st.stop()

# Encode catalog
catalog_embeddings = encode_catalog(model, df['text_blob'].tolist())

def get_top_k(user_query, k=5):
    """Get top k recommendations with error handling."""
    try:
        # Validate input
        if not user_query or not isinstance(user_query, str):
            raise ValueError("Invalid query input")
            
        if k <= 0 or k > len(df):
            k = min(5, len(df))
            
        # Encode query
        query_embedding = model.encode(user_query, convert_to_tensor=True)
        
        # Compute similarities
        cos_scores = util.pytorch_cos_sim(query_embedding, catalog_embeddings)[0]
        
        # Get top results
        top_results = torch.topk(cos_scores, k=k)
        indices = top_results[1].cpu().numpy().tolist()
        
        # Prepare results
        return [{
            'assessment_name': df.iloc[idx]['assessment_name'],
            'description': df.iloc[idx]['description'],
            'duration': df.iloc[idx]['duration_minutes'],
            'url': f"https://example.com/assessment/{df.iloc[idx]['assessment_id']}",
            'score': float(cos_scores[idx])  # Add similarity score
        } for idx in indices]
        
    except Exception as e:
        logger.error(f"Recommendation error: {str(e)}")
        st.error("Error generating recommendations. Please try again.")
        return []
