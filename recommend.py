









"""
SHL Assessment Recommendation Engine Core Module

This module provides recommendation functionality using sentence-transformers
to match job descriptions with relevant SHL assessments.
"""

# Standard library imports
import logging
import time
from typing import List, Dict, Optional, Tuple
import json
import os

# Third-party imports
import pandas as pd
import torch
import streamlit as st
from sentence_transformers import SentenceTransformer, util
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

MIN_SCORE_THRESHOLD = 0.2  # Minimum similarity score to include
MAX_DURATION = 240  # Maximum allowed duration in minutes

# ------------------------------
# Configuration
# ------------------------------
MODEL_NAME = "all-MiniLM-L6-v2"  # Lightweight but effective model
DEFAULT_K = 5  # Default number of recommendations
MIN_QUERY_LENGTH = 10  # Minimum characters for valid query
CACHE_DIR = "model_cache"  # Directory for caching models

# ------------------------------
# Logging Setup
# ------------------------------
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("recommendation_engine.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# ------------------------------
# Data Loading & Validation
# ------------------------------
def load_and_validate_data(file_path: str = 'data/catalog.csv') -> pd.DataFrame:
    """
    Load and validate the product catalog with comprehensive checks,
    including the new columns: remote_testing, adaptive_irt, test_type, url.

    Args:
        file_path: Path to the catalog CSV file

    Returns:
        pandas.DataFrame: Validated and processed catalog data

    Raises:
        ValueError: If data validation fails
    """
    try:
        # Check file existence
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"Catalog file not found at {file_path}")

        # Load data with error handling for malformed files
        try:
            df = pd.read_csv(file_path)
        except pd.errors.EmptyDataError:
            raise ValueError("Catalog file is empty")
        except pd.errors.ParserError:
            raise ValueError("Catalog file is malformed")

        # Required columns check (including new ones)
        required_cols = {
            'assessment_name': str,
            'description': str,
            'skills_measured': str,
            'job_roles': str,
            'duration_minutes': (int, float),
            'assessment_id': (str, int),
            'remote_testing': str,
            'adaptive_irt': str,
            'test_type': str,
            'url': str
        }

        missing_cols = [col for col in required_cols if col not in df.columns]
        if missing_cols:
            raise ValueError(f"Missing required columns: {', '.join(missing_cols)}")

        # Type validation
        type_errors = []
        for col, expected_type in required_cols.items():
            try:
                # Check type of the first non-null element
                first_valid = df[col].first_valid_index()
                if first_valid is not None and not isinstance(df[col].iloc[first_valid], expected_type):
                    type_errors.append(f"{col} should be of type {expected_type}")
            except KeyError:
                pass # Column already checked for existence

        if type_errors:
            raise ValueError(f"Type mismatch: {'; '.join(type_errors)}")

        # Data cleaning
        df.fillna('', inplace=True)

        # Create enhanced searchable text
        df['text_blob'] = (
            "Assessment: " + df['assessment_name'] + ". " +
            "Description: " + df['description'] + ". " +
            "Skills Measured: " + df['skills_measured'] + ". " +
            "Job Roles: " + df['job_roles']
        )

        logger.info(f"Successfully loaded catalog with {len(df)} assessments")
        return df

    except Exception as e:
        logger.error(f"Data loading failed: {str(e)}", exc_info=True)
        raise

# ------------------------------
# Model Management
# ------------------------------
@st.cache_resource(show_spinner="Initializing AI model...")
def initialize_model(model_name: str = MODEL_NAME) -> SentenceTransformer:
    """
    Initialize and cache the sentence transformer model with comprehensive error handling.

    Args:
        model_name: Name of the model to load

    Returns:
        SentenceTransformer: Initialized model

    Raises:
        RuntimeError: If model initialization fails
    """
    try:
        # Create cache directory if it doesn't exist
        os.makedirs(CACHE_DIR, exist_ok=True)

        start_time = time.time()
        logger.info(f"Loading model: {model_name}")

        model = SentenceTransformer(
            model_name,
            cache_folder=CACHE_DIR,
            device='cuda' if torch.cuda.is_available() else 'cpu'
        )

        logger.info(f"Model loaded in {time.time() - start_time:.2f}s on "
                    f"{'GPU' if torch.cuda.is_available() else 'CPU'}")
        return model

    except Exception as e:
        logger.error(f"Model initialization failed: {str(e)}", exc_info=True)
        raise RuntimeError("Failed to initialize recommendation model")

# ------------------------------
# Core Recommendation Function
# ------------------------------
def get_top_k(
    query: str,
    df: pd.DataFrame,
    k: int = DEFAULT_K,
    model: Optional[SentenceTransformer] = None,
    embeddings: Optional[torch.Tensor] = None,
    max_duration: Optional[int] = None,
    min_score: Optional[float] = None
) -> List[Dict]:
    """
    Enhanced with duration filtering and score thresholds, and includes
    remote_testing, adaptive_irt, test_type, and url in recommendations.
    """
    try:
        # Validate inputs
        if not query or len(query.strip()) < MIN_QUERY_LENGTH:
            raise ValueError(f"Query must be at least {MIN_QUERY_LENGTH} characters")

        if k <= 0:
            raise ValueError("Number of recommendations must be positive")

        # Apply duration filter if specified
        if max_duration is not None:
            df = df[df['duration_minutes'] <= max_duration]
            if len(df) == 0:
                raise ValueError(f"No assessments found within {max_duration} minutes")

        # Initialize model if not provided
        if model is None:
            model = initialize_model()

        # Generate embeddings if not provided
        if embeddings is None:
            embeddings = generate_embeddings(model, df['text_blob'].tolist())

        # Encode query
        query_embedding = model.encode(
            query,
            convert_to_tensor=True,
            show_progress_bar=False
        )

        # Compute cosine similarities
        cos_scores = util.pytorch_cos_sim(query_embedding, embeddings)[0]

        # Apply score threshold if specified
        if min_score is None:
            min_score = MIN_SCORE_THRESHOLD

        # Get top k results above threshold
        top_results = torch.topk(cos_scores, k=min(k, len(cos_scores)))

        # Prepare results
        recommendations = []
        for score, idx in zip(top_results.values, top_results.indices):
            if score < min_score:
                continue
            idx_int = idx.item()
            row = df.iloc[idx_int]
            recommendations.append({
                'assessment_name': row['assessment_name'],
                'description': row['description'],
                'skills_measured': row['skills_measured'],
                'job_roles': row['job_roles'],
                'duration_minutes': row['duration_minutes'],
                'assessment_id': row['assessment_id'],
                'score': float(score),
                'url': row['url'],
                'adaptive_irt': row['adaptive_irt'],
                'remote_testing': row['remote_testing'],
                'test_type': row['test_type']
            })

        return recommendations

    except Exception as e:
        logger.error(f"Recommendation failed: {str(e)}", exc_info=True)
        raise RuntimeError(f"Failed to generate recommendations: {str(e)}")


def format_for_api(recommendations: List[Dict]) -> List[Dict]:
    """Convert to API-required JSON format"""
    return [{
        "url": r['url'],
        "adaptive_support": r['adaptive_irt'],
        "description": r['description'],
        "duration": r['duration_minutes'],
        "remote_support": r['remote_testing'],
        "test_type": r['test_type']
    } for r in recommendations]

# New evaluation metrics functions
def calculate_recall(recommendations, relevant_items, k=3):
    """Calculate Recall@K"""
    top_k = recommendations[:k]
    relevant_in_top_k = len(set(top_k) & set(relevant_items))
    return relevant_in_top_k / len(relevant_items) if relevant_items else 0

def calculate_map(recommendations, relevant_items, k=3):
    """Calculate MAP@K"""
    average_precision = 0
    relevant_count = 0

    for i in range(1, k+1):
        if recommendations[i-1] in relevant_items:
            relevant_count += 1
            precision_at_i = relevant_count / i
            average_precision += precision_at_i

    return average_precision / min(k, len(relevant_items)) if relevant_items else 0

# ------------------------------
# Embedding Generation
# ------------------------------
@st.cache_data(show_spinner="Encoding catalog assessments...")
def generate_embeddings(  # Fixed typo in function name (removed extra 'd')
    _model: SentenceTransformer,  # Added underscore to prevent hashing
    texts: List[str],
    batch_size: int = 32
) -> torch.Tensor:
    """
    Generate embeddings for catalog items.

    Args:
        _model: Initialized sentence transformer model (not hashed)
        texts: List of text items to encode
        batch_size: Number of items to process at once

    Returns:
        torch.Tensor: Matrix of embeddings
    """
    try:
        return _model.encode(
            texts,
            convert_to_tensor=True,
            show_progress_bar=True,
            batch_size=batch_size,
            normalize_embeddings=True
        )
    except Exception as e:
        logger.error(f"Embedding generation failed: {str(e)}", exc_info=True)
        raise RuntimeError("Failed to generate embeddings")

# ------------------------------
# Main Execution (for testing)
# ------------------------------
if __name__ == "__main__":
    # Test the recommendation system
    print("Testing recommendation system...")
    try:
        test_df = load_and_validate_data()
        test_query = "sales manager with communication and leadership skills"
        recommendations = get_top_k(test_query, test_df)
        print(f"Recommendations for '{test_query}':")
        print(json.dumps(recommendations, indent=2))
    except Exception as e:
        print(f"Error: {str(e)}")
