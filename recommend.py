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
def load_and_validate_data(file_path: str = 'data/catalogue.csv') -> pd.DataFrame:
    """
    Load and validate the product catalog with comprehensive checks.
    
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

        # Required columns check
        required_cols = {
            'assessment_name': str,
            'description': str,
            'skills_measured': str,
            'job_roles': str,
            'duration_minutes': (int, float),
            'assessment_id': (str, int)
        }
        
        missing_cols = [col for col in required_cols if col not in df.columns]
        if missing_cols:
            raise ValueError(f"Missing required columns: {', '.join(missing_cols)}")
            
        # Type validation
        type_errors = []
        for col, expected_type in required_cols.items():
            if not isinstance(df[col].iloc[0], expected_type):
                type_errors.append(f"{col} should be {expected_type}")
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
        
        # Add metadata
        df['metadata'] = df.apply(lambda row: {
            'duration': row['duration_minutes'],
            'id': str(row['assessment_id'])
        }, axis=1)
        
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
# Embedding Generation
# ------------------------------
@st.cache_data(show_spinner="Encoding catalog assessments...")
def generate_embeddings(
    _model: SentenceTransformer, 
    texts: List[str],
    batch_size: int = 32
) -> torch.Tensor:
    """
    Generate embeddings for catalog items with progress tracking and batching.
    
    Args:
        _model: Initialized sentence transformer model
        texts: List of text items to encode
        batch_size: Number of items to process at once
        
    Returns:
        torch.Tensor: Matrix of embeddings (num_items x embedding_dim)
    """
    try:
        start_time = time.time()
        logger.info(f"Generating embeddings for {len(texts)} items")
        
        embeddings = _model.encode(
            texts,
            convert_to_tensor=True,
            show_progress_bar=True,
            batch_size=batch_size,
            normalize_embeddings=True
        )
        
        logger.info(f"Embeddings generated in {time.time() - start_time:.2f}s")
        return embeddings
        
    except Exception as e:
        logger.error(f"Embedding generation failed: {str(e)}", exc_info=True)
        raise RuntimeError("Failed to generate catalog embeddings")

# ------------------------------
# Core Recommendation Logic
# ------------------------------
def get_top_k(
    user_query: str,
    k: int = DEFAULT_K,
    score_threshold: float = 0.2,
    diversity: bool = True,
    _model: Optional[SentenceTransformer] = None,
    _embeddings: Optional[torch.Tensor] = None,
    _df: Optional[pd.DataFrame] = None
) -> List[Dict]:
    """
    Get top-k recommendations for a user query with enhanced features.
    
    Args:
        user_query: Input job description or query text
        k: Number of recommendations to return
        score_threshold: Minimum similarity score (0-1)
        diversity: Whether to apply diversity filtering
        _model: Optional pre-loaded model (for caching)
        _embeddings: Optional pre-computed embeddings
        _df: Optional pre-loaded catalog data
        
    Returns:
        List of recommendation dictionaries with scores and metadata
    """
    # Validate inputs
    try:
        if not user_query or not isinstance(user_query, str):
            raise ValueError("Query must be a non-empty string")
            
        if len(user_query.strip()) < MIN_QUERY_LENGTH:
            raise ValueError(f"Query too short (min {MIN_QUERY_LENGTH} chars)")
            
        if k <= 0:
            raise ValueError("Number of recommendations must be positive")
            
        # Load data if not provided
        df = _df if _df is not None else load_and_validate_data()
        
        # Initialize model if not provided
        model = _model if _model is not None else initialize_model()
        
        # Generate embeddings if not provided
        embeddings = (_embeddings if _embeddings is not None 
                     else generate_embeddings(model, df['text_blob'].tolist()))
        
        # Encode query
        start_time = time.time()
        query_embedding = model.encode(
            user_query,
            convert_to_tensor=True,
            show_progress_bar=False
        )
        
        # Compute similarities (using both pytorch and sklearn for validation)
        cos_scores_pytorch = util.pytorch_cos_sim(query_embedding, embeddings)[0]
        cos_scores_sklearn = cosine_similarity(
            query_embedding.cpu().numpy().reshape(1, -1),
            embeddings.cpu().numpy()
        )[0]
        
        # Validate consistency between methods
        if not np.allclose(cos_scores_pytorch, cos_scores_sklearn, atol=1e-4):
            logger.warning("Similarity score discrepancy between methods")
        
        # Get top results
        top_results = torch.topk(cos_scores_pytorch, k=min(k * 2, len(df)))
        
        # Apply diversity filter if enabled
        if diversity:
            indices = []
            current_scores = []
            for idx, score in zip(top_results.indices, top_results.values):
                if len(indices) >= k:
                    break
                if score < score_threshold:
                    continue
                    
                # Check similarity with already selected items
                if indices:
                    max_sim = max(
                        util.pytorch_cos_sim(
                            embeddings[idx].reshape(1, -1),
                            embeddings[i].reshape(1, -1)
                        ).item()
                        for i in indices
                    )
                    if max_sim > 0.7:  # Skip too similar items
                        continue
                        
                indices.append(idx)
                current_scores.append(score)
        else:
            indices = top_results.indices[:k].tolist()
            current_scores = top_results.values[:k].tolist()
        
        # Prepare enhanced results
        results = []
        for idx, score in zip(indices, current_scores):
            row = df.iloc[idx]
            results.append({
                'assessment_id': str(row['assessment_id']),
                'assessment_name': row['assessment_name'],
                'description': row['description'],
                'skills': row['skills_measured'],
                'job_roles': row['job_roles'],
                'duration': row['duration_minutes'],
                'url': f"https://platform.shl.com/assessment/{row['assessment_id']}",
                'score': float(score),
                'score_percentage': f"{float(score) * 100:.1f}%",
                'metadata': {
                    'embedding_version': MODEL_NAME,
                    'processing_time': f"{time.time() - start_time:.3f}s"
                }
            })
        
        logger.info(f"Generated {len(results)} recommendations for query: '{user_query[:50]}...'")
        return results
        
    except Exception as e:
        logger.error(f"Recommendation failed for query '{user_query[:50]}...': {str(e)}", exc_info=True)
        raise RuntimeError(f"Recommendation error: {str(e)}")

# ------------------------------
# Main Execution (for testing)
# ------------------------------
if __name__ == "__main__":
    # Test the recommendation system
    print("Testing recommendation system...")
    try:
        test_query = "sales manager with communication and leadership skills"
        recommendations = get_top_k(test_query)
        print(f"Recommendations for '{test_query}':")
        print(json.dumps(recommendations, indent=2))
    except Exception as e:
        print(f"Error: {str(e)}")
