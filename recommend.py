# recommend.py

import pandas as pd
from sentence_transformers import SentenceTransformer, util
import torch

# Load the product catalogue
df = pd.read_csv('data/shl_catalog.csv')

# Ensure required columns are present
required_cols = ['assessment_name', 'description', 'duration', 'url']
if not all(col in df.columns for col in required_cols):
    raise ValueError("CSV missing required columns")

# Fill missing values to avoid errors
df.fillna('', inplace=True)

# Combine relevant columns to form a searchable corpus
df['text_blob'] = df['assessment_name'] + ". " + df['description']

# Load pre-trained transformer model
model = SentenceTransformer('all-MiniLM-L6-v2')

# Encode all SHL assessments in the catalogue
@st.cache_data(show_spinner=False)
def encode_catalog():
    return model.encode(df['text_blob'].tolist(), convert_to_tensor=True, show_progress_bar=True)

catalog_embeddings = encode_catalog()

# Define the recommendation function
def get_top_k(user_query, k=5):
    # Encode the input query
    query_embedding = model.encode(user_query, convert_to_tensor=True)

    # Compute cosine similarities
    cos_scores = util.pytorch_cos_sim(query_embedding, catalog_embeddings)[0]

    # Get top k result indexes
    top_results = torch.topk(cos_scores, k=k)
    indices = top_results[1].cpu().numpy()

    # Prepare results
    results = []
    for idx in indices:
        results.append({
            'assessment_name': df.iloc[idx]['assessment_name'],
            'description': df.iloc[idx]['description'],
            'duration': df.iloc[idx]['duration'],
            'url': df.iloc[idx]['url']
        })

    return results
