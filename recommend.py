# recommend.py
from sentence_transformers import SentenceTransformer, util
import pandas as pd

model = SentenceTransformer('all-MiniLM-L6-v2')
catalog = pd.read_csv('data/shl_catalog.csv')
catalog['embedding'] = catalog['description'].apply(lambda x: model.encode(x, convert_to_tensor=True))

def get_top_k(query, k=10):
    query_emb = model.encode(query, convert_to_tensor=True)
    scores = [util.pytorch_cos_sim(query_emb, emb)[0][0].item() for emb in catalog['embedding']]
    catalog['score'] = scores
    top = catalog.sort_values(by='score', ascending=False).head(k)
    return top[['assessment_name', 'description', 'duration', 'url']].to_dict(orient='records')
