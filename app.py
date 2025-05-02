### app.py (Flask Backend)

from flask import Flask, request, jsonify
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

app = Flask(__name__)

# Load the detailed SHL product catalogue
df = pd.read_csv("shl_catalogue_detailed.csv")

# Create a combined column for NLP feature extraction
df["combined"] = df["Skills Covered"] + " " + df["Job Roles"] + " " + df["Description"]

# Initialize and fit TF-IDF Vectorizer to the combined text
vectorizer = TfidfVectorizer(stop_words="english")
tfidf_matrix = vectorizer.fit_transform(df["combined"])

@app.route("/recommend", methods=["POST"])
def recommend():
    """
    Recommends top 5 assessments based on user input.
    Accepts JSON body: {"text": "..."}
    Returns: JSON list of recommended assessments.
    """
    user_input = request.json.get("text", "")
    if not user_input:
        return jsonify({"error": "No input provided"}), 400

    user_tfidf = vectorizer.transform([user_input])
    similarity_scores = cosine_similarity(user_tfidf, tfidf_matrix)
    top_indices = similarity_scores[0].argsort()[::-1][:5]

    recommendations = df.iloc[top_indices][["Assessment Name", "Skills Covered", "Job Roles", "Description"]].to_dict(orient="records")

    return jsonify({"recommendations": recommendations})

if __name__ == "__main__":
    app.run(debug=True, port=5000)
