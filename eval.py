# eval.py

from recommend import get_top_k
import pandas as pd
from sklearn.metrics import precision_score, recall_score, f1_score

# Load test cases or validation prompts
test_data = [
    {"query": "Assess leadership qualities in managers", "expected": "Leadership Assessment"},
    {"query": "Evaluate programming skills of fresh graduates", "expected": "Coding Test"},
    {"query": "Measure communication and teamwork", "expected": "Soft Skills Test"},
    {"query": "Recruit customer support agents", "expected": "Customer Service Assessment"}
]

# Initialize lists for evaluation
y_true = []
y_pred = []

print("Evaluating Recommendation Engine...\n")

for item in test_data:
    query = item["query"]
    expected = item["expected"].lower()
    recommendations = get_top_k(query, k=3)
    predicted_names = [rec["assessment_name"].lower() for rec in recommendations]

    y_true.append(expected)
    if expected in predicted_names:
        y_pred.append(expected)
    else:
        y_pred.append(predicted_names[0] if predicted_names else "none")

    print(f"Query: {query}")
    print(f"Expected: {expected}")
    print(f"Predicted: {predicted_names}\n")

# Compute evaluation metrics
def label_match(y_true, y_pred):
    return [1 if yt == yp else 0 for yt, yp in zip(y_true, y_pred)]

precision = precision_score(label_match(y_true, y_pred), [1]*len(y_true), average='binary')
recall = recall_score(label_match(y_true, y_pred), [1]*len(y_true), average='binary')
f1 = f1_score(label_match(y_true, y_pred), [1]*len(y_true), average='binary')

print("Final Evaluation Metrics:")
print(f"Precision: {precision:.2f}")
print(f"Recall: {recall:.2f}")
print(f"F1 Score: {f1:.2f}")
