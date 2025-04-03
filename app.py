import gradio as gr
import joblib
from transformers import pipeline
import torch
import pandas as pd

# Load models
clf = joblib.load("churn_model.pkl")
vec = joblib.load("tfidf_vectorizer.pkl")
sentiment_model = pipeline("sentiment-analysis", model="distilbert-base-uncased-finetuned-sst-2-english", device=0 if torch.cuda.is_available() else -1)
flan = pipeline("text2text-generation", model="google/flan-t5-base", device=0 if torch.cuda.is_available() else -1)

# Main logic
def analyze(feedback):
    if not feedback or len(feedback.split()) < 4:
        return "Please provide more detailed feedback."

    # Churn
    X_new = vec.transform([feedback])
    prob = clf.predict_proba(X_new)[0][1]
    churn = "Yes" if prob > 0.5 else "No"

    # Sentiment
    sentiment = sentiment_model(feedback)[0]['label']

    # Routing
    if "debit card" in feedback.lower():
        route = "Forwarded to Debit Card Team"
    elif "app" in feedback.lower():
        route = "Forwarded to Mobile App Support"
    else:
        route = "General Support"

    # Retention Strategy
    prompt = f"""Customer Feedback: {feedback}\nSentiment: {sentiment}\nChurn Risk: {churn}\n\nGenerate a helpful response."""
    strategy = flan(prompt, max_new_tokens=100)[0]['generated_text'].strip()

    # Basic post-processing
    if strategy.lower().startswith("1. no 2. no"):
        strategy = "Customer is not likely to churn. No specific retention action required."

    return f"""📈 Churn Prediction: {churn} (Confidence: {prob:.2f})\n😊 Sentiment: {sentiment}\n🚀 Routing: {route}\n💬 Strategy: {strategy}"""

# UI
demo = gr.Interface(fn=analyze, inputs="text", outputs="text", title="💬 Churn & Retention Assistant")
demo.launch()
