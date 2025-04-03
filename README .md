
# 🤖 Customer Churn Prediction & Retention Strategy Chatbot

A comprehensive AI-powered solution for XYZ Bank to predict customer churn, understand dissatisfaction, and generate personalized retention strategies using ML, NLP, and Generative AI.

🔗 **Live Demo:** [Hugging Face Space](https://huggingface.co/spaces/MyTrunk/churn-predictor-bot)

---

## 🧠 Project Overview

This project fulfills the complete Use Case 1.2 requirement for XYZ Bank:

- 📉 Predict customer churn from structured data
- 💬 Analyze customer feedback (text) using sentiment analysis
- 🤖 Generate real-time personalized retention responses
- 🔁 Route customer queries to relevant teams
- 🤝 Provide a complete conversational interface

---

## 🗂️ Folder Structure

```
churn-chatbot-app/
│
├── app.py                   # 🧠 Gradio-based chatbot pipeline
├── requirements.txt         # 📦 Python libraries for deployment
├── customer_data.csv        # 📊 Raw customer dataset
├── churn_model.pkl          # 💾 Trained churn model (Logistic Regression + TF-IDF)
├── tfidf_vectorizer.pkl     # 💾 Trained TF-IDF vectorizer
├── README.md                # 📖 Project documentation
```

---

## 🧱 System Architecture

Here’s the **block diagram** illustrating the full pipeline:

```
[Customer Feedback] + [Customer Profile Data]
             │
             ▼
🔮 Step 1: Churn Prediction (XGBoost / TF-IDF + LR)
             │
             ▼
📊 Step 2: Sentiment Analysis (Transformers model)
             │
             ▼
🧠 Step 3: Retention Strategy Generation (FLAN-T5)
             │
             ▼
📦 Step 4: Routing Logic (Keyword-based)
             │
             ▼
💬 Step 5: Chatbot Output via Gradio
             │
             ▼
🗃️ Step 6: (Optional) Store in Database
```

🖼️ **Block Diagram Image**:  
You can optionally add this image to your repo and reference it like this:

```markdown
![Block Diagram](https://huggingface.co/spaces/MyTrunk/churn-predictor-bot/resolve/main/architecture.png)
```

---

## ✅ Solution Highlights

### 1. 📊 Churn Prediction Models

- **Random Forest & XGBoost (Structured Data)**
- **TF-IDF + Logistic Regression (Text-Based Feedback)**
- Balanced with class weights
- Feature importance visualization

### 2. 💬 Sentiment Analysis

- Hugging Face `distilbert-base-uncased-finetuned-sst-2-english`
- Batch inference on all feedback
- Distribution plots for exploratory insights

### 3. 🧠 Retention Strategy Generation

- Prompt-based inference using FLAN-T5
- Customized to each customer’s feedback
- Includes fallback logic for edge cases

### 4. 🧠 Intelligent Routing

- Keyword-based routing to:
  - Debit Card Team
  - Mobile App Support
  - General Support

### 5. 🤖 Gradio Chatbot UI

- Fully interactive chatbot
- Works on live customer inputs
- Returns:
  - Churn Prediction + Confidence
  - Sentiment
  - Routing Info
  - Retention Strategy

---

## 🚀 Deployment (Hugging Face Spaces)

1. ✅ Create your space at [huggingface.co/spaces](https://huggingface.co/spaces)
2. ✅ Clone and copy app files:
   ```bash
   git lfs install
   git clone https://huggingface.co/spaces/MyTrunk/churn-predictor-bot
   cd churn-predictor-bot
   cp -r /path/to/churn-chatbot-app/* .
   git add .
   git commit -m "Initial commit"
   git push
   ```

✅ Your app will be live in ~2 minutes!

---

## 🛠️ Dependencies

```text
transformers==4.50.3
torch
pandas
scikit-learn
gradio
matplotlib
seaborn
joblib
xgboost
```

---

## 🔐 Ethics & Privacy

- Customer data is anonymized
- Models avoid demographic bias
- Feedback routing and clarification prevent misclassification

---

## 🔮 Future Enhancements

- 🔗 Integrate SQLite/MySQL for query logging
- 📊 Dashboard with visual feedback analytics
- 🌍 Multilingual support (MarianMT, BLOOM)
- 🧠 Personalization via user profile embeddings

---

## 👨‍💻 Developed By

- **Author:** `MyTrunk`
- **Role:** AI Engineer
- **Platform:** Hugging Face Spaces + Scikit-Learn + Transformers + Gradio
