# 📊 Sentiment Analysis with RAG-Enhanced Retrieval Pipeline

This repository contains an end-to-end sentiment analysis system that compares two different retrieval strategies:

1. **Non-RAG (baseline):**  
   Standard sentiment scoring and filtering over the full dataset.

2. **RAG-Enhanced Retrieval:**  
   Query-based semantic retrieval using SentenceTransformer embeddings, followed by sentiment aggregation and summary generation.

The project includes data cleaning, feature engineering, embedding generation, model inference, and detailed visualizations.  
It is written in a fully modular and scalable structure suitable for production-level sentiment analytics.

---

## 🚀 Features

### ✔️ **Data Cleaning Module**
- Lowercasing, whitespace normalization  
- Stopword removal  
- Special character cleanup  
- Config-driven preprocessing  

### ✔️ **Embedding Module**
- SentenceTransformer embeddings  
- Pre-computed embedding matrix for fast retrieval  
- Cosine similarity ranking  

### ✔️ **Sentiment Model**
- Transformer-based sentiment inference  
- Outputs: label + probability + normalized sentiment score  

### ✔️ **RAG Retrieval Pipeline**
- Query embedding generation  
- Similarity search against embedding matrix  
- Filtering and ranking  
- Subset scoring + summary generation  

### ✔️ **Visualization Tools**
- Sentiment score distributions  
- Query-based comparison charts  
- RAG vs Non-RAG result comparison  

---

## 📁 Project Structure
```
Sentiment-Analysis/
│
├── full_sentiment_rag_pipeline.ipynb
│   
│
├── utils/
│   ├── cleaning.py
│   ├── embedding.py
│   ├── sentiment_model.py
│   ├── visualize.py
│   └── config.yaml
│
├── data/
│   └── .gitignore
│
├── README.md
├── requirements.txt
└── .gitignore
```
