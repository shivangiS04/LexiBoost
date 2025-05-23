# ✨ LexiBoost-Automated Essay Scoring (AES)

An AI-powered web application that automatically evaluates and scores student essays using natural language processing (NLP) and machine learning techniques.

## 🚀 Project Overview

**Automated Essay Scoring (AES)** leverages machine learning models to predict essay scores based on linguistic and structural features. The system includes a user-friendly web interface built with HTML, Bootstrap, and JavaScript, which communicates with a Flask-based backend API for real-time scoring.

### 🎯 Key Features

- 📝 Accepts user essays through a clean web UI  
- ⚙️ Preprocesses and vectorizes input essays  
- 🤖 Predicts scores using trained machine learning models  
- 📊 Normalizes and interprets scores across different essay prompts  
- 🌐 Hosted on Render for public access

---

## 🧠 Technologies Used

### 💻 Frontend
- HTML5, CSS3
- Bootstrap 4.4
- JavaScript (vanilla)

### 🔙 Backend
- Flask 3.0.3
- Flask-CORS
- Gunicorn

### 🧠 Machine Learning
- scikit-learn (Random Forest, SVR, Linear Regression)
- NLTK for text preprocessing
- Gensim
- TensorFlow/Keras (for potential deep learning extensions)

### 📦 Other Libraries
- Pandas, NumPy
- Matplotlib, Seaborn (for EDA)
- Regular Expressions (re)

---

## 🗂 Dataset

The model is trained on the **ASAP Automated Student Assessment Prize** dataset provided by Kaggle.

- 📄 Format: `.tsv` file with essays and human-assigned scores  
- 📊 8 different essay prompts, each with varying score ranges  
- 🧹 Extensive preprocessing: punctuation removal, stopword filtering, tokenization, POS tagging, and spell-checking.

---

## 🧪 Model Training Pipeline

1. **Data Cleaning & Preprocessing**
   - Removing usernames, punctuation, and stopwords
   - Tokenizing and POS tagging for feature extraction

2. **Feature Engineering**
   - Word count, character count, average word length
   - POS-based features (nouns, verbs, adjectives, adverbs)
   - Misspelled words count
   - CountVectorizer-based n-grams

3. **Modeling**
   - Normalization of scores for uniform scaling
   - Training ML regressors (Random Forest, SVR, Linear Regression)
   - Evaluation using Mean Squared Error (MSE)

4. **Scoring API**
   - Flask API receives essays and returns predicted scores scaled to a 10-point system
