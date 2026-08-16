# SentimentAnalysis

A machine learning project that analyzes movie reviews and classifies them as positive or negative using Natural Language Processing (NLP).

## 📌 Project Overview

This project uses the IMDb movie review dataset to build a sentiment classification system.

The text data is cleaned and transformed using TF-IDF, followed by training multiple machine learning models.

The models compared were:

- Logistic Regression
- Multinomial Naive Bayes
- Linear SVM

Linear SVM achieved the best performance with approximately 89.92% accuracy and a 90.01% F1-score.

## 🛠️ Technologies Used

- Python
- Pandas
- NumPy
- NLTK
- Scikit-learn
- Matplotlib
- Seaborn
- Streamlit
- Joblib

## 🔄 Machine Learning Pipeline

Dataset
↓
Data Cleaning
↓
Exploratory Data Analysis
↓
Text Preprocessing
↓
TF-IDF Vectorization
↓
Model Training
↓
Model Comparison
↓
Linear SVM
↓
Sentiment Prediction

## 📊 Model Performance

| Model | Accuracy | F1 Score |
|---|---:|---:|
| Logistic Regression | 89.71% | 89.88% |
| Naive Bayes | 87.98% | 88.10% |
| Linear SVM | 89.92% | 90.01% |

## 🚀 Run Locally

Clone the repository:

```bash
git clone YOUR_GITHUB_REPOSITORY_URL
