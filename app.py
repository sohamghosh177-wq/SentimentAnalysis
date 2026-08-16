import streamlit as st
import joblib
import re
import os
from nltk.corpus import stopwords
import nltk

# Download NLTK stopwords if needed
try:
    stop_words = set(stopwords.words('english'))
except LookupError:
    nltk.download('stopwords')
    stop_words = set(stopwords.words('english'))


# -----------------------------
# Load trained model
# -----------------------------

model = joblib.load(
    "models/sentiment_model.pkl"
)

tfidf = joblib.load(
    "models/tfidf_vectorizer.pkl"
)


# -----------------------------
# Text cleaning function
# -----------------------------

def clean_text(text):

    text = text.lower()

    # Remove HTML tags
    text = re.sub(r'<.*?>', '', text)

    # Remove URLs
    text = re.sub(r'http\S+|www\S+', '', text)

    # Remove punctuation and numbers
    text = re.sub(r'[^a-zA-Z\s]', '', text)

    # Remove extra spaces
    text = re.sub(r'\s+', ' ', text).strip()

    # Remove stopwords
    words = text.split()
    words = [
        word for word in words
        if word not in stop_words
    ]

    return ' '.join(words)


# -----------------------------
# Streamlit UI
# -----------------------------

st.set_page_config(
    page_title="Sentiment Analyzer",
    page_icon="🎬",
    layout="centered"
)

st.title("🎬 Movie Sentiment Analyzer")

st.write(
    "Enter a movie review and the machine-learning model "
    "will predict whether the sentiment is Positive or Negative."
)


review = st.text_area(
    "Enter your movie review:",
    placeholder="Example: This movie was absolutely fantastic!"
)


if st.button("Analyze Sentiment"):

    if review.strip() == "":
        st.warning("Please enter a review.")

    else:

        # Clean text
        cleaned_review = clean_text(review)

        # Convert text to TF-IDF
        review_tfidf = tfidf.transform(
            [cleaned_review]
        )

        # Prediction
        prediction = model.predict(
            review_tfidf
        )[0]

        if prediction == 1:

            st.success("😊 Positive Sentiment")

        else:

            st.error("😡 Negative Sentiment")