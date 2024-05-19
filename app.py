import streamlit as st
import joblib
import numpy as np
import re
from nltk.stem.snowball import SnowballStemmer

# Load the model and vectorizer
model = joblib.load('logistic_regression_cv_model.joblib')
tfidf = joblib.load('tfidf_vectorizer.joblib')

# Preprocess the input text
def preprocessor(text):
    emoticons = re.findall('(?::|;|=)(?:-)?(?:\)|\(|D|P)', text)
    text = re.sub('[\W]+', ' ', text.lower()) + ' '.join(emoticons).replace('-', '')
    return text

snowball = SnowballStemmer("english")

def process_text(text):
    text = preprocessor(text)
    tokens = text.split()
    stems = [snowball.stem(word) for word in tokens]
    stop_words = set(['a', 't', 'd', 'y', 'it', 'that'])
    final_tokens = [word for word in stems if word not in stop_words]
    return ' '.join(final_tokens)

# Streamlit app
st.title("Restaurant Review Sentiment Analysis")

review = st.text_area("Enter a restaurant review:")

if st.button("Predict"):
    processed_review = process_text(review)
    review_tfidf = tfidf.transform([processed_review])
    prediction = model.predict(review_tfidf)
    prediction_proba = model.predict_proba(review_tfidf)
    st.write(f"Prediction: {'Liked' if prediction[0] == 1 else 'Not Liked'}")
    st.write(f"Prediction Probability: {prediction_proba[0][prediction[0]]:.2f}")
