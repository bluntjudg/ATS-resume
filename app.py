import streamlit as st
import joblib
import re

# Title
st.title("📄 ATS Resume Category Predictor")
st.markdown("Use this simple ML app to match a job description to a resume category using a Naive Bayes model.")

# Load model and vectorizer
model = joblib.load('ats_nb_model.pkl')
vectorizer = joblib.load('ats_vectorizer.pkl')

# Text cleaner
def clean_text(text):
    text = re.sub(r'<[^>]+>', ' ', text)
    text = re.sub(r'[^a-zA-Z]', ' ', text)
    text = text.lower().split()
    return ' '.join(text)

# Input section
job_description = st.text_area("📝 Paste Job Description here")

# Predict button
if st.button("🔍 Predict Category"):
    if job_description.strip() == "":
        st.warning("Please enter a job description.")
    else:
        cleaned = clean_text(job_description)
        vectorized = vectorizer.transform([cleaned]).toarray()
        prediction = model.predict(vectorized)[0]
        st.success(f"✅ Predicted Resume Category: **{prediction}**")
