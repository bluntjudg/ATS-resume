import streamlit as st
import joblib
import re
import PyPDF2
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

# Load model, vectorizer, and centroids
model = joblib.load('ats_nb_model.pkl')
vectorizer = joblib.load('ats_vectorizer.pkl')
centroids = joblib.load('category_centroids.pkl')  # Must match model's labels

# Clean text
def clean_text(text):
    text = re.sub(r'<[^>]+>', ' ', text)
    text = re.sub(r'[^a-zA-Z]', ' ', text)
    text = text.lower().split()
    return ' '.join(text)

# Extract text from PDF
def extract_text_from_pdf(pdf_file):
    text = ""
    reader = PyPDF2.PdfReader(pdf_file)
    for page in reader.pages:
        page_text = page.extract_text()
        if page_text:
            text += page_text
    return text

# Score label
def score_label(score):
    if score >= 80:
        return "🟢 Excellent match"
    elif score >= 60:
        return "🟡 Good match"
    else:
        return "🔴 Needs improvement"

# Streamlit UI
st.title("📄 ATS Resume Analyzer")
st.markdown("Upload your resume to get the predicted category and how well your resume fits that field.")

tab1, tab2 = st.tabs(["📤 Upload Resume", "✍️ Paste Job Description"])

# === Tab 1: Upload Resume ===
with tab1:
    uploaded_pdf = st.file_uploader("Upload your resume PDF file", type=['pdf'])
    if uploaded_pdf is not None:
        extracted = extract_text_from_pdf(uploaded_pdf)
        cleaned = clean_text(extracted)
        vectorized = vectorizer.transform([cleaned]).toarray()
        prediction = model.predict(vectorized)[0]

        # Resume Similarity Score
        centroid_vec = centroids[prediction].reshape(1, -1)
        similarity_score = cosine_similarity(vectorized, centroid_vec)[0][0] * 100

        st.success(f"🧠 Predicted Resume Category: **{prediction}**")
        st.info(f"📊 Resume Similarity Score: **{similarity_score:.2f}%** ({score_label(similarity_score)})")
        st.progress(int(similarity_score))

# === Tab 2: Paste JD (Optional) ===
with tab2:
    jd_text = st.text_area("Paste Job Description here")
    if st.button("Predict Category from JD"):
        if jd_text.strip() == "":
            st.warning("Please enter a job description.")
        else:
            cleaned = clean_text(jd_text)
            vectorized = vectorizer.transform([cleaned]).toarray()
            prediction = model.predict(vectorized)[0]
            st.success(f"🧠 Predicted JD Category: **{prediction}**")
