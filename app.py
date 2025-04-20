import streamlit as st
import joblib
import re
import PyPDF2
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

# Load model, vectorizer, and centroids
model = joblib.load('ats_nb_model.pkl')
vectorizer = joblib.load('ats_vectorizer.pkl')
centroids = joblib.load('category_centroids.pkl')

# Dynamically get the list of categories from centroids
resume_categories = list(centroids.keys())

# Helper functions
def clean_text(text):
    text = re.sub(r'<[^>]+>', ' ', text)
    text = re.sub(r'[^a-zA-Z]', ' ', text)
    text = text.lower().split()
    return ' '.join(text)

def extract_text_from_pdf(pdf_file):
    text = ""
    reader = PyPDF2.PdfReader(pdf_file)
    for page in reader.pages:
        text += page.extract_text()
    return text

def calculate_similarity(user_input, selected_category):
    if selected_category not in centroids:
        return None  # Category not found
    cleaned_input = clean_text(user_input)
    user_vec = vectorizer.transform([cleaned_input]).toarray()
    centroid_vec = centroids[selected_category].reshape(1, -1)
    similarity = cosine_similarity(user_vec, centroid_vec)[0][0]
    return round(similarity * 100, 2)

# Streamlit UI
st.title("📄 ATS Resume Category Checker")
st.markdown("Upload a resume or paste a job description to analyze and match it with job categories.")

# Tabs
tab1, tab2, tab3 = st.tabs(["📤 Upload Resume", "✍️ Paste Job Description", "📊 Resume vs Job Category"])

# 📤 Upload Resume Tab
with tab1:
    uploaded_pdf = st.file_uploader("Upload your resume PDF file", type=['pdf'])
    if uploaded_pdf is not None:
        extracted = extract_text_from_pdf(uploaded_pdf)
        cleaned = clean_text(extracted)
        vectorized = vectorizer.transform([cleaned]).toarray()
        prediction = model.predict(vectorized)[0]
        st.success(f"🧠 Predicted Resume Category: **{prediction}**")

# ✍️ Paste Job Description Tab
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

# 📊 Category Match Score Tab
with tab3:
    st.markdown("### Upload Resume or Paste Description to Match Against Selected Category")
    selected_category = st.selectbox("Select Job Category to Match", options=[""] + resume_categories)
    source_text = ""

    col1, col2 = st.columns(2)
    with col1:
        uploaded_resume = st.file_uploader("Upload Resume for Matching", type=['pdf'], key='match_pdf')
        if uploaded_resume:
            source_text = extract_text_from_pdf(uploaded_resume)
    with col2:
        pasted_text = st.text_area("Or Paste Resume/Description Here", key="match_text")

    if pasted_text.strip() and not source_text:
        source_text = pasted_text.strip()

    if selected_category and source_text:
        match_score = calculate_similarity(source_text, selected_category)
        if match_score is not None:
            st.success(f"📈 Resume matches **{selected_category}** with a score of: **{match_score}%**")
        else:
            st.error(f"Selected category '{selected_category}' not found in training data.")
    elif not selected_category:
        st.info("Select a job category to match against.")