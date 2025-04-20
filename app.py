import streamlit as st
import joblib
import re
import PyPDF2
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Load model & vectorizer
model = joblib.load('ats_nb_model.pkl')
vectorizer = joblib.load('ats_vectorizer.pkl')

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

# Match score
def get_match_score(resume_text, job_text):
    cleaned_resume = clean_text(resume_text)
    cleaned_jd = clean_text(job_text)

    tfidf = TfidfVectorizer()
    vectors = tfidf.fit_transform([cleaned_resume, cleaned_jd])
    score = cosine_similarity(vectors[0:1], vectors[1:2])[0][0]
    return round(score * 100, 2)

# UI
st.title("📄 ATS Resume Category Predictor")
st.markdown("Upload a resume or paste job description to predict the category, or check resume-job match score.")

# Tabs
tab1, tab2, tab3 = st.tabs(["📤 Upload Resume", "✍️ Paste Job Description", "🧪 Match Resume with Job Description"])

# --- Tab 1: Resume Upload
with tab1:
    uploaded_pdf = st.file_uploader("Upload your resume PDF file", type=['pdf'], key="resume_tab1")
    if uploaded_pdf:
        extracted = extract_text_from_pdf(uploaded_pdf)
        cleaned = clean_text(extracted)
        vectorized = vectorizer.transform([cleaned]).toarray()
        prediction = model.predict(vectorized)[0]
        st.success(f"🧠 Predicted Resume Category: **{prediction}**")

# --- Tab 2: JD Input
with tab2:
    jd_text = st.text_area("Paste Job Description here", key="jd_tab2")
    if st.button("Predict Category from JD", key="predict_jd_btn"):
        if jd_text.strip() == "":
            st.warning("Please enter a job description.")
        else:
            cleaned = clean_text(jd_text)
            vectorized = vectorizer.transform([cleaned]).toarray()
            prediction = model.predict(vectorized)[0]
            st.success(f"🧠 Predicted JD Category: **{prediction}**")

# --- Tab 3: Match Score
with tab3:
    st.write("Upload your resume and paste a job description to check how well they match.")
    match_pdf = st.file_uploader("Upload Resume PDF", type=['pdf'], key="resume_tab3")
    match_jd = st.text_area("Paste Job Description (optional)", key="jd_tab3")

    if st.button("Check Match Score", key="match_btn"):
        if match_pdf is None:
            st.warning("Please upload a resume to check match.")
        else:
            resume_text = extract_text_from_pdf(match_pdf)
            if match_jd.strip() == "":
                st.info("Job description not provided. Showing resume text only.")
                st.success("✅ Resume uploaded successfully, but no JD to match with.")
            else:
                score = get_match_score(resume_text, match_jd)
                st.markdown(f"### 🔍 Match Score: **{score}%**")
                if score >= 60:
                    st.success("✅ Good match with the job description!")
                else:
                    st.error("❌ Resume does not align well with the job description.")
