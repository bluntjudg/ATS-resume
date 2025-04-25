import streamlit as st
import joblib
import re
import PyPDF2

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
        text += page.extract_text()
    return text

# Streamlit UI
st.title("📄 ATS Resume Predictor")
st.markdown("Upload a resume PDF or paste job description to get the predicted category and resume score.")

# Tabs for options
tab1, tab2 = st.tabs(["📤 Upload Resume", "✍️ Paste Job Description"])

# 📤 Resume Upload Tab
with tab1:
    uploaded_pdf = st.file_uploader("Upload your resume PDF file", type=['pdf'])
    if uploaded_pdf is not None:
        extracted = extract_text_from_pdf(uploaded_pdf)
        cleaned = clean_text(extracted)
        vectorized = vectorizer.transform([cleaned]).toarray()
        prediction = model.predict(vectorized)[0]
        score = max(model.predict_proba(vectorized)[0]) * 100  # Resume score as top class probability
        st.success(f"🧠 Predicted Resume Category: **{prediction}**")
        st.info(f"📊 Resume Score: **{score:.2f}%**")

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
            score = max(model.predict_proba(vectorized)[0]) * 100  # JD score as top class probability
            st.success(f"🧠 Predicted JD Category: **{prediction}**")
            st.info(f"📊 JD Score: **{score:.2f}%**")
