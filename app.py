import streamlit as st
import joblib
import re
import PyPDF2
from sklearn.metrics.pairwise import cosine_similarity

# Load model, vectorizer, and centroids
model = joblib.load('ats_nb_model.pkl')
vectorizer = joblib.load('ats_vectorizer.pkl')
centroids = joblib.load('category_centroids.pkl')  # precomputed average vectors

# === Helper Functions ===

def clean_text(text):
    text = re.sub(r'<[^>]+>', ' ', text)
    text = re.sub(r'[^a-zA-Z]', ' ', text)
    text = text.lower().split()
    return ' '.join(text)

def extract_text_from_pdf(pdf_file):
    text = ""
    reader = PyPDF2.PdfReader(pdf_file)
    for page in reader.pages:
        page_text = page.extract_text()
        if page_text:
            text += page_text
    return text

def score_label(score):
    if score >= 80:
        return "🟢 Excellent match"
    elif score >= 60:
        return "🟡 Good match"
    else:
        return "🔴 Needs improvement"

def get_prediction_and_score(text):
    cleaned = clean_text(text)
    vectorized = vectorizer.transform([cleaned]).toarray()
    prediction = model.predict(vectorized)[0]
    centroid_vec = centroids[prediction].reshape(1, -1)
    similarity_score = cosine_similarity(vectorized, centroid_vec)[0][0] * 100
    return prediction, similarity_score  # return both prediction and score


# === Streamlit UI ===

st.title("📄 ATS Resume Analyzer")
st.markdown("Upload a resume or paste a job description to find the **predicted category** and a **score showing how well it fits that category**.")

tab1, tab2 = st.tabs(["📤 Upload Resume", "✍️ Paste Resume Text"])

# === 📤 Resume Upload Tab ===
with tab1:
    uploaded_pdf = st.file_uploader("Upload your resume PDF file", type=['pdf'])
    if uploaded_pdf is not None:
        extracted = extract_text_from_pdf(uploaded_pdf)
        prediction, resume_score = get_prediction_and_score(extracted)

        st.success(f"🧠 Predicted Resume Category: **{prediction}**")
        st.info(f"📊 Resume Similarity Score: **{resume_score:.2f}%** ({score_label(resume_score)})")
        st.progress(int(resume_score))  # Show the score as a progress bar

# === ✍️ Paste Resume Text Tab ===
with tab2:
    jd_text = st.text_area("Paste your resume content here:")
    if st.button("Analyze Resume Text"):
        if jd_text.strip() == "":
            st.warning("Please enter your resume text.")
        else:
            prediction, resume_score = get_prediction_and_score(jd_text)
            st.success(f"🧠 Predicted Resume Category: **{prediction}**")
            st.info(f"📊 Resume Similarity Score: **{resume_score:.2f}%** ({score_label(resume_score)})")
            st.progress(int(resume_score))  # Show the score as a progress bar
