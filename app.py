import streamlit as st
import pandas as pd
import re
from sklearn.metrics.pairwise import cosine_similarity
from PyPDF2 import PdfReader
import joblib

# ---------- Resume Cleaning Function ----------
def clean_text(text):
    text = re.sub(r"http\S+|www\S+|https\S+", '', str(text))
    text = re.sub(r'\@w+|\#', '', text)
    text = re.sub(r'[^A-Za-z\s]', '', text)
    text = text.lower()
    return text

# ---------- Load CSV Dataset ----------
@st.cache_data
def load_data():
    df = pd.read_csv("resume_data.csv")
    df['Cleaned_Resume'] = df['Resume'].apply(clean_text)
    return df

# ---------- Extract Text from PDF ----------
def extract_text_from_pdf(file):
    reader = PdfReader(file)
    text = ""
    for page in reader.pages:
        text += page.extract_text()
    return text

# ---------- Calculate Similarity Score ----------
def calculate_score(user_resume, category_resumes, vectorizer):
    all_resumes = category_resumes + [user_resume]
    tfidf_matrix = vectorizer.fit_transform(all_resumes)
    similarity_matrix = cosine_similarity(tfidf_matrix[-1], tfidf_matrix[:-1])
    score = similarity_matrix.max() * 100
    return round(score, 2)

# ---------- Load Everything ----------
df = load_data()

# Load trained model and extract vectorizer
pipeline = joblib.load("resume_classifier_model.joblib")
vectorizer = pipeline.named_steps['tfidf']

# ---------- Streamlit Frontend ----------
st.set_page_config(page_title="Resume Scorer", page_icon="📄")
st.title("📄 Resume Scorer by Job Category")
st.markdown("Upload your resume or paste the text below, select a job category, and receive a match score + improvement suggestions!")

# Choose Input Method
st.subheader("1. Choose How You Want to Submit Your Resume")
upload_col, text_col = st.columns(2)

with upload_col:
    uploaded_file = st.file_uploader("📤 Upload Resume (.txt or .pdf)", type=["txt", "pdf"])

with text_col:
    typed_resume = st.text_area("✍️ Or Paste Resume Text Here (optional)", height=200)

# Select Category
category = st.selectbox("📌 Select Job Category", sorted(df["Category"].unique()))

# Process Resume
resume_text = ""

if uploaded_file:
    if uploaded_file.type == "application/pdf":
        resume_text = extract_text_from_pdf(uploaded_file)
    else:
        resume_text = uploaded_file.read().decode("utf-8")
elif typed_resume:
    resume_text = typed_resume

if resume_text and category:
    cleaned_resume = clean_text(resume_text)
    category_df = df[df["Category"] == category]
    category_resumes = category_df["Cleaned_Resume"].tolist()

    score = calculate_score(cleaned_resume, category_resumes, vectorizer)

    st.success(f"🎯 Resume Match Score: **{score}%**")

    # # Keyword-based feedback
    # top_keywords = vectorizer.get_feature_names_out()
    # missing_keywords = [kw for kw in top_keywords if kw in " ".join(category_resumes) and kw not in cleaned_resume]

# Get TF-IDF scores for the selected category
category_tfidf = vectorizer.fit_transform(category_resumes)
feature_array = vectorizer.get_feature_names_out()
avg_tfidf_scores = category_tfidf.mean(axis=0).A1  # average across resumes

# Create a sorted list of top keywords by average importance
top_keywords = [
    (feature_array[i], avg_tfidf_scores[i])
    for i in range(len(feature_array))
]
top_keywords = sorted(top_keywords, key=lambda x: x[1], reverse=True)

# Take top N (e.g., 20) most important keywords
top_keywords_only = [kw for kw, _ in top_keywords[:20]]

# Find which of those are missing from the resume
missing_keywords = [kw for kw in top_keywords_only if kw not in cleaned_resume]


if score < 70:
        st.warning("📝 Feedback:")
        st.markdown("- Try adding more relevant skills, tools, projects, or domain keywords.")
        if missing_keywords:
            st.markdown("- Missing keywords (examples):")
            st.write(", ".join(missing_keywords[:10]))
else:
        st.success("✅ Great! Your resume is well-aligned with this job category.")

# Footer
st.markdown("---")
