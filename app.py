import streamlit as st
import PyPDF2
import re
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Clean text function
def clean_text(text):
    text = re.sub(r'<[^>]+>', ' ', text)
    text = re.sub(r'[^a-zA-Z]', ' ', text)
    text = text.lower().split()
    return ' '.join(text)

# Extract PDF resume text
def extract_text_from_pdf(pdf_file):
    text = ""
    reader = PyPDF2.PdfReader(pdf_file)
    for page in reader.pages:
        page_text = page.extract_text()
        if page_text:
            text += page_text
    return text

# Matching logic
def get_match_score(resume_text, job_desc):
    cleaned_resume = clean_text(resume_text)
    cleaned_jd = clean_text(job_desc)

    vectorizer = TfidfVectorizer()
    tfidf_matrix = vectorizer.fit_transform([cleaned_resume, cleaned_jd])

    similarity = cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:2])[0][0]
    return round(similarity * 100, 2)

# Streamlit UI
st.title("🧠 ATS Resume Match Checker")
st.write("Upload your resume to see if it matches your job title or description (optional).")

# Optional job title
job_input = st.text_input("💼 Enter the Job Title or Job Description (Optional)")

# Required resume upload
uploaded_pdf = st.file_uploader("📄 Upload your Resume PDF", type=['pdf'])

if st.button("🔍 Analyze"):
    if uploaded_pdf is None:
        st.warning("Please upload your resume to proceed.")
    else:
        resume_text = extract_text_from_pdf(uploaded_pdf)
        st.success("✅ Resume uploaded and extracted successfully.")
        
        if job_input.strip() != "":
            match_score = get_match_score(resume_text, job_input)
            st.markdown(f"### 📊 Match Score: **{match_score}%**")
            if match_score >= 60:
                st.success("✅ Good Match! Your resume aligns well with the job.")
            else:
                st.error("❌ Not a strong match. Consider tailoring your resume.")
        else:
            st.info("ℹ️ No job title/description entered, so match score is not calculated.")
