import streamlit as st
import PyPDF2
import re
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Category Templates (you can extend this as needed)
category_templates = {
    "Data Science": """
    Python, Machine Learning, Data Analysis, Pandas, NumPy, Scikit-learn,
    Deep Learning, AI, Statistics, Data Visualization, TensorFlow, Keras, NLP
    """,
    "Web Development": """
    HTML, CSS, JavaScript, React, Angular, Node.js, Django, Flask, PHP, MySQL,
    MongoDB, Web APIs, Bootstrap, Frontend, Backend, Full Stack
    """,
    "Human Resources": """
    Recruitment, Employee Relations, Payroll, Performance Management,
    Talent Acquisition, HR Policies, Onboarding, Training, HRMS
    """,
    "Android Development": """
    Java, Kotlin, Android Studio, XML, Firebase, SQLite, UI/UX, Play Store,
    Jetpack, MVVM, Mobile App Development
    """
}

# Clean text
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
def get_match_score(text1, text2):
    cleaned_1 = clean_text(text1)
    cleaned_2 = clean_text(text2)

    vectorizer = TfidfVectorizer()
    tfidf_matrix = vectorizer.fit_transform([cleaned_1, cleaned_2])

    similarity = cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:2])[0][0]
    return round(similarity * 100, 2)

# UI
st.title("🧠 ATS Resume Analyzer & Category Matcher")

st.markdown("### 📤 Upload Resume")
uploaded_pdf = st.file_uploader("Upload your Resume (PDF)", type=['pdf'])

st.markdown("### ✍️ Paste Job Description (Optional)")
job_input = st.text_area("Paste Job Description Here (Optional)")

st.markdown("### 📂 Select Resume Category to Match Against")
category_choice = st.selectbox("Choose the job field", ["Select", *category_templates.keys()])

if st.button("🔍 Analyze Resume"):

    if uploaded_pdf is None:
        st.warning("⚠️ Please upload your resume to continue.")
    else:
        resume_text = extract_text_from_pdf(uploaded_pdf)

        # If Job Description is entered
        if job_input.strip() != "":
            match_score = get_match_score(resume_text, job_input)
            st.markdown(f"### 💼 Match Score with Job Description: **{match_score}%**")
            if match_score >= 60:
                st.success("✅ Resume aligns well with the job description.")
            else:
                st.error("❌ Resume does not match the job description well.")

        # Category-based match
        if category_choice != "Select":
            template_text = category_templates[category_choice]
            category_score = get_match_score(resume_text, template_text)
            st.markdown(f"### 📊 Match Score with *{category_choice}* Category: **{category_score}%**")
            if category_score >= 60:
                st.success(f"✅ Strong match with the **{category_choice}** category.")
            else:
                st.warning(f"⚠️ Resume could be improved for the **{category_choice}** field.")
