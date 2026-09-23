# 📄 ATS Resume Analyzer

[Live Demo](https://ats-resume-asp.streamlit.app/)

![app output](https://github.com/user-attachments/assets/ff79d209-2764-4386-af80-11c3ea03006a)

## What it actually does

This app is branded as an "ATS-friendliness" checker, but it's important to be precise about what it evaluates. It does not parse resume formatting, check for tables/columns/fonts, or extract keywords the way a real Applicant Tracking System does. What it actually does:

1. Predicts which job category a resume belongs to (e.g. Java Developer, Data Science, HR) out of 25 possible categories, using a Naive Bayes classifier trained on TF-IDF text features.
2. Scores how closely the resume matches the *average* resume in its predicted category (cosine similarity to that category's centroid vector).

So it's a resume-to-job-category matcher with a confidence score, not a literal ATS parsing simulator. The README description has been corrected to reflect this.

## Real, verified results

Retrained the exact pipeline (TfidfVectorizer, max_features=3000, MultinomialNB, 90/10 split, random_state=42) from `ATSresume.ipynb` to confirm the numbers:

- **Dataset:** 962 resumes across 25 job categories (Kaggle resume dataset), heavily imbalanced — Java Developer is the largest class (84 resumes), Advocate the smallest (20).
- **Test accuracy: 92.8%** (90/97 correct) on 25-way classification. Majority-class baseline (always predicting "Java Developer") would only get 7.2%.
- **Per-class breakdown:** several categories (Operations Manager, PMO, Python Developer, Sales, Web Designing) hit F1 = 1.00 on the test split. Advocate scored F1 = 0.0 — all 3 Advocate test resumes were misclassified as Java Developer, the largest class in the dataset.
- **Similarity score behaves as intended:** correctly classified resumes score 61.2% average similarity to their category centroid (range 31–85%), while misclassified resumes score much lower on average (28.1%). So the score does carry real signal about classification confidence.
- **Worth knowing if you use the app:** the app's own "🟢 Excellent match" label requires a score of 80%+. Even among correctly classified resumes in testing, only about 6% crossed that bar — most correct matches land in the 60–80% "Good match" range. Don't read a "Good" instead of "Excellent" label as the model being unsure; it's the normal range for a correct prediction.

## Tech Stack
- **Frontend:** Streamlit
- **Backend:** Python, scikit-learn (MultinomialNB, TF-IDF vectorization), pandas, numpy
- **Development:** Jupyter Notebook for training (`ATSresume.ipynb`), Git/GitHub
- **Deployment:** Streamlit Cloud

## Pipeline
```
Resume (.txt / .pdf) → clean_resume() → TF-IDF vectorize
    → MultinomialNB predicts category
    → cosine similarity vs. that category's centroid
    → category label + similarity score
```

## Code Organization
```
├── app.py                    # Streamlit app
├── ATSresume.ipynb           # Model training & evaluation
├── ats_nb_model.pkl
├── ats_vectorizer.pkl
├── category_centroids.pkl
├── resume_data.csv           # training data (962 resumes, 25 categories)
├── requirements.txt
└── README.md
```

## Limitations (honestly)
- Not a real ATS simulator — no formatting/layout/keyword-density checks a real ATS would do.
- Smallest classes (Advocate, Civil Engineer, SAP Developer) are the hardest to classify reliably; Advocate had zero correct predictions on this test split.
- Trained on a fixed 2019-era public resume dataset, so results won't necessarily generalize to resume styles or job categories outside that set.
- No formal test suite; validated via notebook + manual checks only.

## Future Enhancements
- Add real ATS-style formatting/keyword checks alongside the category classifier
- Support `.docx` uploads
- Try transformer-based embeddings instead of TF-IDF for the similarity score
