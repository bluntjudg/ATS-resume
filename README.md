https://ats-resume-asp.streamlit.app/

link to my app



demo resume data :- Skills * Programming Languages: Python (pandas, numpy, scipy, scikit-learn, matplotlib), Sql, Java, JavaScript/JQuery. * Machine learning: Regression, SVM, NaÃ¯ve Bayes, KNN, Random Forest, Decision Trees, Boosting techniques, Cluster Analysis, Word Embedding, Sentiment Analysis, Natural Language processing, Dimensionality reduction, Topic Modelling (LDA, NMF), PCA & Neural Nets. * Database Visualizations: Mysql, SqlServer, Cassandra, Hbase, ElasticSearch D3.js, DC.js, Plotly, kibana, matplotlib, ggplot, Tableau. * Others: Regular Expression, HTML, CSS, Angular 6, Logstash, Kafka, Python Flask, Git, Docker, computer vision - Open CV and understanding of Deep learning.Education Details Data Science Assurance Associate Data Science Assurance Associate - Ernst & Young LLP Skill Details JAVASCRIPT- Exprience - 24 months jQuery- Exprience - 24 months Python- Exprience - 24 monthsCompany Details company - Ernst & Young LLP description - Fraud Investigations and Dispute Services Assurance TECHNOLOGY ASSISTED REVIEW TAR (Technology Assisted Review) assists in accelerating the review process and run analytics and generate reports. * Core member of a team helped in developing automated review platform tool from scratch for assisting E discovery domain, this tool implements predictive coding and topic modelling by automating reviews, resulting in reduced labor costs and time spent during the lawyers review. * Understand the end to end flow of the solution, doing research and development for classification models, predictive analysis and mining of the information present in text data. Worked on analyzing the outputs and precision monitoring for the entire tool. * TAR assists in predictive coding, topic modelling from the evidence by following EY standards. Developed the classifier models in order to identify "red flags" and fraud-related issues. Tools & Technologies: Python, scikit-learn, tfidf, word2vec, doc2vec, cosine similarity, NaÃ¯ve Bayes, LDA, NMF for topic modelling, Vader and text blob for sentiment analysis. Matplot lib, Tableau dashboard for reporting. MULTIPLE DATA SCIENCE AND ANALYTIC PROJECTS (USA CLIENTS) TEXT ANALYTICS - MOTOR VEHICLE CUSTOMER REVIEW DATA * Received customer feedback survey data for past one year. Performed sentiment (Positive, Negative & Neutral) and time series analysis on customer comments across all 4 categories. * Created heat map of terms by survey category based on frequency of words * Extracted Positive and Negative words across all the Survey categories and plotted Word cloud. * Created customized tableau dashboards for effective reporting and visualizations. CHATBOT * Developed a user friendly chatbot for one of our Products which handle simple questions about hours of operation, reservation options and so on. * This chat bot serves entire product related questions. Giving overview of tool via QA platform and also give recommendation responses so that user question to build chain of relevant answer. * This too has intelligence to build the pipeline of questions as per user requirement and asks the relevant /recommended questions. Tools & Technologies: Python, Natural language processing, NLTK, spacy, topic modelling, Sentiment analysis, Word Embedding, scikit-learn, JavaScript/JQuery, SqlServer INFORMATION GOVERNANCE Organizations to make informed decisions about all of the information they store. The integrated Information Governance portfolio synthesizes intelligence across unstructured data sources and facilitates action to ensure organizations are best positioned to counter information risk. * Scan data from multiple sources of formats and parse different file formats, extract Meta data information, push results for indexing elastic search and created customized, interactive dashboards using kibana. * Preforming ROT Analysis on the data which give information of data which helps identify content that is either Redundant, Outdated, or Trivial. * Preforming full-text search analysis on elastic search with predefined methods which can tag as (PII) personally identifiable information (social security numbers, addresses, names, etc.) which frequently targeted during cyber-attacks. Tools & Technologies: Python, Flask, Elastic Search, Kibana FRAUD ANALYTIC PLATFORM Fraud Analytics and investigative platform to review all red flag cases. â¢ FAP is a Fraud Analytics and investigative platform with inbuilt case manager and suite of Analytics for various ERP systems. * It can be used by clients to interrogate their Accounting systems for identifying the anomalies which can be indicators of fraud by running advanced analytics Tools & Technologies: HTML, JavaScript, SqlServer, JQuery, CSS, Bootstrap, Node.js, D3.js, DC.js


Output :- 

![image](https://github.com/user-attachments/assets/ff79d209-2764-4386-af80-11c3ea03006a)



# 📝 ATS Resume Analyzer

## 1. Core Project Information
- **Project Name:** ATS Resume Analyzer  
- **Brief Description:**  
  A Streamlit-based application that evaluates resumes for ATS-friendliness using NLP and machine learning models.  
- **Problem Statement / Use Case:**  
  Many resumes are filtered out by Applicant Tracking Systems (ATS) due to formatting or missing keywords. This project helps job seekers check if their resumes are ATS-compatible.  
- **Target Audience:** Job seekers, career coaches, HR professionals  
- **Project Timeline & Status:** Prototype with functional demo (ongoing improvements possible)  

---

## 2. Tech Stack
- **Frontend:** Streamlit (Python-based UI)  
- **Backend:** Python, scikit-learn (Naive Bayes, vectorization), pandas, numpy  
- **Development Tools:** Jupyter Notebook, Git/GitHub, VS Code  
- **Deployment:** Streamlit Cloud ([ats-resume-asp.streamlit.app](https://ats-resume-asp.streamlit.app))  
- **Third-party APIs:** None (self-contained ML)  

---

## 3. Architecture & Design
### System Architecture
- **Input:** Resume text (via `.txt` or `.pdf`)  
- **Processing:**  
  - Text preprocessing (vectorization)  
  - Classification via Naive Bayes model  
  - Resume category clustering (centroids)  
- **Output:** ATS-friendliness score & report  

*(Diagram can be added here)*  

### Data & API
- **Database Schema:** CSV files (`resume_data.csv`, `resume_analysis_results.csv`); no relational DB  
- **API Endpoints:** None (direct Streamlit app)  
- **Data Flow:** Resume → Preprocessing → Vectorizer → ML Model → ATS Score → User Report  
- **Security:** Relies on Streamlit sandbox (no explicit measures)  

### Code Organization
```
├── app.py                # Main Streamlit app
├── ATSresume.ipynb       # Model training & experiments
├── models/
│   ├── ats_nb_model.pkl
│   ├── ats_vectorizer.pkl
├── data/
│   ├── resume_data.csv
│   ├── resume_analysis_results.csv
├── requirements.txt
└── README.md
```

- **Architecture Style:** Monolithic (single Streamlit app)  
- **Design Patterns:** Standard ML pipeline  
- **Algorithms:** Naive Bayes (classification), clustering for categories  

---

## 4. Features & Functionality
### Core Features
- Upload resumes (`.txt` / `.pdf`)  
- Analyze resumes for ATS-friendliness  
- Extract keywords & classify resume categories  
- Provide ATS compatibility score  

### Unique Selling Points
- Lightweight, fast evaluation  
- Offline-capable (except for deployment)  
- Tailored to ATS parsing rules  

---

## 5. Technical Complexity
- **Challenges:** Handling multiple file formats, preprocessing varied resume styles, generalizing ML models across industries  
- **Decisions:**  
  - Chose **Naive Bayes** for simplicity & interpretability  
  - Used **Streamlit** for rapid prototyping & deployment  
  - Opted for **CSV storage** instead of a database for minimal setup  

---

## 6. Development Process
- **Version Control:** GitHub repo (no branching strategy, solo dev)  
- **Collaboration:** Individual project  
- **Testing:** No formal tests; validation via Jupyter Notebook & manual checks  

---

## 7. Performance & Metrics
- **Performance:**  
  - Response time: < 1 second per resume  
  - Lightweight, memory-efficient model  
- **Monitoring:**  
  - No built-in monitoring  
  - Streamlit Cloud usage stats available  

---

## 8. Deployment & Operations
- **Hosting:** Streamlit Cloud  
- **Environment:** Single production environment  
- **CI/CD:** Streamlit auto-deployment from GitHub  
- **Maintenance:** Manual repo updates, retrainable models  

---

## 9. Learning & Growth
- Learned deploying ML apps with Streamlit  
- Gained exposure to resume parsing challenges  
- Hands-on with scikit-learn pipelines  

---

## 10. Future Enhancements
- Add REST API for external integration  
- Support for `.docx` format  
- Upgrade to transformer models (e.g., BERT)  
- ATS-specific formatting checks (fonts, layouts)  
- Multi-language support  

---

## 11. Documentation & Communication
- **Docs:**  
  - `requirements.txt` for dependencies  
  - Basic README with setup instructions  
  - Minimal inline code comments  
- **Presentation:**  
  - [Live Demo](https://ats-resume-asp.streamlit.app)  
  - Example dataset included  

---

## 12. Interview Preparation Notes
### Common Questions
- Walk me through the ATS resume pipeline  
- Why choose Naive Bayes?  
- How would you scale this for thousands of resumes?  
- What limitations exist (file formats, dataset bias)?  

### Code Deep Dive
- Vectorization & feature extraction  
- Model training workflow (`ATSresume.ipynb`)  
- Streamlit integration in `app.py`  

### Quantifiable Results
- Resume analysis in < 1 second  
- Trained on CSV dataset with multiple categories  


