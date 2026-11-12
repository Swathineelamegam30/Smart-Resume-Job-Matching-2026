
import streamlit as st
import pandas as pd
import re
import os
from pypdf import PdfReader
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# PAGE SETTINGS
st.set_page_config(
    page_title="Smart Resume Job Matching",
    page_icon="📄",
    layout="wide"
)

# CUSTOM CSS
st.markdown("""
<style>
.stApp {
    background: linear-gradient(135deg, #0f172a, #1e1b4b);
    color: white;
}

/* HEADINGS */
h1, h2, h3 {
    color: #c4b5fd !important;
}

/* NORMAL TEXT */
p, label, span, div {
    color: inherit;
}

/* TEXT INPUT */
textarea {
    color: white !important;
    background-color: #25244f !important;
    -webkit-text-fill-color: white !important;
    caret-color: white !important;
}

textarea::placeholder {
    color: #cccccc !important;
    -webkit-text-fill-color: #cccccc !important;
}

.stTextArea label {
    color: white !important;
}

/* METRIC CARDS */
[data-testid="stMetric"] {
    background-color: #25244f !important;
    padding: 20px !important;
    border-radius: 12px !important;
    border: 1px solid #6366f1 !important;
}

/* METRIC LABEL */
[data-testid="stMetricLabel"] {
    color: #ffffff !important;
}

[data-testid="stMetricLabel"] * {
    color: #ffffff !important;
}

/* METRIC VALUE */
[data-testid="stMetricValue"] {
    color: #ffffff !important;
    font-size: 32px !important;
    font-weight: bold !important;
}

[data-testid="stMetricValue"] * {
    color: #ffffff !important;
}

/* BUTTON */
.stButton > button {
    background: linear-gradient(90deg, #7c3aed, #2563eb);
    color: white !important;
    border-radius: 10px;
    border: none;
    padding: 10px 25px;
    font-weight: bold;
}

.stButton > button:hover {
    background: #6d28d9;
    color: white !important;
}

/* FILE UPLOADER */
[data-testid="stFileUploader"] {
    background-color: #25244f;
    padding: 15px;
    border-radius: 12px;
}

/* EXPANDER */
[data-testid="stExpander"] {
    background-color: #25244f;
    border-radius: 10px;
}

/* PROGRESS BAR */
.stProgress > div > div > div {
    background-color: #a78bfa !important;
}
</style>
""", unsafe_allow_html=True)


# SKILLS LIST
skills_list = [
    "python", "java", "machine learning",
    "deep learning", "artificial intelligence",
    "data analysis", "pandas", "numpy",
    "excel", "power bi", "sql",
    "html", "css", "javascript",
    "react.js", "git", "github",
    "scikit-learn"
]


# EXTRACT SKILLS
def extract_skills(text):
    text = text.lower()
    found = []

    for skill in skills_list:
        pattern = r"(?<!\w)" + re.escape(skill) + r"(?!\w)"
        if re.search(pattern, text):
            found.append(skill)

    return found


# EXTRACT RESUME TEXT
def extract_resume_text(pdf_file):
    reader = PdfReader(pdf_file)
    text = ""

    for page in reader.pages:
        page_text = page.extract_text()

        if page_text:
            text += page_text + "\n"

    return text


# TEXT SIMILARITY
def calculate_similarity(resume_text, job_description):

    if not resume_text.strip() or not job_description.strip():
        return 0

    vectorizer = TfidfVectorizer()

    vectors = vectorizer.fit_transform(
        [resume_text, job_description]
    )

    score = cosine_similarity(
        vectors[0], vectors[1]
    )[0][0]

    return round(score * 100, 2)


# DATASET JOB RECOMMENDATIONS
def recommend_dataset_jobs(resume_skills):

    file_path = "Dataset/job_postings_cleaned.csv"

    if not os.path.exists(file_path):
        return None

    jobs = pd.read_csv(file_path)

    if "skills" not in jobs.columns:
        return None

    results = []

    for _, job in jobs.iterrows():

        job_skills = set(
            re.split(r"[,;|]", str(job["skills"]).lower())
        )

        job_skills = {
            skill.strip() for skill in job_skills
            if skill.strip()
        }

        if not job_skills:
            continue

        matched = set(resume_skills) & job_skills

        score = (
            len(matched) / len(job_skills)
        ) * 100

        results.append({
            "Title": job.get("title", "Not available"),
            "Company": job.get("company", "Not available"),
            "Role": job.get("role", "Not available"),
            "City": job.get("city", "Not available"),
            "Remote": job.get("remote", "Not available"),
            "Matching Skills": round(score, 2),
            "Matched Skills": ", ".join(sorted(matched))
        })

    results.sort(
        key=lambda x: x["Matching Skills"],
        reverse=True
    )

    return results[:5]


# MAIN APPLICATION
st.title("📄 Smart Resume–Job Matching System")

st.write(
    "Upload your resume and find suitable job matches!"
)

# RESUME UPLOAD
st.header("📤 Upload Your Resume")

uploaded_file = st.file_uploader(
    "Upload PDF Resume",
    type=["pdf"]
)

# JOB DESCRIPTION
st.header("💼 Enter Job Details")

job_description = st.text_area(
    "Paste Job Description Here",
    height=150,
    placeholder="Enter job requirements and skills..."
)


# ANALYZE BUTTON
if st.button("🔍 Analyze Resume"):

    if uploaded_file is None:

        st.warning("Please upload your resume first.")

    else:

        try:

            resume_text = extract_resume_text(uploaded_file)

            if not resume_text.strip():
                st.error("No readable text found in this PDF.")
                st.stop()

            resume_skills = extract_skills(resume_text)

            job_skills = extract_skills(job_description)

            matched_skills = [
                skill for skill in job_skills
                if skill in resume_skills
            ]

            missing_skills = [
                skill for skill in job_skills
                if skill not in resume_skills
            ]

            if job_skills:

                skill_score = (
                    len(matched_skills) / len(job_skills)
                ) * 100

            else:

                skill_score = 0

            text_score = calculate_similarity(
                resume_text,
                job_description
            )

            st.success("Resume Analysis Completed!")

            # MATCHING RESULTS
            st.header("📊 Matching Results")

            col1, col2 = st.columns(2)

            col1.metric(
                "Skill Matching",
                f"{skill_score:.2f}%"
            )

            col2.metric(
                "Text Similarity",
                f"{text_score:.2f}%"
            )

            # MATCHED SKILLS
            st.subheader("✅ Matched Skills")

            if matched_skills:

                for skill in matched_skills:
                    st.write("✅", skill)

            else:

                st.write("No matched skills found.")

            # MISSING SKILLS
            st.subheader("⚠️ Missing Skills")

            if missing_skills:

                for skill in missing_skills:
                    st.write("⚠️", skill)

            else:

                st.write("No missing skills!")

            # RESUME SKILLS
            st.subheader("📝 Skills Found in Resume")

            st.write(", ".join(resume_skills))

            # JOB ROLE RECOMMENDATIONS
            st.header("🎯 Recommended Job Roles")

            job_roles = {

                "Machine Learning Engineer": [
                    "python", "machine learning",
                    "deep learning", "numpy"
                ],

                "Data Analyst": [
                    "python", "pandas", "numpy",
                    "data analysis", "excel", "power bi"
                ],

                "Python Developer": [
                    "python", "git", "github"
                ],

                "AI Engineer": [
                    "python", "artificial intelligence",
                    "machine learning", "numpy"
                ],

                "Frontend Developer": [
                    "html", "css", "javascript", "react.js"
                ]
            }

            role_results = []

            for role, required_skills in job_roles.items():

                matched = set(resume_skills) & set(required_skills)

                score = (
                    len(matched) / len(required_skills)
                ) * 100

                role_results.append((role, score))

            role_results.sort(
                key=lambda x: x[1],
                reverse=True
            )

            for role, score in role_results[:3]:

                st.write(f"**{role}**")

                st.progress(int(score))

                st.write(
                    f"Matching Skills: {score:.2f}%"
                )

            # DATASET JOB RECOMMENDATIONS
            st.header("💼 Dataset-Based Job Recommendations")

            dataset_jobs = recommend_dataset_jobs(resume_skills)

            if dataset_jobs is None:

                st.warning(
                    "Job dataset not found. Check Dataset folder."
                )

            elif not dataset_jobs:

                st.info("No matching job postings found.")

            else:

                for index, job in enumerate(dataset_jobs, 1):

                    with st.expander(
                        f"{index}. {job['Title']} - {job['Company']}"
                    ):

                        st.write("Role:", job["Role"])

                        st.write("Company:", job["Company"])

                        st.write("City:", job["City"])

                        st.write("Remote:", job["Remote"])

                        st.write(
                            "Matching Skills:",
                            f"{job['Matching Skills']}%"
                        )

                        st.write(
                            "Matched Skills:",
                            job["Matched Skills"]
                        )

            # EXTRACTED TEXT
            with st.expander("View Extracted Resume Text"):

                st.text(resume_text)

            st.info(
                "Note: Matching percentages are based on "
                "skill and text similarity, not hiring probability."
            )

        except Exception as e:

            st.error(f"Error: {e}")
