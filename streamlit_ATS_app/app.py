import streamlit as st
import PyPDF2
from sentence_transformers import SentenceTransformer, util
from data.role_keywords import ROLE_KEYWORDS
from data.job_roles_by_industry import ALL_JOB_ROLES_BY_INDUSTRY


st.set_page_config(page_title="ATS Score Checker", layout="centered")

@st.cache_resource
def load_model():
    return SentenceTransformer("all-MiniLM-L6-v2")

model = load_model()

def extract_text_from_pdf(file):
    reader = PyPDF2.PdfReader(file)
    return " ".join(page.extract_text() or "" for page in reader.pages)

def calculate_similarity_bert(cv_text, job_description):
    embeddings = model.encode([cv_text, job_description], convert_to_tensor=True)
    score = util.pytorch_cos_sim(embeddings[0], embeddings[1]).item()
    return round(score * 100, 2)

def calculate_keyword_score(cv_text, keywords):
    cv_lower = cv_text.lower()
    matched = [kw for kw in keywords if kw.lower() in cv_lower]
    score = (len(matched) / len(keywords)) * 100 if keywords else 0
    return round(score, 2), matched

st.title("📄 ATS Score Checker")
st.markdown("Upload your CV and check how well it matches a job role or description.")

cv_file = st.file_uploader("📤 Upload your PDF Resume", type=["pdf"])
job_description = st.text_area("📝 Optional: Paste Job Description", height=150)

industry = st.selectbox("🏭 Choose an industry (No need to choose IF you have added job description)", ["None"] + list(ALL_JOB_ROLES_BY_INDUSTRY.keys()))
role = None
custom_role = None

if industry != "None":
    role_options = ALL_JOB_ROLES_BY_INDUSTRY[industry] + ["Other (not listed)"]
    selected = st.selectbox("💼 Select a job role", role_options)

    if selected == "Other (not listed)":
        custom_role = st.text_input("✏️ Enter your job title manually")
        role = custom_role.strip() if custom_role else None
    else:
        role = selected

if st.button("🔍 Calculate ATS Score"):
    if not cv_file:
        st.warning("⚠️ Please upload your resume.")
    else:
        with st.spinner("Analyzing your resume..."):
            cv_text = extract_text_from_pdf(cv_file)

            if job_description.strip():
                score = calculate_similarity_bert(cv_text, job_description)
                st.success(f"✅ BERT-based ATS Score: **{score}%** match with job description")
                st.progress(int(score))
            elif role:
                if role in ROLE_KEYWORDS:
                    score, matched = calculate_keyword_score(cv_text, ROLE_KEYWORDS[role])
                    st.success(f"✅ Keyword Match Score for '{role}': **{score}%**")
                    st.markdown(f"**Matched Keywords:** {', '.join(matched) if matched else 'None'}")
                    st.progress(int(score))
                else:
                    st.warning(f"⚠️ No predefined keywords for '{role}'. Please paste a job description above for better results.")
            else:
                st.warning("⚠️ Please provide a job description or select a role.")
