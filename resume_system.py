import streamlit as st
import PyPDF2
from sentence_transformers import SentenceTransformer, util

st.set_page_config(page_title="ATS Score Checker", layout="centered")

ALL_JOB_ROLES_BY_INDUSTRY = {
    "Technology & IT": ["Software Engineer", "Frontend Developer", "Backend Developer", "Full Stack Developer",
                        "DevOps Engineer", "Cloud Engineer", "Data Scientist", "Data Analyst", "AI Engineer",
                        "Machine Learning Engineer", "Cybersecurity Analyst", "IT Support Specialist",
                        "Network Administrator", "UI Designer", "UX Designer", "Web Developer", "QA Engineer"],
    "Business & Management": ["Business Analyst", "Project Manager", "Product Manager", "Operations Manager",
                              "Account Manager", "Risk Analyst", "Management Consultant", "Strategy Manager"],
    "Marketing & Sales": ["Digital Marketer", "SEO Specialist", "Content Strategist", "Social Media Manager",
                          "Email Marketing Specialist", "Marketing Manager", "Sales Executive",
                          "Account Executive", "Business Development Manager", "Brand Manager", "Market Research Analyst"],
    "Creative & Design": ["Graphic Designer", "Motion Designer", "Art Director", "Animator", "Illustrator",
                          "UI Designer", "Video Editor", "Photographer", "Copywriter", "Content Writer"],
    "Finance & Accounting": ["Financial Analyst", "Accountant", "Auditor", "Controller", "Investment Analyst",
                             "Treasury Analyst", "Finance Manager", "Budget Analyst", "Tax Advisor"],
    "Healthcare": ["Registered Nurse", "Physician", "Pharmacist", "Dentist", "Medical Technologist",
                   "Radiologist", "Physical Therapist", "Medical Assistant", "Clinical Research Associate",
                   "Occupational Therapist"],
    "Engineering": ["Mechanical Engineer", "Civil Engineer", "Electrical Engineer", "Chemical Engineer",
                    "Structural Engineer", "Environmental Engineer", "Industrial Engineer", "Biomedical Engineer",
                    "Petroleum Engineer"],
    "Education": ["Teacher", "Professor", "Academic Advisor", "Curriculum Developer", "Instructional Designer",
                  "Librarian", "Education Coordinator", "School Administrator", "Special Education Teacher"],
    "Legal": ["Lawyer", "Paralegal", "Legal Assistant", "Compliance Officer", "Contract Manager",
              "Legal Analyst", "Corporate Counsel"],
    "Logistics & Supply Chain": ["Supply Chain Analyst", "Logistics Manager", "Procurement Specialist",
                                 "Warehouse Manager", "Inventory Analyst", "Operations Coordinator", "Fleet Manager"],
    "Human Resources": ["HR Manager", "Recruiter", "Talent Acquisition Specialist", "Compensation Analyst",
                        "Training Coordinator", "HR Generalist", "Employee Relations Specialist"],
    "Manufacturing & Trades": ["Manufacturing Engineer", "Machine Operator", "Maintenance Technician",
                               "Electrician", "Welder", "CNC Programmer", "Assembly Line Worker", "Quality Inspector"],
    "Hospitality & Tourism": ["Hotel Manager", "Front Desk Agent", "Concierge", "Event Coordinator",
                              "Tour Guide", "Chef", "Travel Agent", "Restaurant Manager"],
    "Customer Service": ["Customer Service Representative", "Call Center Agent", "Customer Success Manager",
                         "Technical Support Specialist", "Help Desk Technician"],
    "Government & Public Sector": ["Policy Analyst", "Public Administrator", "Urban Planner", "Civil Servant",
                                   "Customs Officer", "Environmental Inspector"],
    "Nonprofit & NGOs": ["Program Coordinator", "Grant Writer", "Fundraising Manager", "Advocacy Officer",
                         "NGO Project Manager", "Volunteer Coordinator"],
    "Real Estate & Construction": ["Real Estate Agent", "Property Manager", "Construction Manager", "Architect",
                                   "Site Supervisor", "Estimator", "Surveyor"],
    "Science & Research": ["Research Scientist", "Lab Technician", "Biologist", "Chemist", "Physicist",
                           "Environmental Scientist", "Data Researcher"],
    "Retail & eCommerce": ["Store Manager", "Retail Sales Associate", "Ecommerce Manager", "Merchandise Planner",
                           "Category Manager", "Cashier", "Inventory Specialist"],
    "Agriculture & Environment": ["Agricultural Scientist", "Farm Manager", "Environmental Consultant",
                                  "Conservationist", "Soil Scientist", "Forester", "Wildlife Biologist"],
    "Energy & Utilities": ["Energy Analyst", "Renewable Energy Engineer", "Utility Manager",
                           "Electric Power Technician", "Oil & Gas Technician", "Solar Installer"]
}

ROLE_KEYWORDS = {
    "Data Analyst": ["sql", "excel", "python", "tableau", "statistics", "data", "visualization", "business intelligence"],
    "Software Engineer": ["python", "java", "c++", "git", "api", "docker", "linux", "algorithms", "oop"],
    "Digital Marketer": ["seo", "google ads", "social media", "email marketing", "analytics", "content marketing"],
    # Add more roles & keywords as needed
}

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
