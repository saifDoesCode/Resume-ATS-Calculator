# 📄 ATS Score Checker – Resume Matcher App

A simple, open-source Streamlit app that helps users evaluate how well their resume matches a specific job role or job description. Whether you're applying for a tech role or a business position, this tool helps you optimize your CV for Applicant Tracking Systems (ATS).

![Streamlit](https://img.shields.io/badge/built%20with-Streamlit-orange)
![License](https://img.shields.io/badge/license-MIT-blue)
![Python](https://img.shields.io/badge/python-3.7%2B-green)

---

## 🚀 Features

- 📤 Upload your **PDF Resume**
- 🧠 Choose from hundreds of job roles across **20+ industries**
- ✏️ Enter a **custom job title** if your role isn't listed
- 📝 Optionally paste a **job description**
- ✅ View ATS **match score** using:
  - **Keyword-based matching** for known roles
  - **BERT-based semantic similarity** for job descriptions
- 🔎 Matched keywords are highlighted
- ⚙️ Powered by `sentence-transformers` and `PyPDF2`

---

## 🧰 Built With

- [Streamlit](https://streamlit.io/) – UI framework
- [PyPDF2](https://pypi.org/project/PyPDF2/) – PDF parsing
- [Sentence Transformers](https://www.sbert.net/) – BERT embeddings
- Python 3.7+

---

## 📦 Installation

Clone the repository and install dependencies:

```bash
git clone https://github.com/your-username/ats-score-checker.git
cd ats-score-checker
pip install -r requirements.txt
