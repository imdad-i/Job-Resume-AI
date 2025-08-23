import streamlit as st
# Ensure NLTK stopwords are available at runtime (Streamlit Cloud won't have them by default)
# Do this before importing `infer` so infer's import-time STOPWORDS resolution can use nltk.
import nltk
from pathlib import Path
from file_readers import read_any
from sklearn.metrics.pairwise import cosine_similarity

stopwords_source = "builtin"
try:
    nltk.data.find('corpora/stopwords')
    stopwords_source = 'nltk'
except Exception:
    try:
        nltk.download('stopwords')
        stopwords_source = 'nltk'
    except Exception:
        stopwords_source = 'builtin'

from infer import ResumeClassifier, clean_text

st.set_page_config(page_title="AI Resume Classifier", layout="centered")

# Load classifier
@st.cache_resource
def load_classifier():
    # Compute a repo-relative model path so Streamlit runs find the model regardless of CWD
    model_path = Path(__file__).resolve().parent.parent.joinpath('models', 'resume_clf.joblib')
    return ResumeClassifier(model_path=str(model_path))

clf = load_classifier()

st.title("📄 AI Resume Classifier")

# Show runtime info in the sidebar for debugging
st.sidebar.markdown("### Runtime info")
st.sidebar.write(f"Stopwords source: **{stopwords_source}**")
try:
    st.sidebar.write(f"Model path: `{clf.pipeline}`")
except Exception:
    # pipeline may not be loaded in some error states; show path instead
    model_path = Path(__file__).resolve().parent.parent.joinpath('models', 'resume_clf.joblib')
    st.sidebar.write(f"Model path (computed): `{model_path}`")

# File upload
uploaded_file = st.file_uploader("Upload your resume (PDF, DOCX, or TXT)", type=["pdf", "docx", "txt"])

# Optional job description
job_desc = st.text_area("Paste the job description here (optional):")

# Button to trigger prediction
if uploaded_file:
    if st.button("Predict Resume Category"):
        with st.spinner("Processing resume..."):
            # Extract and clean text
            text = read_any(uploaded_file)
            cleaned = clean_text(text)

            # Prediction
            pred, conf, topk = clf.predict(cleaned)

            # Display main prediction
            st.subheader("🔮 Predicted Resume Category")
            st.write(f"**Category:** {pred}")
            st.write(f"**Confidence:** {conf:.2%}")

            # Display top 5 classes nicely, skipping the top one (already shown)
            st.subheader("🏆 Top 5 Categories")
            for label, prob in topk:
                st.write(f"- {label}: {prob:.2%}")

            # Similarity with job description
            if job_desc.strip():
                job_clean = clean_text(job_desc)
                vecs = clf.pipeline.named_steps["tfidf"].transform([cleaned, job_clean])
                sim = cosine_similarity(vecs[0], vecs[1])[0][0]
                st.subheader("📊 Similarity with Job Description")
                st.write(f"{sim:.2%}")
