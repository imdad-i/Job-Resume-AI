# AI Resume Classifier

Live demo: https://job-resume-ai-l8lw8x6ezm49kbwmfejzym.streamlit.app/

## Short description

AI Resume Classifier is a small, production-oriented web demo that classifies uploaded resumes into job categories and computes similarity against a provided job description. It demonstrates the full ML lifecycle: data cleaning, model training, packaging, and cloud deployment with Streamlit.

## Why this project

- Problem solved: Fast, automated resume categorization and similarity scoring to help recruiters and applicants match resumes to roles.
- My contributions: end-to-end implementation — data preprocessing, feature engineering (TF-IDF), model training and calibration, inference pipeline, Streamlit UI, and cloud deployment.
- Impact: a shareable demo for interviews and portfolio pages that showcases ML engineering and productization skills.

## Features

- Upload PDF/DOCX/TXT resumes and get a predicted category with confidence.
- See top-5 predicted categories with probabilities.
- Paste an optional job description to compute TF-IDF cosine similarity.
- Robust deployment: repo-relative model loading, NLTK-safe fallback for stopwords, and Streamlit-ready packaging.

## Tech stack

- Python 3.11+
- Streamlit (UI)
- scikit-learn (training & pipeline)
- joblib (model serialization)
- pandas, numpy (data handling)
- nltk, spacy (text cleaning)
- pdfminer.six, python-docx (resume parsing)

## Repo layout

- `src/` – app and helper code (`app_streamlit.py`, `infer.py`, `train.py`, `file_readers.py`)
- `models/` – trained model artifact (`resume_clf.joblib`)
- `data/` – raw and processed data used for training
- `requirements.txt` – dependency list

## Run locally (quick)

1. Create & activate a virtual environment (Windows PowerShell):

   ```powershell
   python -m venv .venv; .venv\Scripts\Activate
   pip install -r requirements.txt
   ```

2. (Optional) Train the model if `models/resume_clf.joblib` is missing:

   ```powershell
   python src/train.py
   ```

3. Run the app from the repository root:

   ```powershell
   streamlit run src/app_streamlit.py
   ```

## Deploy (Streamlit Cloud)

1. Push your repo to GitHub. If you want to version the trained model file inside the repo use Git LFS:

   ```powershell
   # install Git LFS if needed, then
   git lfs install
   git lfs track "models/*.joblib"
   git add .gitattributes
   git add models/resume_clf.joblib
   git commit -m "Add model via Git LFS"
   git push origin main
   ```

2. On https://streamlit.io/cloud create a new app and connect your GitHub repo. Pick branch `main` and file `src/app_streamlit.py`.


## Operational & privacy notes

- Privacy: resumes contain PII. The app processes uploads in-memory; consider adding a clear privacy notice and do not persist uploads unless you have explicit consent.
- Cold starts: loading the model on first request may cause a short delay — we added caching and a repo-relative loader to reduce failure modes.

## Roadmap / improvements

- Add authentication for private demos
- Add CI for tests and model reproducibility
- Improve model (transformer embeddings) for better matching accuracy
- Add analytics and monitoring for production usage

## Contact & attribution

MD IMDADUL ISLAM — imdad.eshan.101@gmail.com

