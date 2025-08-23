# AI Resume Classifier

This repository contains a Streamlit app that classifies resumes into categories using a sklearn pipeline.

Quick start (local)

1. Create a virtual environment and activate it (Windows PowerShell):

```powershell
python -m venv .venv; .venv\Scripts\Activate
pip install -r requirements.txt
```

2. (Optional) Train the model if `models/resume_clf.joblib` is missing:

```powershell
python src/train.py
```

3. Run the Streamlit app from the repository root:

```powershell
streamlit run src/app_streamlit.py
```

Deploy to Streamlit Cloud (recommended for sharing)

1. Initialize git, commit, and push to a new GitHub repository (see below). Do NOT commit `models/resume_clf.joblib` if it's large; instead use one of these options:
   - Upload the model to a cloud storage (S3, Azure Blob) and modify `src/infer.py` to fetch it at startup.
   - Use Git LFS for large files.

2. On https://streamlit.io/cloud, create a new app and point it to your GitHub repo and the `src/app_streamlit.py` file.

Notes

- `src/infer.py` resolves the model path relative to the repository root; `src/app_streamlit.py` also computes a repo-relative path, so running from the repo root works best.
- If you see warnings about scikit-learn versions when loading the model, consider matching the training scikit-learn version in your deployment environment.
