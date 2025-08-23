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

Option 2 — Use Git LFS (good for personal projects where you want the model versioned with code)

1. Install Git LFS locally (follow platform instructions). On Windows with Chocolatey or scoop, or from https://git-lfs.github.com/.

2. Track the model file and commit it:

```powershell
git lfs install
git lfs track "models/*.joblib"
git add .gitattributes
git add models/resume_clf.joblib
git commit -m "Add trained model via Git LFS"
```

3. Create a repository on GitHub (via the website), then add the remote and push:

```powershell
git remote add origin https://github.com/<your-username>/<repo-name>.git
git branch -M main
git push -u origin main
```

4. On https://streamlit.io/cloud, click "New app", connect your GitHub repo, pick branch `main` and file `src/app_streamlit.py`.

Notes about Git LFS

- Git LFS stores the pointer file in git and the actual binary in LFS storage. Your GitHub account has LFS quotas; for personal projects this is usually fine but check usage.

Option 3 — External hosting (alternative)

- Host the model in S3/GCS or a release asset, set `MODEL_URL` in Streamlit Cloud secrets, and update `src/infer.py` to download the model at first run. This keeps the git repo small and is more flexible for production.

Notes

- `src/infer.py` resolves the model path relative to the repository root; `src/app_streamlit.py` also computes a repo-relative path, so running from the repo root works best.
- If you see warnings about scikit-learn versions when loading the model, consider matching the training scikit-learn version in your deployment environment.
