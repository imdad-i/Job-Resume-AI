import re
import joblib
from typing import List, Tuple
from pathlib import Path
import os

# Try to load NLTK stopwords; if unavailable try to download them. As a last
# resort fall back to a small built-in stopword set so imports never fail in
# environments without nltk data (e.g., fresh Streamlit Cloud containers).
try:
    from nltk.corpus import stopwords
    try:
        STOPWORDS = set(stopwords.words('english'))
    except LookupError:
        # Attempt to download stopwords data
        try:
            import nltk
            nltk.download('stopwords')
            STOPWORDS = set(stopwords.words('english'))
        except Exception:
            # Fallback small stopword set
            STOPWORDS = {
                'the', 'and', 'is', 'in', 'to', 'of', 'a', 'for', 'on', 'with',
                'as', 'by', 'an', 'be', 'this', 'that', 'it', 'from', 'or'
            }
except Exception:
    # If nltk is not installed or import fails, provide a minimal stopword set
    STOPWORDS = {
        'the', 'and', 'is', 'in', 'to', 'of', 'a', 'for', 'on', 'with',
        'as', 'by', 'an', 'be', 'this', 'that', 'it', 'from', 'or'
    }

# Default model path resolved relative to the repository root (two levels up from this file)
MODEL_PATH = Path(__file__).resolve().parent.parent.joinpath("models", "resume_clf.joblib")

def clean_text(text: str) -> str:
    if not isinstance(text, str): return ""
    text = text.lower()
    text = re.sub(r'\S+@\S+', ' ', text)
    text = re.sub(r'http\S+|www\.\S+', ' ', text)
    text = re.sub(r'\d+', ' ', text)
    text = re.sub(r'[\r\n]+', ' ', text)
    text = re.sub(r'[^a-z\s+#.]', ' ', text)
    tokens = [w for w in text.split() if w not in STOPWORDS]
    return " ".join(tokens)

class ResumeClassifier:
    def __init__(self, model_path: str | Path = MODEL_PATH):
        # Accept a Path or string. For relative paths we try a few locations so the
        # app works whether Streamlit changed the current working directory or not.
        mp = Path(model_path)

        candidates: list[Path] = []
        # If absolute path provided, try it directly
        if mp.is_absolute():
            candidates.append(mp)
        else:
            # 1) Resolve relative to repository root (parent.parent of this file)
            repo_root = Path(__file__).resolve().parent.parent
            candidates.append((repo_root / mp).resolve())
            # 2) Resolve relative to current working directory
            try:
                candidates.append((Path.cwd() / mp).resolve())
            except Exception:
                pass
            # 3) Common fallback: look inside repo's models folder using the filename
            candidates.append((repo_root / 'models' / mp.name).resolve())

        # Remove duplicates while preserving order
        seen = set()
        unique_candidates = []
        for p in candidates:
            if p not in seen:
                seen.add(p)
                unique_candidates.append(p)

        found = None
        for p in unique_candidates:
            if p.exists():
                found = p
                break

        if found is None:
            tried = "\n".join(str(p) for p in unique_candidates)
            raise FileNotFoundError(
                f"Model file not found. Tried the following paths:\n{tried}\n\n"
                f"Place `resume_clf.joblib` in one of these locations or pass an absolute path."
            )

        self.pipeline = joblib.load(found)
        self.classes_ = list(self.pipeline.named_steps['clf'].classes_) \
            if hasattr(self.pipeline.named_steps['clf'], 'classes_') else None

    def predict(self, text: str) -> Tuple[str, float, List[Tuple[str, float]]]:
        cleaned = clean_text(text)
        pred = self.pipeline.predict([cleaned])[0]
        # calibrated proba
        proba = self.pipeline.predict_proba([cleaned])[0]
        idx = self.classes_.index(pred)
        topk_idx = proba.argsort()[::-1][:5]
        topk = [(self.classes_[i], float(proba[i])) for i in topk_idx]
        return pred, float(proba[idx]), topk
