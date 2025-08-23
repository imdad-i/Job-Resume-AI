import os
import re
import joblib
import pandas as pd

from sklearn.model_selection import train_test_split, StratifiedKFold, GridSearchCV
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.calibration import CalibratedClassifierCV
from sklearn.svm import LinearSVC
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report, confusion_matrix

from nltk.corpus import stopwords
import nltk

try:
    _ = stopwords.words("english")
except LookupError:
    nltk.download("stopwords")

DATA_PATH = "data/raw/UpdatedResumeDataSet.csv" 
MODEL_PATH = "models/resume_clf.joblib"
REPORT_PATH = "models/classification_report.txt"

STOPWORDS = set(stopwords.words("english"))

def clean_text(text: str) -> str:
    """Light resume-aware cleaning that keeps tokens like 'c#' and '.net'."""
    if not isinstance(text, str):
        return ""
    text = text.lower()
    text = re.sub(r"\S+@\S+", " ", text)             
    text = re.sub(r"http\S+|www\.\S+", " ", text)    
    text = re.sub(r"\d+", " ", text)                
    text = re.sub(r"[\r\n]+", " ", text)              
    text = re.sub(r"[^a-z\s+#.]", " ", text)         
    tokens = [w for w in text.split() if w not in STOPWORDS]
    return " ".join(tokens)

def main():
    # ---- Load & sanity checks
    if not os.path.exists(DATA_PATH):
        raise FileNotFoundError(f"Could not find dataset at {DATA_PATH}")
    df = pd.read_csv(DATA_PATH)
    df = df.rename(columns={c: c.strip() for c in df.columns})
    if "Category" not in df.columns or "Resume" not in df.columns:
        raise ValueError(f"CSV must have 'Category' and 'Resume' columns; found {df.columns.tolist()}")

    # ---- Preprocess text
    df["text"] = df["Resume"].astype(str).apply(clean_text)
    X = df["text"].values
    y = df["Category"].astype(str).values

    # ---- Split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # ---- Pipeline 
    pipeline = Pipeline([
        ("tfidf", TfidfVectorizer(
            analyzer="word",
            ngram_range=(1, 2),
            min_df=2,
            max_features=100_000,
            sublinear_tf=True
        )),
        ("clf", CalibratedClassifierCV(
            estimator=LinearSVC(class_weight="balanced", random_state=42),
            cv=3,            
            method="sigmoid",     
            n_jobs=-1             
        ))
    ])

    # ---- Hyperparameters to search
    param_grid = {
        "tfidf__ngram_range": [(1, 1), (1, 2)],
        "tfidf__min_df": [2, 3],
        "clf__estimator__C": [0.5, 1.0, 2.0],
    }

    cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)
    grid = GridSearchCV(
        estimator=pipeline,
        param_grid=param_grid,
        scoring="f1_macro",
        cv=cv,
        n_jobs=-1,
        verbose=1,
        refit=True,       
        return_train_score=False
    )

    # ---- Train
    grid.fit(X_train, y_train)

    print("Best params:", grid.best_params_)
    print("Best CV score (f1_macro):", grid.best_score_)
    best = grid.best_estimator_

    # ---- Evaluate
    y_pred = best.predict(X_test)
    print("\nClassification Report:\n")
    rep = classification_report(y_test, y_pred, digits=3)
    print(rep)
    print("\nConfusion Matrix:\n")
    print(confusion_matrix(y_test, y_pred))

    # ---- Persist
    os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)
    joblib.dump(best, MODEL_PATH)

    with open(REPORT_PATH, "w", encoding="utf-8") as f:
        f.write("Best params:\n")
        f.write(str(grid.best_params_) + "\n\n")
        f.write("Best CV score (f1_macro): " + str(grid.best_score_) + "\n\n")
        f.write("Classification report:\n")
        f.write(rep + "\n")

    print(f"\nSaved model to {MODEL_PATH}")
    print(f"Saved report to {REPORT_PATH}")

if __name__ == "__main__":
    main()
