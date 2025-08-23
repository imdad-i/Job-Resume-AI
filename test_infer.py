from src.infer import ResumeClassifier

clf = ResumeClassifier()

sample = "Experienced data scientist with Python, scikit-learn, NLP, and ML pipelines. Deployed models with Flask."
print(clf.predict(sample))
