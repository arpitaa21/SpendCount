"""Train and persist the reproducible description categoriser."""
from app.ml.categorisation import train_model, TRAINING
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
import numpy as np
texts=[x for xs in TRAINING.values() for x in xs]
labels=[k for k,xs in TRAINING.items() for _ in xs]
n_splits=min(5,min(np.unique(labels,return_counts=True)[1]))
evaluation=Pipeline([("tfidf",TfidfVectorizer(ngram_range=(1,2),strip_accents="unicode")),("classifier",LogisticRegression(max_iter=500,class_weight="balanced"))])
scores=cross_val_score(evaluation,texts,labels,cv=StratifiedKFold(n_splits=n_splits,shuffle=True,random_state=42))
model=train_model()
print(f"Saved categoriser to the ignored local models directory; {len(TRAINING)} labels, {sum(map(len, TRAINING.values()))} synthetic phrase examples.")
print(f"{n_splits}-fold stratified phrase-corpus accuracy: {scores.mean():.3f} ± {scores.std():.3f}")
print("This small synthetic phrase evaluation is only a reproducibility check, not real-world accuracy evidence.")
