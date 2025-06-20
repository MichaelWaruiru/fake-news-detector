import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.utils import shuffle
import joblib
import re

# Load dataset
df = pd.read_csv("kenyan_news.csv")

# Drop NAs, duplicates, and short articles
df.dropna(subset=["text", "label"], inplace=True)
df.drop_duplicates(subset=["text"], inplace=True)
df = df[df["text"].str.len() > 300]  # Filter too-short articles

# Text normalization
def clean_text(text):
    text = text.lower()
    text = re.sub(r'<.*?>', ' ', text)  # HTML tags
    text = re.sub(r'[^a-zA-Z\s]', ' ', text)  # Remove non-alpha
    text = re.sub(r'\s+', ' ', text).strip()
    return text

df["text"] = df["text"].apply(clean_text)

# Shuffle dataset
df = shuffle(df, random_state=42)

# Train stratified data
X = df["text"]
y = df["label"]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# Build pipeline
pipeline = Pipeline([
    ("tfidf", TfidfVectorizer(stop_words="english", max_features=15000, ngram_range=(1, 2), min_df=3)),
    ("clf", LogisticRegression(max_iter=2000, class_weight="balanced", C=0.8))
])

# Train
pipeline.fit(X_train, y_train)

# Evaluate
y_pred = pipeline.predict(X_test)
print(classification_report(y_test, y_pred))
print(confusion_matrix(y_test, y_pred))

# Save model
joblib.dump(pipeline, "model.pkl")
print("Model saved to model.pkl")