import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report
import joblib
# import os

# Load your dataset
df = pd.read_csv('news.csv')  # Must have 'text' and 'label' columns

# Basic preprocessing (remove NAs)
df.dropna(subset=['text', 'subject'], inplace=True)

X = df['text']
y = df['subject']

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Build pipeline: TF-IDF + Logistic Regression
pipeline = Pipeline([
    ('tfidf', TfidfVectorizer(stop_words='english', max_features=10000)),
    ('clf', LogisticRegression(max_iter=1000))
])

# Train
pipeline.fit(X_train, y_train)

# Evaluate
y_pred = pipeline.predict(X_test)
print(classification_report(y_test, y_pred))

# Save the entire pipeline
# os.makedirs('model', exist_ok=True)
joblib.dump(pipeline, 'model.pkl')
print("Model saved to model.pkl")