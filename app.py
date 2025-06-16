from flask import Flask, render_template, request, jsonify
import joblib
import os
from newspaper import Article
import validators

app = Flask(__name__)

# Load model
MODEL_PATH = os.path.join('model', 'model.pkl')
pipeline = joblib.load(MODEL_PATH)


def extract_text_from_url(url):
  article = Article(url)
  article.download()
  article.parse()
  return article.text


@app.route("/", methods=["GET", "POST"])
def index():
    prediction = None
    proba = None
    user_text = ""
    if request.method == "POST":
        user_text = request.form.get("news")
        if user_text:
          try:
            # If it's a URL, extract article file content
            if validators.url(user_text):
              user_text = extract_text_from_url(user_text)
            pred_proba = pipeline.predict_proba([user_text])[0]
            pred = pipeline.predict([user_text])[0]
            prediction = "True/Real" if pred == 1 else "Fake/False"
            proba = round(100 * max(pred_proba), 2)
          except Exception as e:
            prediction = "Error processing the input. Please try again."
    return render_template("index.html", prediction=prediction, proba=proba, user_text=user_text)

# API Endpoint for AJAX/mobile use
@app.route("/api/predict", methods=["POST"])
def api_predict():
    data = request.get_json()
    text = data.get("text", "")
    if not text:
        return jsonify({"error": "No text provided"}), 400
    try:
      if validators.url(text):
        text = extract_text_from_url(text)
      pred_proba = pipeline.predict_proba([text])[0]
      pred = pipeline.predict([text])[0]
      prediction = "True/Real" if pred == 1 else "Fake/False"
      proba = round(100 * max(pred_proba), 2)
      return jsonify({"prediction": prediction, "confidence": proba})
    except Exception as e:
      return jsonify({"error", str(e)}), 500

if __name__ == "__main__":
    app.run(debug=True, host='0.0.0.0')