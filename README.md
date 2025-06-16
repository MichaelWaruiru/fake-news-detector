# 📰 Real-Time Fake News Detector

A real-time web-based machine learning app that detects whether a news article is **Real** or **Fake** using Logistic Regression and Natural Language Processing (TF-IDF). Built with Flask, scikit-learn, and Bootstrap.

---

## 🚀 Features

- **Binary Fake News Detection:** Classifies news as True/Real or Fake/False.
- **Real-Time Analysis:** Instant results via web UI or API.
- **Mobile-Ready Design:** Responsive Bootstrap interface works on phones, tablets, and desktops.
- **REST API:** Easily integrate prediction into any app or workflow.
- **Easy Model Training:** Train your own model on any news dataset.
- **Lightweight & Fast:** Optimized for quick deployment and low resource usage.

---

## 📂 Project Structure

```
fake-news-detector/
├── app.py                  # Flask app (web UI + API)
├── model/
│   ├── train_model.py      # Model training script
│   ├── model.pkl           # Saved ML pipeline (after training)
├── templates/
│   └── index.html          # Web UI template (Bootstrap)
├── static/
│   └── style.css           # (Optional) Custom styles
├── requirements.txt        # Python dependencies
└── README.md               # This file
```

---

## 🧠 Model Training

### Dataset

* Download `Fake.csv` and `True.csv` from:
  [Fake and Real News Dataset](https://www.kaggle.com/datasets/clmentbisaillon/fake-and-real-news-dataset)

### Prepare Combined Dataset

```bash
cd models
python prepare_data.py  # Generates news.csv with binary labels
```

### Train the Model

```bash
python train_model.py  # Trains and saves model to model/model.pkl
```

---

## 🚀 Running the Web App

### 1. **Clone the Repository**

```bash
git clone https://github.com/<MichaelWaruiru>/fake-news-detector.git
cd fake-news-detector
```

### 2. **Install Dependencies**

```bash
pip install -r requirements.txt
```

### 3. **Start the Flask Server**

```bash
cd ..
python app.py
```

Visit: `http://localhost:5000`

You can paste:

* A raw news **article or headline**
* A **URL** of a news article (it will be fetched & parsed)

---

## Features

* Input either **raw text** or a **URL**
* Extracts article content using `newspaper3k`
* Uses TF-IDF vectorization
* Logistic Regression for binary classification (Fake = 0, Real = 1)
* Displays prediction result and confidence level
* Styled with Bootstrap 5 for mobile responsiveness

---

## 🧪 API Endpoint

### POST `/api/predict`

#### Request Body:

```json
{
  "text": "Your news article or URL here"
}
```

#### Response:

```json
{
  "prediction": "True/Real",
  "confidence": 97.52
}
```

---

## ✅ Requirements

```
Flask
scikit-learn
pandas
joblib
newspaper3k
validators
lxml
lxml_html_clean
```

---

## ⚠️ Notes

- **Do NOT upload large datasets to GitHub.** Host them externally and provide a download script or link.
- Predictions are only as good as the data/model used. For critical applications, always validate with multiple sources.
- For best results, retrain your model regularly with recent, high-quality labeled news.

Just in case you can bypass this in your git terminal do the following:
    # Install Git LFS
    git lfs install

    # Track CSV files
    git lfs track "*.csv"

    # Add and commit as usual
    git add .gitattributes
    git add yourfile.csv
    git commit -m "Add large CSV with LFS"
    git push

Note: Free LFS storage is limited; for big datasets, you may hit quota.

---

## ✨ Credits

* Dataset: [Fake and Real News Dataset](https://www.kaggle.com/datasets/clmentbisaillon/fake-and-real-news-dataset)
* ML/NLP: scikit-learn, TfidfVectorizer
* UI: Bootstrap 5

---

## 📌 License

MIT License.
Feel free to use, modify, and share!

---

## 👤 Author

[Michael Waruiru]  
[https://github.com/MichaelWaruiru]

---

## 🙌 Contributions

Pull requests and suggestions are welcome!  
Please open an issue or PR for bug fixes, features, or improvements.

---