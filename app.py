from flask import Flask, render_template, request, jsonify
import joblib
import os
from newspaper import Article
import validators

app = Flask(__name__)

# Load model
MODEL_PATH = os.path.join('model', 'model.pkl')
pipeline = joblib.load(MODEL_PATH)

# List of key Kenyan politicians for detection
POLITICIANS = [
  # Presidents
  "Jomo Kenyatta", "Daniel arap Moi", "Mwai Kibaki", "Uhuru Kenyatta", "William Ruto",
  
  # Deputy Presidents / Vice Presidents
  "Jaramogi Oginga Odinga", "Joseph Murumbi", "Daniel arap Moi", "Mwai Kibaki", "George Saitoti", "Kithure Kindiki",
  "Musalia Mudavadi", "Michael Wamalwa", "Moody Awori", "Kalonzo Musyoka", "William Ruto", "Rigathi Gachagua",

  # Prime Ministers
  "Raila Odinga",

  # Cabinet Secretaries (Recent/Prominent)
  "Moses Kuria", "Musalia Mudavadi", "Aden Duale", "Alfred Mutua", "Kindiki Kithure", "Cleophas Malala",
  "David Ndii", "Ezekiel Machogu", "Susan Nakhumicha", "Kipchumba Murkomen", "Ababu Namwamba", "Simon Chelugui",
  "Njuguna Ndung'u", "Rebecca Miano", "Florence Bore", "Soipan Tuya", "Eliud Owalo", "Peninah Malonza",
  "Mithika Linturi", "Davis Chirchir", "Zachariah Njeru", "Salim Mvurya", "Aisha Jumwa", "Alice Wahome",
  "Susan Wafula", "Roselinda Soipan Tuya", "Mercy Wanjau", "Julius Bitok", "Monica Juma",

  # Former Cabinet Secretaries/Ministers
  "Rashid Echesa", "Fred Matiang'i", "Henry Rotich", "Rachel Omamo", "Najib Balala", "James Macharia",
  "George Magoha", "Mutahi Kagwe", "Joe Mucheru", "Ukur Yatani", "Peter Munya", "Charles Keter",
  "Cicily Kariuki", "Simon Kachapin", "Phyllis Kandie", "Adan Mohamed", "Dan Kazungu", "Raychelle Omamo",

  # Governors (2013-present, some key names)
  "Johnson Sakaja", "Mike Sonko", "Evans Kidero", "Anne Waiguru", "Anyang' Nyong'o", "Alfred Mutua",
  "Charity Ngilu", "Ali Hassan Joho", "James Orengo", "Wycliffe Oparanya", "Jackson Mandago", "Sospeter Ojaamong",
  "Kivutha Kibwana", "Lee Kinyanjui", "Mutahi Kahiga", "Mwangi wa Iria", "Amason Kingi", "Granton Samboja",
  "Patrick Khaemba", "Josphat Nanok", "Cornel Rasanga", "Paul Chepkwony", "Joyce Laboso", "Samuel Tunai",

  # Senators (Prominent)
  "James Orengo", "Susan Kihika", "Samson Cherargei", "Mutula Kilonzo Jr.", "Moses Wetangula", "Johnson Sakaja",

  # MPs and Other Political Figures
  "Raila Odinga", "Martha Karua", "Moses Wetangula", "Kalembe Ndile", "George Aladwa", "Millie Odhiambo",
  "John Mbadi", "Kimani Ichung'wah", "Ndindi Nyoro", "Junet Mohamed", "Babu Owino", "Charles Njagua",
  "Oscar Sudi", "Mohamed Ali", "Didmus Barasa", "Aisha Jumwa", "Catherine Waruguru", "Gladys Wanga",

  # Party Leaders
  "Moses Wetangula", "Musalia Mudavadi", "Gideon Moi", "Isaac Ruto", "Martha Karua",

  # Notable Political Figures
  "Tom Mboya", "Pio Gama Pinto", "Ronald Ngala", "Masinde Muliro", "Paul Ngei", "Martin Shikuku",
  "Charles Njonjo", "Nicholas Biwott", "Simeon Nyachae", "John Michuki", "William ole Ntimama",
  "Josphat Karanja", "Robert Ouko", "Mulu Mutisya",

  # Others (recent prominent)
  "Murkomen", "Gladys Boss Shollei", "Beatrice Elachi", "Sabina Chege", "Mutula Kilonzo Jr.",
  "Ken Lusaka", "Isaac Mwaura", "Peter Kenneth", "Eugene Wamalwa", "Raphael Tuju", "David Murathe"
  # Add more as needed
]

POLITICIANS = list(set(POLITICIANS))

def extract_text_from_url(url):
  article = Article(url)
  article.download()
  article.parse()
  return article.text

def detect_politician(text):
  return list({p for p in POLITICIANS if p.lower() in text.lower()})


@app.route("/", methods=["GET", "POST"])
def index():
  prediction = None
  proba = None
  user_text = ""
  mentions = []
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
          mentions = detect_politician(user_text)
        except Exception as e:
          prediction = "Error processing the input. Please try again."
  return render_template("index.html", prediction=prediction, proba=proba, user_text=user_text, mentions=mentions)

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
    mentions = detect_politician(text)
    return jsonify({"prediction": prediction, "confidence": proba, "politicians": mentions})
  except Exception as e:
    return jsonify({"error", str(e)}), 500

if __name__ == "__main__":
    app.run(debug=True, host='0.0.0.0')