📰 Fake News Detection App
📌 Project Overview
This project is a Fake News Detection System built using Machine Learning and Flask. The goal is to classify news articles as Fake or Real based on textual content.

The project consists of two major phases:

Training the Model 🏋️‍♂️ in Jupyter Notebook using NLP techniques.
Deploying the Model 🚀 as a Flask web application for real-time predictions.
📂 Dataset
The dataset used comes from two CSV files:

True.csv – Contains genuine news articles.
Fake.csv – Contains misleading or fake news articles.
The dataset is preprocessed and labeled as follows:

Fake News → 0
Real News → 1
🏋️‍♂️ Model Training (Jupyter Notebook)
1️⃣ Data Preprocessing
✔️ Combined True.csv and Fake.csv into a single dataset.
✔️ Removed missing values and cleaned text data.
✔️ Converted all text to lowercase.
✔️ Removed punctuations, stopwords, and performed tokenization.

2️⃣ Feature Engineering
✔️ Used TF-IDF Vectorization (TfidfVectorizer from sklearn) to convert text into numerical form.
✔️ Limited vocabulary size to remove noise and improve efficiency.

3️⃣ Model Selection & Training
The dataset was split into 80% training and 20% testing, and the following machine learning models were trained:

Logistic Regression
Naïve Bayes (MultinomialNB)
Random Forest Classifier
Support Vector Machine (SVM)
📊 Best Performing Model: (Specify your best model here, e.g., Logistic Regression)
📈 Accuracy Achieved: XX%

4️⃣ Model Saving
After evaluation, the best model and the vectorizer were saved using joblib:

python
Copy
Edit
import joblib
joblib.dump(model, 'fake_news_model.pkl')
joblib.dump(vectorizer, 'vectorizer.pkl')
🌐 Deploying with Flask
Once the model was trained, it was integrated into a Flask web application.

1️⃣ Flask App Structure
pgsql
Copy
Edit
📂 fake_news_app/
 ├── 📂 model/
 │   ├── fake_news_model.pkl
 │   ├── vectorizer.pkl
 ├── 📂 templates/
 │   ├── index.html
 ├── app.py
 ├── requirements.txt
 ├── README.md
2️⃣ Flask App Implementation
The Flask app loads the trained model and vectorizer, processes user input, and returns predictions.

User enters news text on the web UI.
The model predicts whether it's "Fake" or "Real."
The result is displayed on the page.
📜 Flask Code (app.py):

python
Copy
Edit
from flask import Flask, render_template, request
import joblib

app = Flask(__name__)

# Load the trained model and vectorizer
model = joblib.load("model/fake_news_model.pkl")
vectorizer = joblib.load("model/vectorizer.pkl")

@app.route("/", methods=["GET", "POST"])
def index():
    if request.method == "POST":
        news_text = request.form["news_text"]
        transformed_text = vectorizer.transform([news_text])
        prediction = model.predict(transformed_text)[0]
        
        result = "Fake News" if prediction == 0 else "Real News"
        return render_template("index.html", result=result, news_text=news_text)

    return render_template("index.html", result=None)

if __name__ == "__main__":
    app.run(debug=True)
🚀 Running the App Locally
To run the Flask app on your local machine:

1️⃣ Install Dependencies
Ensure you have Python installed, then install the required packages:

bash
Copy
Edit
pip install -r requirements.txt
2️⃣ Start the Flask App
bash
Copy
Edit
python app.py
It will run on http://127.0.0.1:5000/. Open your browser and test it!

📌 Future Improvements
✅ Improve model accuracy using deep learning (LSTMs, BERT, or Transformers).
✅ Deploy the app on Heroku, AWS, or Render.
✅ Add browser extensions for fake news detection on social media.
✅ Implement an API for external use.

✨ Final Thoughts
This Fake News Detection App is a practical application of Natural Language Processing (NLP) and Machine Learning to combat misinformation. It can be expanded with advanced AI models for greater accuracy.

Let me know if you need further improvements! 🚀







