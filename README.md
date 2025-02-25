The project consists of two major phases:

->Training the Model 🏋️‍♂️ in Jupyter Notebook using NLP techniques.
->Deploying the Model 🚀 as a Flask web application for real-time predictions.
📂 Dataset
True.csv – Contains genuine news articles, Real News → 1
Fake.csv – Contains misleading or fake news articles, Fake News → 0.

Model Training (Jupyter Notebook)
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

Logistic Regression - 98%
DecisionTreeClassifier - 99%
Random Forest Classifier - 98%
GradientBoostingClassifier - 99%


4️⃣ Model Saving
After evaluation, the best model and the vectorizer were saved using joblib:

{import joblib
joblib.dump(model, 'fake_news_model.pkl')
joblib.dump(vectorizer, 'vectorizer.pkl')}{}

🌐 Deploying with Flask
Once the model was trained, it was integrated into a Flask web application.

1️⃣ Flask App Structure

📂 fake_news_app/
 ├── 📂 model/
 │   ├── fake_news_model.pkl
 │   ├── vectorizer.pkl
 ├── 📂 templates/
 │   ├── index.html
 ├── app.py

2️⃣ Flask App Implementation
The Flask app loads the trained model and vectorizer, processes user input, and returns predictions.

User enters news text on the web UI.
The model predicts whether it's "Fake" or "Real."
The result is displayed on the page.
✨ Final Thoughts
This Fake News Detection App is a practical application of Natural Language Processing (NLP) and Machine Learning to combat misinformation. It can be expanded with advanced AI models for greater accuracy.








