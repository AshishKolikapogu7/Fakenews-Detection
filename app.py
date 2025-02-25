from flask import Flask, render_template, request
import joblib

app = Flask(__name__)

# Load the trained model and vectorizer
model = joblib.load(r"C:\\Users\\ADMIN\\Desktop\\fake_news_app\\model\\fake_news_model.pkl")
vectorizer = joblib.load("C:\\Users\ADMIN\\Desktop\\fake_news_app\\model\\vectorizer.pkl")

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
