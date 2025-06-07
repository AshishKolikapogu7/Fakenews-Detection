# Fake News Detection

This project is a Flask web application that uses a machine learning model to detect fake news. The model is trained on a dataset of real and fake news articles and can classify new articles as either "Real News" or "Fake News".

## Project Structure

```
Fakenews-Detection/
├── static/
│   └── style.css
├── templates/
│   └── index.html
├── app.py
├── fake_news_model.pkl
├── vectorizer.pkl
└── README.md
```

## How to Run

1.  **Clone the repository:**
    ```bash
    git clone https://github.com/AshishKolikapogu7/Fakenews-Detection.git
    cd Fakenews-Detection
    ```

2.  **Install the dependencies:**
    It is recommended to use a virtual environment.
    ```bash
    pip install -r requirements.txt
    ```

3.  **Run the Flask application:**
    ```bash
    python app.py
    ```

4.  **Open your browser** and navigate to `http://127.0.0.1:5000/`.

## Dependencies

The required Python libraries are listed in the `requirements.txt` file.
- Flask
- scikit-learn
- joblib

## Model Training

The model was trained using a Jupyter Notebook with the following steps:

### 1. Data Preprocessing
- Combined `True.csv` and `Fake.csv` into a single dataset.
- Cleaned the text data by removing missing values, converting to lowercase, and removing punctuation and stopwords.

### 2. Feature Engineering
- Used TF-IDF Vectorization (`TfidfVectorizer` from scikit-learn) to convert the text data into numerical features.

### 3. Model Selection & Training
The following models were trained and evaluated:
- **Logistic Regression:** 98% accuracy
- **Decision Tree Classifier:** 99% accuracy
- **Random Forest Classifier:** 98% accuracy
- **Gradient Boosting Classifier:** 99% accuracy

The best performing model and the TF-IDF vectorizer were saved using `joblib`.

## Deployment
The trained model is deployed as a Flask web application. The application provides a simple user interface to enter a news article, and it will predict whether the news is real or fake.
