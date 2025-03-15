import gradio as gr
import joblib
import string
import nltk
from nltk.corpus import stopwords
import re

nltk.download('stopwords')
nltk.download('punkt')
nltk.download('omw-1.4')

stop_words = set(stopwords.words('english'))

# preprocessing of text
def preprocess_text(text):
    text = text.lower()
    text = re.sub(r'\d+', '', text)
    text = text.translate(str.maketrans('', '', string.punctuation))
    words = text.split()
    words = [word for word in words if word not in stop_words]
    return ' '.join(words)


#load the model and vectorizer
model = joblib.load("sentiment_model.pkl")
vectorizer = joblib.load("vectorizer.pkl")

#predict function
def predict_sentiment(review):
    review_tfidf = vectorizer.transform([preprocess_text(review)])
    prediction = model.predict(review_tfidf)[0]
    return "Positive 😊" if prediction == 1 else "Negative 😠"

#frontend interface
iface = gr.Interface(
    fn=predict_sentiment,
    inputs=gr.Textbox(lines=3, placeholder="Enter a movie review..."),
    outputs="text",
    title="IMDB Sentiment Analysis",
    description="Enter a movie review, and the model will classify it as Positive or Negative."
)

iface.launch()