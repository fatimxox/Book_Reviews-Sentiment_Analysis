from flask import Flask, request, render_template, jsonify
import pickle
import re
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import numpy as np

app = Flask(__name__)

# Download NLTK data
nltk.download('stopwords')
nltk.download('punkt_tab')
stop_words = set(stopwords.words('english'))

# Load the model and vectorizer
with open('sentiment_model_oversample.pkl', 'rb') as f:
    model = pickle.load(f)
with open('vectorizer_oversample.pkl', 'rb') as f:
    vectorizer = pickle.load(f)

def clean_text(text):
    if not isinstance(text, str):
        return ''
    text = text.lower()
    text = re.sub(r'[^\w\s]', '', text)  # Remove punctuation
    text = re.sub(r'\d+', '', text)  # Remove numbers
    tokens = word_tokenize(text)
    tokens = [word for word in tokens if word not in stop_words and len(word) > 2]
    # Fallback for single words or filtered inputs
    if not tokens and len(text.strip()) > 0:
        tokens = [text.strip()]
    return ' '.join(tokens)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    review = request.form.get('review', '')
    if not review or len(review.strip()) < 1:
        return jsonify({'error': 'Please enter a valid review (at least one character).', 'status': 'error'}), 400
    
    # Clean and preprocess the review
    cleaned_review = clean_text(review)
    if not cleaned_review.strip():
        return jsonify({'error': 'Review is empty after cleaning. Please provide a meaningful word.', 'status': 'error'}), 400
    
    # Transform the review using the vectorizer
    review_tfidf = vectorizer.transform([cleaned_review])
    
    # Predict sentiment and get probabilities
    prediction = model.predict(review_tfidf)[0]
    probabilities = model.predict_proba(review_tfidf)[0]
    
    # Confidence is the probability of the predicted class
    confidence = max(probabilities) * 100  # Convert to percentage
    sentiment_score = probabilities[1]  # Probability of Positive class (index 1)
    
    return jsonify({
        'sentiment': prediction.capitalize(),
        'confidence': round(confidence, 0),
        'sentimentScore': sentiment_score,  # For gauge positioning (0 to 1)
        'status': 'success'
    })

if __name__ == '__main__':
    app.run(debug=True)