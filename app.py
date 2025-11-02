from flask import Flask, request, jsonify
import joblib

# Load model and vectorizer
model = joblib.load('logreg_model.pkl')
vectorizer = joblib.load('tfidf_vectorizer.pkl')

# Initialize Flask app
app = Flask(__name__)

@app.route('/')
def home():
    return "Digital Heritage Classification API is running!"

@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json()
    text = data.get('text', '')

    if not text:
        return jsonify({'error': 'No text provided'}), 400

    # Transform and predict
    X = vectorizer.transform([text])
    pred = int(model.predict(X)[0])

    mapping = {-1: "Delete", 0: "Review", 1: "Preserve"}
    return jsonify({'prediction': mapping[pred]})

if __name__ == '__main__':
    app.run(debug=True)
