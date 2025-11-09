from flask import Flask, request, jsonify, render_template
import joblib

# Load model and vectorizer
model = joblib.load('logreg_model.pkl')
vectorizer = joblib.load('tfidf_vectorizer.pkl')

# Initialize Flask app
app = Flask(__name__)

# Label mapping
mapping = {-1: "Delete", 0: "Review", 1: "Preserve"}

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    text = request.form.get('text')

    if not text:
        return render_template('index.html', prediction="Please enter some text.")

    # Transform and predict
    X = vectorizer.transform([text])
    pred = int(model.predict(X)[0])
    result = mapping[pred]

    return render_template('index.html', prediction=result, user_input=text)

# API endpoint (still available)
@app.route('/api/predict', methods=['POST'])
def api_predict():
    data = request.get_json()
    text = data.get('text', '')
    if not text:
        return jsonify({'error': 'No text provided'}), 400
    X = vectorizer.transform([text])
    pred = int(model.predict(X)[0])
    return jsonify({'prediction': mapping[pred]})

if __name__ == '__main__':
    app.run(debug=True)
