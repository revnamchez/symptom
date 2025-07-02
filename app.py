from flask import Flask, request, jsonify
import joblib

app = Flask(__name__)

# Load the model and vectorizer
try:
    model = joblib.load("log_symptom.pkl")
    vectorizer = joblib.load("vectorizer.pkl")
except FileNotFoundError:
    print("Error: Model or vectorizer file not found.")
    exit()

@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json()
    symptoms = data.get('symptoms', '')
    if symptoms.strip() == "":
        return jsonify({'error': 'Please enter some symptoms'}), 400
    
    try:
        # Transform the symptoms using the loaded vectorizer
        symptoms_tfidf = vectorizer.transform([symptoms])
        
        # Make prediction
        prediction = model.predict(symptoms_tfidf)[0]
        probabilities = model.predict_proba(symptoms_tfidf)[0]
        sickness_probs = list(zip(model.classes_, probabilities))
        sickness_probs.sort(key=lambda x: x[1], reverse=True)
        
        # Create response
        response = {
            'primary_prediction': prediction,
            'confidence': max(probabilities) * 100,
            'top_3_predictions': [
                {'sickness': sickness, 'probability': prob * 100} 
                for sickness, prob in sickness_probs[:3]
            ]
        }
        return jsonify(response)
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', debug=True)
