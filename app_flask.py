from flask import Flask, request, jsonify
import joblib
import pandas as pd

app = Flask(__name__)

# Load the saved model
model = joblib.load('rf_model.joblib')

@app.route('/')
def home():
    return "<h1>🍷 Wine Classifier API</h1><p>Use POST /predict to make predictions.</p>"

@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json()
    input_df = pd.DataFrame([data])
    prediction = model.predict(input_df)
    return jsonify({'predicted_class': int(prediction[0])})

if __name__ == '__main__':
    app.run(debug=True)
