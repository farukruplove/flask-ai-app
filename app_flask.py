# 📚 Import Libraries
from flask import Flask, request, jsonify, render_template
import joblib
import pandas as pd

# Initialize Flask App
app = Flask(__name__)

# Load Optimized Random Forest Model
model = joblib.load('rf_model_optimized.joblib')

# ------------------------------
# 1️⃣ ROUTES
# ------------------------------

# Home Route (Web UI)
@app.route('/')
def home():
    return render_template('index.html')

# Predict Route (API for POST requests)
@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get data from the POST request
        data = request.get_json()

        # Convert data to DataFrame
        input_data = pd.DataFrame([data])

        # Make Prediction
        prediction = model.predict(input_data)[0]

        # Return Result as JSON
        return jsonify({'predicted_class': int(prediction)})

    except Exception as e:
        return jsonify({'error': str(e)})

# Run Flask App
if __name__ == '__main__':
    app.run(debug=True)
