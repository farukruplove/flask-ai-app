from flask import Flask, request, jsonify, render_template
import joblib
import pandas as pd

app = Flask(__name__)

# Load model
model = joblib.load('rf_model.joblib')

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json()
    input_df = pd.DataFrame([data])
    prediction = model.predict(input_df)
    return jsonify({'predicted_class': int(prediction[0])})

if __name__ == '__main__':
    app.run(debug=True)
