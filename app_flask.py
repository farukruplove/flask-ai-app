from flask import Flask, request, jsonify, render_template
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences
import joblib
import numpy as np

# Initialize Flask app
app = Flask(__name__)

# Load the trained Keras model
model = load_model('model.h5')

# Load the tokenizer and label encoder
tokenizer = joblib.load('tokenizer.joblib')
label_encoder = joblib.load('label_encoder.joblib')

# Maximum sequence length (used during training)
MAX_LEN = model.input_shape[1]

# ------------------------------
# 1️⃣ Home Route (Web Interface)
# ------------------------------
@app.route('/')
def home():
    return render_template('index.html')

# ------------------------------
# 2️⃣ Prediction API Route
# ------------------------------
@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get user message from JSON payload
        data = request.get_json()
        user_input = data.get('user_message', '')

        # Preprocess: tokenize and pad the input text
        seq = tokenizer.texts_to_sequences([user_input])
        padded_seq = pad_sequences(seq, maxlen=MAX_LEN, padding='post')

        # Make prediction
        prediction = model.predict(padded_seq)[0][0]

        # Decode label (if binary classification)
        sentiment = 'positive' if prediction > 0.5 else 'negative'

        return jsonify({
            'user_input': user_input,
            'predicted_class': sentiment,
            'confidence': f"{prediction * 100:.2f}%"
        })

    except Exception as e:
        return jsonify({'error': str(e)})

# ------------------------------
# 3️⃣ Run Flask App
# ------------------------------
if __name__ == '__main__':
    app.run(debug=True)
