from flask import Flask, request, jsonify, render_template
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences
import joblib
import numpy as np

# Initialize Flask app
app = Flask(__name__)

# Load the new Keras model
model = load_model('model.h5')  # ✅ Use model.h5

# Load tokenizer and label encoder
tokenizer = joblib.load('tokenizer.joblib')
label_encoder = joblib.load('label_encoder.joblib')

# Max sequence length (used during training)
MAX_LEN = model.input_shape[1]

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.get_json()
        user_input = data.get('user_message', '')

        # Tokenize and pad input
        seq = tokenizer.texts_to_sequences([user_input])
        padded_seq = pad_sequences(seq, maxlen=MAX_LEN, padding='post')

        # Predict
        prediction = model.predict(padded_seq)[0][0]
        sentiment = 'positive' if prediction > 0.5 else 'negative'

        return jsonify({
            'user_input': user_input,
            'predicted_class': sentiment,
            'confidence': f"{prediction * 100:.2f}%"
        })

    except Exception as e:
        return jsonify({'error': str(e)})

if __name__ == '__main__':
    app.run(debug=True)
