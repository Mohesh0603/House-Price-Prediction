from flask import Flask, render_template, request, jsonify
import numpy as np
import pickle
import logging

app = Flask(__name__)
model = pickle.load(open('model.pkl', 'rb'))
scaler = pickle.load(open('scaler.pkl', 'rb'))

# Configure logging
logging.basicConfig(level=logging.INFO)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.get_json(force=True)
        logging.info(f"Received data: {data}")
        if not data:
            raise ValueError("No data provided")
        val1 = data.get('bedrooms')
        val2 = data.get('bathrooms')
        val3 = data.get('floors')
        val4 = data.get('yr_built')
        if None in [val1, val2, val3, val4]:
            raise ValueError("Missing values in data")
        arr = np.array([val1, val2, val3, val4]).reshape(1, -1)
        arr = arr.astype(np.float64)
        arr_scaled = scaler.transform(arr)
        pred = model.predict(arr_scaled)
        return jsonify({'prediction': int(pred[0])})
    except Exception as e:
        logging.error(f"Error during prediction: {e}")
        return jsonify({'error': str(e)}), 400

if __name__ == '__main__':
    app.run(debug=True)