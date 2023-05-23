from flask import Flask, request, jsonify
from flask_cors import CORS
import joblib
import numpy as np

app = Flask(__name__)
CORS(app)

# Load your trained model
model = joblib.load('chat_model.joblib')

@app.route('/predict', methods=['POST'])
def predict():
    new_text = request.form.get('newText')
    
    # Reshape the input data to match the expected format
    new_text_reshaped = np.array([new_text]).reshape(-1, 1)
    
    # Perform the prediction based on your loaded model
    predicted_label = model.predict(new_text_reshaped)[0]
    
    response = jsonify({'predicted_label': predicted_label})
    return response

if __name__ == '__main__':
    app.run(port=7083)
