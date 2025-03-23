from flask import Flask, request, jsonify
import numpy as np
import pickle
from sklearn.preprocessing import StandardScaler

app = Flask(__name__)

# Load the pre-trained model and scaler
with open('hyptertension_model1.pkl', 'rb') as model_file:
    model = pickle.load(model_file)

with open('scaler_hyper.pkl', 'rb') as scaler_file:
    scaler = pickle.load(scaler_file)

# Define the mappings
mappings = {
    'gender': {'Male': 1, 'Female': 0},
    'BPMeds': {'Yes': 1, 'No': 0}
}

# Define the feature order
features = ['gender', 'age', 'cigsPerDay', 'BPMeds', 'totChol', 'sysBP', 'diaBP', 'BMI', 'heartRate']

@app.route('/predict', methods=['POST'])
def predict():
    data = request.json
    input_data = []

    for feature in features:
        value = data.get(feature)
        if value is None:
            return jsonify({'error': f'Missing value for {feature}'}), 400
        if feature in mappings:
            value = mappings[feature].get(value, value)
        input_data.append(float(value))  # Convert value to float

    # Convert to numpy array and reshape for normalization
    input_array = np.array(input_data).reshape(1, -1)

    # Normalize the data using the standard scaler
    normalized_data = scaler.transform(input_array)

    # Make prediction
    prediction = model.predict(normalized_data)
    print(f"This is array ${prediction}")

    return jsonify({
        'prediction': int(prediction[0]),
        }
 )

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)

