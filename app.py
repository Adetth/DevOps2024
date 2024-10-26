from flask import Flask, request, jsonify
import pickle
import numpy as np

# Load the trained model from the pickle file
with open('diabetes_model.pkl', 'rb') as file:
    model = pickle.load(file)

# Create a Flask app
app = Flask(__name__)

# Define an endpoint to make predictions
@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json(force=True)
    print("incoming data : ",data)  
    input_data = np.array(data['input']).reshape(1, -1)
    print("after reshape ",input_data)
    prediction = model.predict(input_data)
    return jsonify({'prediction': prediction[0]})

# Run the Flask app
if __name__ == '__main__':
    app.run(port=5000, host='0.0.0.0')