from flask import Flask, request, jsonify
from neural_network import NeuralNetwork
app = Flask(__name__)
nn : NeuralNetwork = None
# Define routes
@app.route('/train-model', methods=['POST'])
def train_model():
    # Implement logic to train the model
    nn = NeuralNetwork([10, 10, 1], 0.0002, 'softmax')

    return jsonify({'message': 'Model trained successfully'})

@app.route('/participants', methods=['GET'])
def get_participants():
    # Implement logic to get participants data from data that was
    # used to train the model

    #dummy data
    participants = [
        {'id': 1,  'sequences': [1, 2, 3]},
        {'id': 2, 'sequences': [4, 5, 6]},
        # Add more participants as needed
    ]
    return jsonify(participants)

@app.route('/predict', methods=['POST'])
def predict():
    # Implement prediction logic here
    # dummy data
    prediction = nn.predict(request.json['data'])
    return jsonify({'prediction': prediction})

if __name__ == '__main__':
    app.run(debug=True)
