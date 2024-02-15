import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from neural_network import NeuralNetwork
import src.data_processor as data_processor

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error

def generate_data(size=100000):
    # Define correlation values
    corr_a = 0.8
    corr_b = 0.4
    corr_c = -0.2

    # Generate independent features
    a = np.random.normal(0, 1, size=size)
    b = np.random.normal(0, 1, size=size)
    c = np.random.normal(0, 1, size=size)
    d = np.random.randint(0, 4, size=size)
    e = np.random.binomial(1, 0.5, size=size)

    # Generate target feature based on independent features
    target = 50 + corr_a * a + corr_b * b + corr_c * c + d * 10 + 20 * e + np.random.normal(0, 10, size=size)

    # Create DataFrame with all features
    df = pd.DataFrame({'a': a, 'b': b, 'c': c, 'd': d, 'e': e, 'target': target})
    return df
def run_with_dummy_data():
    df = generate_data(10000)

    # Separate the features and target
    X = df.drop('target', axis=1)
    y = df['target']

    scaler = StandardScaler()
    X = scaler.fit_transform(X)

    # Split the dataset into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=22)
    y_train = y_train.to_numpy().reshape(-1, 1)
    y_test = y_test.to_numpy().reshape(-1, 1)

    layer_structure = [X_train.shape[1], 10, 10, 1]
    nn = NeuralNetwork(layer_structure, 0.0002, 'sigmoid')

    nn.train(X_train, y_train, epochs=1000)

    y_pred = nn.get_prediction(X_test)
    # nn.plot_learning() impleemnt this

    print("Test error: ", mean_squared_error(y_test, y_pred))

def run_with_keystroke_data():
    # Load your keystroke data into a DataFrame
    keystroke_df = data_processor.read_keystroke_data('../data/Keystrokes/files/*_keystrokes.txt', 13)
    print(keystroke_df.head())
    df_preprocessed = data_processor.preprocess_data(keystroke_df)

    # Extract features and target variable, skip headings
    X = df_preprocessed.drop(columns=['PARTICIPANT_ID', 'SEQUENCE_ID']).values  # Features
    y = (df_preprocessed['PARTICIPANT_ID']).values  # Target labels
    # Encode the target variable (SEQUENCE_ID) using LabelEncoder
    label_encoder = LabelEncoder()
    y_encoded = data_processor.encode_participant_ids(label_encoder.fit_transform(y))
    # data needs to have shape of output

    # Split the dataset into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y_encoded, test_size=0.2, random_state=42)

    # Initialize and train the neural network model
    layers = [X_train.shape[1], 63, 3, len(label_encoder.classes_)]
    nn = NeuralNetwork(layer_structure=layers, learning_rate=0.01)
    nn.train(X_train, y_train, epochs=1000)


if __name__ == "__main__":
    run_with_dummy_data()
    # run_with_keystroke_data()
    print("Done")

