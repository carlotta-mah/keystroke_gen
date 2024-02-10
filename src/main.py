import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from neural_network import NeuralNetwork
import src.data_processor as data_processor

if __name__ == "__main__":
    # Load your keystroke data into a DataFrame
    keystroke_df = data_processor.read_keystroke_data('../data/Keystrokes/files/*_keystrokes.txt', 10)
    print(keystroke_df.head())
    df_preprocessed = data_processor.preprocess_data(keystroke_df)

    # Extract features and target variable, skip headings
    X = df_preprocessed.drop(columns=['PARTICIPANT_ID', 'SEQUENCE_ID']).values # Features
    y = (df_preprocessed['PARTICIPANT_ID']).values  # Target labels
    # Encode the target variable (SEQUENCE_ID) using LabelEncoder
    label_encoder = LabelEncoder()
    y_encoded = label_encoder.fit_transform(y)

    # Split the dataset into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Initialize and train the neural network model
    input_size = X_train.shape[1]
    output_size = len(label_encoder.classes_)  # Number of unique sequence IDs
    nn = NeuralNetwork(input_size=input_size, layer_structure=[64, 32], output_size=output_size, learning_rate=0.01)
    nn.train(X_train, y_train, epochs=1000)
