import matplotlib
from sklearn.datasets import make_blobs
from src.data_processor import KeystrokeDataReader
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from neural_network import NeuralNetwork
import matplotlib.pyplot as plt

def run_with_dummy_data():
    # color map for better visualization
    my_cmap = matplotlib.colors.LinearSegmentedColormap.from_list("", ["red", "yellow", "green"])

    # Generating 1000 observations with 10 labels
    data, labels = make_blobs(n_samples=1000, centers=3, n_features=3, random_state=30)

    # plot
    plt.scatter(data[:, 0], data[:, 1], c=labels, cmap=my_cmap)
    plt.show()

    # one hot encoding for y
    label_encoder = LabelEncoder()
    y_encoded = label_encoder.fit_transform(labels)

    # split the data
    X_train, X_test, y_train, y_test = train_test_split(data, y_encoded, stratify=labels, test_size=0.2,
                                                        random_state=42)

    # Create a neural network instance
    layer_structure = [X_train.shape[1], 4, 3]
    nn = NeuralNetwork(file=None, layer_structure=layer_structure, learning_rate=0.00002, activation='relu')

    # Train the neural network and make predictions on training data
    nn.train(X_train, y_train, epochs=1000)
    predictions = nn.get_prediction(X_test)
    print("_________ \n accuracy: ", nn.calculate_accuracy(y_test, predictions))
    print("_________ \n precision: ", nn.calculate_macro_precision(y_test, predictions))
    nn.plot_learning()

def fit_with_keystroke_data():
    data_processor = KeystrokeDataReader()
    # Load the dataset
    dataset = data_processor.get_train_data()

    # Extract features and target variable
    X = dataset.data  # Features
    y = dataset.labels  # Target labels

    # Encode the target variable (PARTICIPANT_ID) using LabelEncoder
    label_encoder = LabelEncoder()
    y_encoded = label_encoder.fit_transform(y)

    # Split the dataset into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y_encoded, test_size=0.2, random_state=42)

    # Initialize and train the neural network model
    layers = [X_train.shape[1], 7, 4, len(label_encoder.classes_)]
    nn = NeuralNetwork(layer_structure=layers, learning_rate=0.0001, activation='relu')
    nn.train(X_train, y_train, epochs=200)

    # Get predictions and calculate the mean squared error
    y_pred = nn.get_prediction(X_test)
    nn.plot_learning()
    nn.save_model('model')
    print("prediction: ", y_pred)
    print("accuracy: ", nn.calculate_accuracy(y_test, y_pred))
    print("precision: ", nn.calculate_macro_precision(y_test, y_pred))


if __name__ == "__main__":
    # save keystroke data in dataset
    data_processor = KeystrokeDataReader()
    keystroke_df = data_processor.read_keystroke_data('../data/Keystrokes/files/*_keystrokes.txt', 1000)
    data_processor.save_keystroke_dataset(keystroke_df, 10)

    fit_with_keystroke_data()
    print("Done")

