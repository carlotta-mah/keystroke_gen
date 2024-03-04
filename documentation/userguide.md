# How to run the program

## Requirements
First activate the virtual environment by running the following command in the terminal: 

```source venv/bin/activate```

Before running the program, make sure you have the requirements in the `requirements.txt` file installed. You can install them by running the following command in the terminal: 

```pip install -r requirements.txt```

## Data
To train the model, you need to download the keystrokes dataset by Dhakal et al. [0]: https://userinterfaces.aalto.fi/136Mkeystrokes/#sect2

## Running the program
To run the program, you may to run the `main.py` file. You can do this by running the following command in the terminal: 

```src/python main.py```

However, the *recommended way* to run the program is using the Jupyter notebook [ks_model_interact.ipynb](..%2Fsrc%2Fks_model_interact.ipynb).

## How the different funcitonalites are used
### Data Preprocessing
The data preprocessing is done in the `data_reader.py` file. The data is read from the csv file and preprocessed. 
The preprocessing includes removing the columns that are not needed, normalizing the data, and one-hot encoding the labels, and feature engineering. 

### Model Training
The model training is done in the `neural_network.py` file. The model is trained using the `train` method.
The `train` method calls the `forward` and `backward` methods to propagate the input data through the network.

### Evaluation
We use Accuracy and Precision metrics to evaluate the model. The evaluation is done in the `neural_network.py` file.

All these functionalies are used in the `ks_model_interact.ipynb` file and can be assessed there.


