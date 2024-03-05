# keystroke detection 
In this repository create a neural network that discriminates keystroke dynamics of multiple typists.

## Data
The model is trained on the keystrokes dataset by Dhakal et al. [0]: https://userinterfaces.aalto.fi/136Mkeystrokes/#sect2, which contains keystroke data from 136,000 users.
Each user has typed 15 sentences, and the dataset contains the time it took to press and release each key.
The dataset is not included in the repository. Place the file in the `data` folder and unzip it.

## Model
The model predicts the user based on the keystroke data. For this, I will use a feedforward neural network architecture.
To do multi-class classification, I use the softmax activation function in the output layer and use the cross-entropy loss function.



