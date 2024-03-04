# Implementation document
This document is a summary of the implementation of the neural network. 
It is a part of the final project for the course "Algorithms and AI" at the University of Helsinki. 

## Data processing
The data processor is responsible for processing the keystroke data in the dataset by Dhakal et al. [0]: https://userinterfaces.aalto.fi/136Mkeystrokes/#sect2.
In the provided dataset we find war keystroke data that consists of key press and key release times for each keystroke. In one file of keystroke data we find the data of one participant.
- `read_keystroke_data`: Read the raw keystroke data and store it in pandas dataframe
- `save_keystroke_dataset`: Saves the processed keystroke data in pytorch datasets. Here, we use the custom Dataset class `KeystrokeDataset`
- `preprocess_data`: This function provides scaled keystroke data that is grouped into sentences.
- `encode_participant_ids`: Used for one-hot-encoding
- `extract_features`: Calculates features from preprocessed keystroke data

## Neural network
The main part of this project, neural network, is implemented in the file `neural_network.py`. I have implemented a class called `NeuralNetwork` which is a feedforward neural network.
This network is capable of processing categorical data with multiple labels. 

### Structure
The network has the following fields that are managed by the class:
- `layer_structure`: A list of numbers of layers in the network. The first layer is the input layer, the last layer is the output layer, and the layers in between are hidden layers.
- `learning_rate`: The learning rate of the network. This is a hyperparameter that determines how much the weights are updated during training.
- `batch_size`: The number of samples in a batch. The network is trained using mini-batch gradient descent. This is a hyperparameter that determines how many samples are used to update the weights at a time.
- `activation`: The activation function used in the hidden layers of the network. The network uses the relu activation function by default.
- `validation_split`: The proportion of the training data that is used for validation. This is a hyperparameter that determines how much of the training data is used to validate the network during training.
- `weights`: The weights of the network. These are initialized randomly and updated during training.
- `biases`: The biases of the network. These are initialized with ones and updated during training.
- `losses`: The losses of the network. These are used to keep track of the loss during training.

### Training
The network can be trained using the `train` method. 
In the training process, the network uses mini-batch gradient descent to update the weights and biases. 
First, the training data is split into mini-batches. These are propagated through the network by the `forward` method.
In the forward pass, the choosen activation function is used for the hidden layers and the softmax function is used for the output layer.
By doing this we receive an easy-to-understand output of probabilities for each class. 
Also, using the softmax function, the output of the network is normalized to sum to 1, which makes it easier to calculate the gradient needed for the backpropagation.
The output of the forward pass is compared to the true labels using the cross-entropy loss function, implemented in the method `cross_entropy`. 
Then the `backward` method is used to calculate the gradients of the weights and biases. 
Finally, the weights and biases are updated using the gradients and the learning rate.


### Inference
The network can be used to make predictions using the `predict` method. Here, I call the `forward` method to propagate the input data through the network. 
The output of the forward pass is modified so that only the class with the highest probability is set to 1 and the rest are set to 0.

## Time Complexity
- Forward Pass: For each layer, we calculate matrix multiplication of the input with the weights. 
Matrix multiplication has a time complexity of O(n^3) where n is the number of data points. However, numpys implementation of matrix multiplication may be faster.
  We also perform the activation function in each iteration. Relu has a complexity of O(n), where n is the number of data points.
  The time complexity of the forward pass is O(l*n^3) where l is the number of layers and n is the number of data points.
- Backward Pass: For each layer, we calculate the gradients multiply these with the output of the previous layer. 
  The relu derivative has a time complexity of O(n) where n is the number of data points. 
  The multiplication of the gradients with the output of the previous layer has a time complexity of O(n^3), where k is the number of features.
  The complexity is O(l*n^3) where l is the number of layers, n is the number of data points and k is the number of features.
- The time complexity of the training process is O(n^3 *e)  where n is the number of data points and e is the number of epochs.

## Space Complexity
- One-hot encoding: The one-hot encoding of the labels has a space complexity of O(n*k) where n is the number of data points and k is the number of classes.
- Weights and biases: The space complexity of the weights and biases is O(l*n) where l is the number of layers and n is the number of neurons in each layer.
- Activation: The space complexity of the activation is O(n*f) where n is the number of data points and f is the number of features.

## Improvements
- Layers as classes: The layers could be implemented as classes. This would make the code more modular and easier to understand.
- Adding Layers with different activation functions: Add the possibility to use different activation functions in different layers.
- Regularization: Add regularization to the network to prevent overfitting.
- Use of different optimizers: Add the possibility to use different optimizers in the network.
- Use of different loss functions: Add the possibility to use different loss functions in the network.
- Feature engineering: My data does not produce any good results. Possibly, the data needs to get better feature engineering to produce better results.

## Use of LLMs
I used ChatGPT to explain the concepts of the neural network and gain better understanding of their functioning. 
It also generated code and pseudocode for neural networks, which was not directly used or copied.

## References
- Dataset: https://userinterfaces.aalto.fi/136Mkeystrokes/#sect2
- Test data:
  - https://stackoverflow.com/questions/47372685/softmax-function-in-neural-network-python
  - https://neuralthreads.medium.com/categorical-cross-entropy-loss-the-most-important-loss-function-d3792151d05b
- Resouces: 
  - http://www.jbgrabowski.com/notebooks/neural-net/
  - https://www.pinecone.io/learn/cross-entropy-loss/
  - https://towardsdatascience.com/deriving-backpropagation-with-cross-entropy-loss-d24811edeaf9
  - https://alexcpn.medium.com/yet-another-backpropagation-article-20ae57aabd1e