# Project Specifications

* Name: Mia Carlotta Mahncke
* Programming language: Python
* Preferred language of communication: English
* Degree:  M.Sc. Computer Science


The conventional approach of using CAPTCHA mechanisms to detect bots is loosing relevance, as bots adeptly enhance their image processing capabilities (e.g. [Jun-Yan Zhu et al., 2017](https://openaccess.thecvf.com/content_iccv_2017/html/Zhu_Unpaired_Image-To-Image_Translation_ICCV_2017_paper.html)). Countering this, biometric data may be used to distinguish between bots and humans. 
Keystroke dynamics have been successfully used for this task [Daniel DeAlcala et al., 2023](https://openaccess.thecvf.com/content/CVPR2023W/Biometrics/html/DeAlcala_BeCAPTCHA-Type_Biometric_Keystroke_Data_Generation_for_Improved_Bot_Detection_CVPRW_2023_paper.html)

For this project I plan to develop a neural network capable of classifying keystrokes of different typists to help authenticate the users.

Training data can be obtained in this [dataset by Vivek Dhakal et al](https://userinterfaces.aalto.fi/136Mkeystrokes/).


For my approach I have chosen to implement a feed forward neural network. For this I have identified the following tasks:
1. Data Preprocessing: Normalize and scale the keystroke data; Convert features (e.g., keycodes) into embeddings.
2. Feature Extraction: Identify and select relevant input features: Most research on this topic uses key press durations and inter-key timing.
3. Activation Function: Define an activation function (e.g., Relu or softmax).
4. Forward Propagation: Implement forward propagation through the network.
5. Loss Function: Define a loss function (e.g., binary cross-entropy or softmax loss).
6. Backpropagation: Implement backpropagation to update weights and biases.
7. Training Loop: Iterate through your dataset to train the model.
8. Prediction: Use the trained model for predictions.

The model will be accessible through a UI, that allows demonstration and manual testing.

## Complexity

Feedforward Pass:
The forward pass involves matrix multiplications and activation functions for each layer. If the network has L layers, and the size of each layer is approximately N, then the overall complexity of the forward pass is often considered to be O(L * N^3), since we do matrix multiplication at each layer. Assuming L <= N, the complexity is O(N^4).

Backpropagation:
The backward pass involves computing the gradients and updating weights. 
Using gradient descent, we know that t_(gd) = iterations * t_(w). 
The time to update weights for one iteration is O(n^4) ([see here for more detail](https://lunalux.io/computational-complexity-of-neural-networks/)). Therefore, we get a total complexity of O(i*n^4), with i being the number of iterations.