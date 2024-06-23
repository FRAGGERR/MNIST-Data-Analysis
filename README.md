# MNIST Digit Classification with PyTorch

This repository contains a neural network model implemented using PyTorch for classifying handwritten digits from the MNIST dataset. The model architecture includes two hidden layers and uses sigmoid activations and softmax for the output layer. This project demonstrates the end-to-end process of loading the dataset, transforming the data, defining and training the model, and evaluating its performance.

## Repository Contents

- **Dataset Loading**: Scripts to download and load the MNIST dataset using `torchvision.datasets.MNIST`.
- **Data Transformation**: Applying transformations to the dataset including normalization and reshaping.
- **Model Definition**: Implementation of a neural network model class with:
  - Two hidden layers with 128 and 64 neurons respectively.
  - Sigmoid activation functions for hidden layers.
  - Softmax activation function for the output layer.
- **Training and Evaluation**: 
  - Training the model over multiple epochs with a training loop.
  - Evaluation function to calculate accuracy on the test dataset.
- **Model Saving and Loading**: Saving the trained model's weights and full architecture for future inference.
- **Testing Predictions**: Running predictions on a batch of test data to validate the model's performance.

## Files

- `mnist_classification.py`: Main script containing the entire pipeline from data loading, training, to evaluation.
- `mnist_model.pth`: Saved weights of the trained model.
- `full_model.pth`: Full model (including architecture and weights) saved for future use.

## How to Use

1. **Setup Environment**:
   - Ensure you have Python and PyTorch installed. You can set up a virtual environment and install the required packages using:
     ```bash
     pip install torch torchvision tqdm
     ```

2. **Run the Script**:
   - To train the model and evaluate it, simply run the main script:
     ```bash
     python mnist_classification.py
     ```

3. **Load and Evaluate Pre-trained Model**:
   - To evaluate the pre-trained model on the test dataset, you can use the following code snippet:
     ```python
     import torch
     from mnist_classification import Model

     # Load the saved model
     model = Model([128, 64, 10])
     model.load_state_dict(torch.load("mnist_model.pth"))

     # Evaluate
     model.evaluate(test_dataloader)
     ```

## Model Architecture

The model consists of:
- **Input Layer**: 784 neurons (28x28 flattened image).
- **Hidden Layer 1**: 128 neurons with Sigmoid activation.
- **Hidden Layer 2**: 64 neurons with Sigmoid activation.
- **Output Layer**: 10 neurons (for 10 digit classes) with Softmax activation.

## Results

The trained model achieves an accuracy of approximately (mention your accuracy) on the MNIST test dataset.

## Contributing

Feel free to submit issues or pull requests if you have any improvements or bug fixes.

## License

This project is licensed under the MIT License.

---

This description provides an overview of my project, including its functionality, how to use it, and details about the model architecture. Adjust the details as needed based on your specific implementation and results.
