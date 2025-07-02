# PyTorch MNIST Example

This example demonstrates how to train a simple neural network on the MNIST dataset using PyTorch.

## Requirements

*   PyTorch
*   torchvision
*   argparse

## Usage

To run the script, use the following command:

```bash
python Lab0/main.py --epochs 10 --lr 0.01 --optimizer adam --activation relu
```

### Arguments

*   `--batch-size`: Input batch size for training (default: 64)
*   `--test-batch-size`: Input batch size for testing (default: 1000)
*   `--epochs`: Number of epochs to train (default: 10)
*   `--lr`: Learning rate (default: 0.01)
*   `--momentum`: SGD momentum (default: 0.5)
*   `--no-cuda`: Disables CUDA training (default: False)
*   `--seed`: Random seed (default: 1)
*   `--log-interval`: How many batches to wait before logging training status (default: 10)
*   `--save-model`: For Saving the current Model (default: False)
*   `--optimizer`: Optimizer to use (default: sgd). Options: sgd, adam
*   `--activation`: Activation function to use (default: relu). Options: relu, sigmoid

## Example

To train the model with Adam optimizer and sigmoid activation function, use the following command:

```bash
python Lab0/main.py --optimizer adam --activation sigmoid