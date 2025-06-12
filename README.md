# Configurable 2D CNN with Numpy (from scratch)

A modular and configurable convolutional neural network (CNN) pipeline built entirely from scratch using NumPy. This project demonstrates the forward pass of a CNN capable of handling RGB image datasets such as CIFAR-10. 

## Features

- Written 100% from scratch using NumPy (no deep learning frameworks)
- User-defined architecture via configuration
- Supports:
  - Convolutional layers
  - ReLU activation
  - Max pooling
  - Fully connected layers
- Handles RGB images
- Clean and extensible codebase for learning or future extension
- (Optional) Download and store CIFAR-10 dataset with labels as numpy files
  > Torchvision was used to download the CIFAR-10 Dataset with Labels.

## What’s Not Included (Yet)

- Backpropagation and training loop
- Loss functions
- Accuracy/metric evaluation
- Batching
- Mapping predicted class to dataset labels

> This is a work in progress. Contributions or feedback are welcome!

## Motivation

This project was created to better understand the inner workings of CNNs and to build a deep learning pipeline entirely from scratch without relying on automatic differentiation or libraries like PyTorch or TensorFlow. The goal of the project is educational: to experiment with different CNN architectures, and explore how a configurable, modular CNN pipeline can be designed using only Numpy.

## How to Use

```python
import numpy as np
from model_test import test

# define your architecture for experimentation
architecture = [
    {"layer": "conv", "filters": 32, "kernel_size": 3, "stride": 1, "padding": 1, "pad_mode": "edge", "activation": "leaky_relu"},    
    {"layer": "pool", "type": "max", "kernel_size": 2, "pad_mode": "edge", "stride": 1},   
    {"layer": "conv", "filters": 64, "kernel_size": 3, "stride": 1, "pad_mode": "edge", "padding": 1, "activation": "leaky_relu"},
    {"layer": "pool", "type": "max", "pad_mode": "edge", "kernel_size": 2},
    {"layer": "fc", "fc_mode": "hidden", "units": 128, "activation": "leaky_relu", "activation_constant": 0.01},
    {"layer": "fc", "fc_mode": "output", "units": 10, "activation": "softmax"},
]

# Example RGB image (32x32x3)
np.random.seed(0)  # For reproducibility
image = np.random.rand(32, 32, 3).astype(np.float32)

# perform forward pass
test(architecture, image)
```
Output:

```
all probability: [0.10038781 0.09990939 0.10031709 0.10017366 0.09982597 0.09953602 0.09973153 0.10022148 0.10010099 0.09979605]

predicted class(es): [0] with score = 0.1003878061814276
```
