Neural Network House Price Prediction on Arduino
Overview

This project implements a simple neural network on an Arduino board to predict the price of a house based on specific features such as the transaction date, house age, distance to the nearest MRT station, number of convenience stores, latitude, and longitude. The code simulates the forward pass of a trained neural network model with 3 layers (2 hidden layers and an output layer) to predict house prices.
Dataset

The input dataset contains six features that represent the following:

    X1 transaction date - The year in which the house was transacted (e.g., 2014.000).
    X2 house age - The age of the house in years (e.g., 5 years old).
    X3 distance to the nearest MRT station - The distance from the house to the nearest MRT station (e.g., 300 meters).
    X4 number of convenience stores - The number of convenience stores near the house (e.g., 7 stores).
    X5 latitude - The latitude of the house (e.g., 24.980).
    X6 longitude - The longitude of the house (e.g., 121.540).

The model is trained using a similar dataset, and this implementation replicates the forward pass of that trained model to perform predictions on new data points.
Code Explanation
Layers and Weights

The neural network consists of three layers:

    Layer 1: This layer accepts 6 input features and produces 64 outputs. The weights (weights1) are a 6x64 matrix, and the biases (biases1) are a 64-element array. The output of this layer passes through a sigmoid activation function.

    Layer 2: This layer accepts 64 inputs from Layer 1 and produces 64 outputs. The weights (weights2) are a 64x64 matrix, and the biases (biases2) are a 64-element array. The output of this layer passes through the tanh activation function.

    Output Layer: This layer accepts 64 inputs from Layer 2 and produces a single output value (the predicted house price). The weights (weights3) are a 64x1 matrix, and the biases (biases3) consist of one value.
