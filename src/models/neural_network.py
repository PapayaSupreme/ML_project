import numpy as np
import torch
import torch.nn as nn
from torchsummary import summary
import random

class NeuralNetworkModel(torch.nn.Module):
    def __init__(self, input_size: int, num_classes: int, hidden_layer_count: int, layer_size: int):
        super().__init__()
        self.sigmoid = nn.Sigmoid()
        self.softmax = nn.Softmax(dim=0)

        self.input_size = input_size
        self.output_size = num_classes
        self.layer_count = hidden_layer_count + 2
        self.hidden_layer_count = hidden_layer_count
        self.layer_size = layer_size
        self.layers = []

        # Each layer (except the last) will have as many neurons as 'layer_size'.
        for layer in range(self.hidden_layer_count + 1): self.layers.append(nn.Linear(self.layer_size, self.layer_size))
        self.layers.append(nn.Linear(self.layer_size, self.output_size))

        # Initializing weights.
        self.reset_weights()

    def reset_weights(self):
        for layer_index in range(self.layer_count):
            coeffs_array = []
            inputs_count = self.input_size if layer_index == 0 else self.layer_size
            current_neurons = self.layer_size if layer_index < self.layer_count - 1 else self.output_size

            for _ in range(inputs_count):
                new_coeffs = [(random.random() * 0.2 - 0.1) for _ in range(current_neurons)]     # Random values in [-0.1 ; 0.1].
                coeffs_array.append(new_coeffs)

            coeff = np.array(coeffs_array, dtype=np.float32)
            self.layers[layer_index].weight.data = torch.tensor(coeff).t()

    def forward(self, x):
        x = torch.tensor(x, dtype=torch.float32)
        self.neuron_results = []

        for layer_index in range(self.layer_count):
            if layer_index < self.layer_count - 1:
                # We use sigmoid activation function for each layer except the last.
                x = self.sigmoid(self.layers[layer_index](x))
            else:
                # Softmax as the last activation function.
                x = self.softmax(self.layers[-1](x))
            self.neuron_results.append(x.tolist())
        return x.tolist()

    def model_summary(self):
        summary(self, input_size=(self.input_size,))

    def predict(self, X: np.ndarray) -> np.ndarray:
        scores = self.forward(X)
        return np.argmax(scores, axis=1)

    def last_layer_initialization(self, P, y):
        delta_column = []
        for i in range(len(self.neuron_results[-1])):
            reducing = 1 if i == y else 0
            delta_column.append(P[i] - reducing)
        return delta_column

    def fill_neurons_derivatives_table(self, P, y):
        last_delta = [P[i] - (1 if i == y else 0) for i in range(len(P))]
        delta = [last_delta]
        for i in range(self.layer_count - 2, -1, -1):
            delta_column = []
            next_layer_weights = self.layers[i+1].weight.data
            for q in range(self.layers[i].out_features):
                error_propagation = sum([delta[0][j] * next_layer_weights[j][q].item() for j in range(len(delta[0]))])
                z = self.neuron_results[i][q]
                phi_prime = z * (1 - z)
                delta_column.append(error_propagation * phi_prime)
            delta.insert(0, delta_column)
        return delta

    def fill_weights_derivative_table(self, X, delta, neuron_results):
        new_weights_grads = []
        for h in range(self.layer_count):
            prev_activation = X if h == 0 else neuron_results[h - 1]
            current_delta = delta[h]
            layer_grads = []
            for q in range(len(current_delta)):
                row_grads = []
                for k in range(len(prev_activation)):
                    grad = current_delta[q] * prev_activation[k]
                    row_grads.append(grad)
                layer_grads.append(row_grads)
            new_weights_grads.append(layer_grads)
        return new_weights_grads

    def its_backpropagation_time(self, P, X, y, d):
        # He then proceeded to backpropagate all over the place
        # Calculation of the dE/do and dE/da.
        deltas = self.fill_neurons_derivatives_table(P, y)
        weight_grads = self.fill_weights_derivative_table(X, deltas, self.neuron_results)

        # Gradient descent.
        with torch.no_grad():
            for h in range(self.layer_count):
                for q in range(self.layers[h].out_features):
                    for k in range(self.layers[h].in_features):
                        self.layers[h].weight[q][k] -= d * weight_grads[h][q][k]