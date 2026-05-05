import numpy as np
import torch
import torch.nn as nn
from torchsummary import summary
import random


class NeuralNetworkModel(torch.nn.Module):
    def __init__(self, input_size: int, num_classes: int, hidden_layer_count: int, layer_size: int):
        super().__init__()
        self.sigmoid = nn.Sigmoid()
        self.softmax = nn.Softmax(dim=-1)  # Fixed: use dim=-1 for last dimension (classes)

        self.input_size = input_size
        self.output_size = num_classes
        self.layer_count = hidden_layer_count + 2
        self.hidden_layer_count = hidden_layer_count
        self.layer_size = layer_size

        # Fixed: Use ModuleList to properly register layers with PyTorch
        self.layers = nn.ModuleList()

        # First layer: input_size -> layer_size
        self.layers.append(nn.Linear(self.input_size, self.layer_size))

        # Hidden layers: layer_size -> layer_size
        for _ in range(self.hidden_layer_count):
            self.layers.append(nn.Linear(self.layer_size, self.layer_size))

        # Output layer: layer_size -> num_classes
        self.layers.append(nn.Linear(self.layer_size, self.output_size))

        # Initialize weights
        self.reset_weights()

    def reset_weights(self):
        """Initialize weights with small random values in [-0.1, 0.1]."""
        for layer in self.layers:
            # Get the input and output dimensions from the layer
            in_features = layer.in_features
            out_features = layer.out_features

            # Create weight matrix: (in_features, out_features)
            coeffs_array = []
            for _ in range(in_features):
                new_coeffs = [(random.random() * 0.2 - 0.1) for _ in range(out_features)]
                coeffs_array.append(new_coeffs)

            coeff = np.array(coeffs_array, dtype=np.float32)
            layer.weight.data = torch.tensor(coeff).t()  # Transpose to match PyTorch's weight shape (out_features, in_features)
            layer.bias.data.zero_()  # Initialize bias to zero

    def forward(self, x):
        """Forward pass through the network."""
        # Convert input to tensor if needed
        if not isinstance(x, torch.Tensor):
            x = torch.tensor(x, dtype=torch.float32)

        # Ensure it's 1D or handle batches
        if x.dim() == 1:
            x = x.unsqueeze(0)  # Add batch dimension if needed

        self.neuron_results = []

        for layer_index in range(self.layer_count):
            x = self.layers[layer_index](x)

            if layer_index < self.layer_count - 1:
                # Sigmoid activation for hidden layers
                x = self.sigmoid(x)
            else:
                # Softmax activation for output layer
                x = self.softmax(x)

            # Store activations for backprop as 1D numpy arrays for a single-sample input
            if isinstance(x, torch.Tensor):
                arr = x.detach().cpu().numpy()
            else:
                arr = np.array(x)
            # If batch dimension present and equal to 1, squeeze it so downstream code sees a 1D vector
            if arr.ndim == 2 and arr.shape[0] == 1:
                arr = arr[0]
            self.neuron_results.append(arr)

        # Return as list and squeeze if batch dimension was added
        result = x.detach().cpu().numpy() if isinstance(x, torch.Tensor) else x
        return result[0] if result.shape[0] == 1 else result

    def model_summary(self):
        """Print model architecture summary."""
        summary(self, input_size=(self.input_size,))

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict digit classes for input images."""
        scores = self.forward(X)
        if isinstance(scores, list):
            scores = np.array(scores)
        if scores.ndim == 1:
            return np.argmax(scores)
        return np.argmax(scores, axis=1)

    def fill_neurons_derivatives_table(self, P, y):
        """Compute deltas for backpropagation."""
        # Convert P to list if it's a numpy array
        if isinstance(P, np.ndarray):
            P = P.tolist()

        last_delta = [P[i] - (1 if i == y else 0) for i in range(len(P))]
        delta = [last_delta]

        for i in range(self.layer_count - 2, -1, -1):
            delta_column = []
            next_layer_weights = self.layers[i + 1].weight.data

            for q in range(self.layers[i].out_features):
                error_propagation = sum([delta[0][j] * next_layer_weights[j][q].item() for j in range(len(delta[0]))])
                z = self.neuron_results[i][q] if isinstance(self.neuron_results[i], list) else self.neuron_results[i][q]
                phi_prime = z * (1 - z)
                delta_column.append(error_propagation * phi_prime)

            delta.insert(0, delta_column)

        return delta

    def fill_weights_derivative_table(self, X, delta, neuron_results):
        """Compute weight gradients for backpropagation."""
        new_weights_grads = []

        for h in range(self.layer_count):
            prev_activation = X if h == 0 else neuron_results[h - 1]

            # Convert to list if needed
            if isinstance(prev_activation, np.ndarray):
                prev_activation = prev_activation.tolist()

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
        """Perform backpropagation and update weights."""
        # Compute gradients
        deltas = self.fill_neurons_derivatives_table(P, y)
        weight_grads = self.fill_weights_derivative_table(X, deltas, self.neuron_results)

        # Update weights via gradient descent
        with torch.no_grad():
            for h in range(self.layer_count):
                for q in range(self.layers[h].out_features):
                    for k in range(self.layers[h].in_features):
                        self.layers[h].weight[q][k] -= d * weight_grads[h][q][k]

