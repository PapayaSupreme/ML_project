import numpy as np


class LinearMultiClassModel:
    """
    Linear multi-class model for MNIST-like images.

    For each input vector x of size 784, the model computes:

        o = A @ x + b

    where:
        x is shape (784,1)
        A is shape (10, 784)
        b is shape (10,1)
        o is shape (10,1)

    Each output o[k] is the score/logit for digit class k.
    """

    def __init__(self, input_size: int = 784, num_classes: int = 10):
        self.input_size = input_size
        self.num_classes = num_classes

        # A contains the weights.
        # One row per class, one column per pixel.
        self.A = np.random.randn(num_classes, input_size) * 0.01

        # b contains one bias per class.
        self.b = np.zeros(num_classes)

    def forward_one(self, x: np.ndarray) -> np.ndarray:
        """
        Compute the score vector for one image.

        Args:
            x: one image vector of shape (784,1)

        Returns:
            o: score vector of shape (10,1)
        """
        return self.A @ x + self.b

    def forward(self, X: np.ndarray) -> np.ndarray:
        """
        Compute scores for several images at once.

        Args:
            X: matrix of images with shape (n_samples, 784)

        Returns:
            O: score matrix with shape (n_samples, 10)
               O[i, k] is the score of image i for class k.
        """
        return X @ self.A.T + self.b

    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Predict the digit class for each image.

        The predicted class is the one with the highest score.
        """
        scores = self.forward(X)
        return np.argmax(scores, axis=1)

    def softmax(self, O: np.ndarray) -> np.ndarray:
        """
        Convert scores into probabilities using softmax.
        Works for batch input (n_samples, 10).
        """
        # stability trick: subtract max to avoid overflow
        O_shifted = O - np.max(O, axis=1, keepdims=True)

        exp_O = np.exp(O_shifted)
        return exp_O / np.sum(exp_O, axis=1, keepdims=True)