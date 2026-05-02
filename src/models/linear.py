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
        x = x.reshape(-1)
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

    def cross_entropy_loss(self, X, y):
        scores = self.forward(X)
        P = self.softmax(scores)

        n = X.shape[0]
        return -np.mean(np.log(P[np.arange(n), y] + 1e-15))

    def gradients(self, X, y):
        n = X.shape[0]

        scores = self.forward(X)
        P = self.softmax(scores)

        Y = np.zeros_like(P)
        Y[np.arange(n), y] = 1

        dO = (P - Y) / n

        dA = dO.T @ X
        db = np.sum(dO, axis=0)

        return dA, db

    def train(self, X, y, lr=0.1, epochs=100):
        for epoch in range(epochs):
            loss = self.cross_entropy_loss(X, y)
            print(f"epoch {epoch}, loss = {loss}")
            
            dA, db = self.gradients(X, y)

            self.A -= lr * dA
            self.b -= lr * db

    def step_gradient(self, X, y, lr: float = 0.1):
        """
        Perform a single full-batch gradient descent update and return the
        gradients that were applied.
        """
        dA, db = self.gradients(X, y)
        self.A -= lr * dA
        self.b -= lr * db
        return dA, db

    def train_verbose(
        self,
        X_train,
        y_train,
        X_val=None,
        y_val=None,
        lr: float = 0.1,
        max_epochs: int = 1000,
        tol: float = 0.01,
        verbose: bool = True,
    ):
        """
        Train the model and print progress each epoch. If validation data is
        provided the stopping criterion is based on validation loss change;
        otherwise training loss is used.

        Returns a history dict with lists: train_loss, val_loss, val_error.
        """
        history = {"train_loss": [], "val_loss": [], "val_error": []}

        if X_val is not None and y_val is not None:
            prev_loss = self.cross_entropy_loss(X_val, y_val)
        else:
            prev_loss = self.cross_entropy_loss(X_train, y_train)

        if verbose:
            if X_val is not None and y_val is not None:
                print(f"Epoch 0: train_loss={self.cross_entropy_loss(X_train,y_train):.6f}, val_loss={prev_loss:.6f}, val_err={self.error_rate(X_val,y_val):.4f}")
            else:
                print(f"Epoch 0: train_loss={prev_loss:.6f}, val_err={self.error_rate(X_train,y_train):.4f}")

        for epoch in range(1, max_epochs + 1):
            # perform one training step
            self.step_gradient(X_train, y_train, lr=lr)

            train_loss = self.cross_entropy_loss(X_train, y_train)
            if X_val is not None and y_val is not None:
                val_loss = self.cross_entropy_loss(X_val, y_val)
                val_err = self.error_rate(X_val, y_val)
            else:
                val_loss = train_loss
                val_err = self.error_rate(X_train, y_train)

            history["train_loss"].append(train_loss)
            history["val_loss"].append(val_loss)
            history["val_error"].append(val_err)

            if verbose:
                print(f"Epoch {epoch}: train_loss={train_loss:.6f}, val_loss={val_loss:.6f}, val_err={val_err:.4f}")

            if abs(prev_loss - val_loss) < tol:
                if verbose:
                    print(f"Converged at epoch {epoch} (loss change {abs(prev_loss - val_loss):.6f} < {tol})")
                break

            prev_loss = val_loss

        else:
            if verbose:
                print("Reached max epochs without meeting tolerance.")

        return history
    def error_rate(self, X, y):
        predictions = self.predict(X)
        return np.mean(predictions != y)