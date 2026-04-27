import numpy as np


def train_test_split(X: np.ndarray, y: np.ndarray, test_ratio: float = 0.2, seed: int | None = None):
    """
    Splits dataset into train and test sets with random shuffling.

    Args:
        X: input data, shape (n_samples, n_features)
        y: labels, shape (n_samples,)
        test_ratio: proportion of data to use for test set (e.g., 0.2 = 20%)
        seed: optional random seed for reproducibility

    Returns:
        X_train, X_test, y_train, y_test
    """

    assert 0 < test_ratio < 1, "test_ratio must be between 0 and 1"
    assert len(X) == len(y), "X and y must have the same number of samples"

    n_samples = X.shape[0]

    # Set seed for reproducibility if provided
    if seed is not None:
        np.random.seed(seed)

    # Generate shuffled indices
    indices = np.random.permutation(n_samples)

    # Compute split index
    test_size = int(n_samples * test_ratio)

    test_indices = indices[:test_size]
    train_indices = indices[test_size:]

    # Split the data
    X_train = X[train_indices]
    X_test = X[test_indices]
    y_train = y[train_indices]
    y_test = y[test_indices]

    return X_train, X_test, y_train, y_test