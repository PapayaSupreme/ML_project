from src.models.linear import LinearMultiClassModel
from src.utilities.vectorize_images import load_vectorized_digits

import numpy as np



if __name__ == "__main__":
    print("Loading data...")
    X, y = load_vectorized_digits(
        normalize=True,
        save_vectors=True,
        save_debug=True,
    )

    print(f"X shape: {X.shape}, dtype: {X.dtype}")
    print(f"y shape: {y.shape}, labels: {np.unique(y)}")
    print("Saved vectors to: data/vectorized_output/X_vectors.npy")
    print("Saved labels to:  data/vectorized_output/y_labels.npy")
    print("Saved debug files to: data/vectorized_output/debug/")

    print("=== DATA IS READY ===")
    end = False
    while not end:
        choice = -1
        while choice < 0 or choice > 6:
            print("\n=== MAIN MENU ===")
            print("1. (re)Load data")
            print("2. Train & test the linear Model")
            print("3. [WIP] Train & test the cross entropy / log loss model")
            print("4. [WIP] Train & test the gradient descent model")
            print("5. [WIP] Error calculation")
            print("6. [WIP] Train & test the neural network model")
            print("0. EXIT")
            choice = int(input("Enter your choice: "))
        if choice == 1:
            print("Loading data...")
            X, y = load_vectorized_digits(
                normalize=True,
                save_vectors=True,
                save_debug=True,
            )

            print(f"X shape: {X.shape}, dtype: {X.dtype}")
            print(f"y shape: {y.shape}, labels: {np.unique(y)}")
            print("Saved vectors to: data/vectorized_output/X_vectors.npy")
            print("Saved labels to:  data/vectorized_output/y_labels.npy")
            print("Saved debug files to: data/vectorized_output/debug/")
        elif choice == 2:
            model = LinearMultiClassModel()

            scores = model.forward(X)
            raw_predictions = model.softmax(scores)
            predictions = np.argmax(raw_predictions, axis=1)

            print(scores.shape)  # (n_samples, 10)
            print(predictions.shape)  # (n_samples,)
        elif choice == 3:
            print("[WIP]")
        elif choice == 4:
            print("[WIP]")
        elif choice == 5:
            print("[WIP]")
        elif choice == 6:
            print("[WIP]")
        elif choice == 0:
            end = True
    print("Shutting down...")

