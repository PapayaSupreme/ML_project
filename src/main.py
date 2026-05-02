from src.models.linear import LinearMultiClassModel
from src.models.neural_network import NeuralNetworkModel
from src.utilities.vectorize_images import load_vectorized_digits

import numpy as np
import torch



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
            print("6. Train the neural network model")
            print("7. Test the neural network model")
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
            for a in range(2):
                models, losses = [], []
                for j in range(30):
                    nn_model = NeuralNetworkModel(784, 10, a+1, 16)

                    loss = 0
                    for i in range(len(X)):
                        image, result = X[i], y[i]
                        prediction = nn_model.forward(image)
                        loss += -np.log(prediction[result] + 0.00001)
                        nn_model.its_backpropagation_time(prediction, image, result, 0.05)
                    avg_loss = loss / len(X)

                    models.append(nn_model)
                    losses.append(avg_loss)
                    print(str(j) + " : " + str(round(avg_loss, 4)))

                minIndex = 0
                for k in range(len(losses)):
                    if losses[k] < losses[minIndex]: minIndex = k
                print("Minimum on model " + str(minIndex) + " at loss = " + str(round(losses[minIndex], 4)))
                torch.save(models[minIndex].state_dict(), str(a+1) + "_hidden.pth")

        elif choice == 7:
            nn_model = NeuralNetworkModel(784, 10, 1, 3)
            nn_model.load_state_dict(torch.load("1_hidden.pth"))
        elif choice == 0:
            end = True
    print("Shutting down...")

