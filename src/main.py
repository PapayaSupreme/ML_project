import torch

from src.models.linear import LinearMultiClassModel
from src.models.neural_network import NeuralNetworkModel
from src.utilities.vectorize_images import load_vectorized_digits
from src.utilities.train_test_splitter import train_test_split
from src.utilities.debug_utils import load_debug_tiles

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
    last_model = None
    while not end:
        choice = -1
        while choice < 0 or choice > 5:
            print("\n=== MAIN MENU ===")
            print("1. (re)Load data")
            print("2. Train & test the linear Model")
            print("3. Test a random debug tile with the last trained model")
            print("4. Train & test the neural network model")
            print("5. Load a trained neural model")
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
            # Split data into train/validation so training can monitor generalization
            try:
                test_ratio_in = input("Validation (test) ratio (0.0-0.5) [default 0.2]: ")
                test_ratio = float(test_ratio_in) if test_ratio_in.strip() != "" else 0.2
            except ValueError:
                test_ratio = 0.2

            X_train, X_val, y_train, y_val = train_test_split(X, y, test_ratio=test_ratio, seed=42)

            # Initialize model with matching input size and number of classes
            model = LinearMultiClassModel(input_size=X.shape[1], num_classes=len(np.unique(y)))

            # Sub-menu: one GD iteration or full train until convergence
            sub_choice = -1
            while sub_choice not in (1, 2):
                print("\nLinear model options:")
                print("1. Perform ONE gradient descent iteration on training data (reports validation error)")
                print("2. Full training loop (prints train/val loss each epoch, stops when val loss diff < tol)")
                sub_choice = int(input("Enter your choice (1 or 2): "))

            if sub_choice == 1:
                try:
                    lr_in = input("Learning rate [default 0.1]: ")
                    lr = float(lr_in) if lr_in.strip() != "" else 0.1
                except ValueError:
                    lr = 0.1

                train_loss_before = model.cross_entropy_loss(X_train, y_train)
                val_err_before = model.error_rate(X_val, y_val)

                # use model method to do a single step
                model.step_gradient(X_train, y_train, lr=lr)

                train_loss_after = model.cross_entropy_loss(X_train, y_train)
                val_err_after = model.error_rate(X_val, y_val)

                print(f"Train loss before: {train_loss_before:.6f}, after: {train_loss_after:.6f}")
                print(f"Validation error before: {val_err_before:.4f}, after: {val_err_after:.4f}")
                # store trained model reference
                last_model = model
                print("Last trained model stored in memory. You can now use option 3 to test debug tiles.")

            elif sub_choice == 2:
                try:
                    lr_in = input("Learning rate [default 0.1]: ")
                    lr = float(lr_in) if lr_in.strip() != "" else 0.1
                except ValueError:
                    lr = 0.1

                try:
                    max_epochs_in = input("Max epochs [default 1000]: ")
                    max_epochs = int(max_epochs_in) if max_epochs_in.strip() != "" else 1000
                except ValueError:
                    max_epochs = 1000

                try:
                    tol_in = input("Stopping tolerance on validation loss diff [default 0.01]: ")
                    tol = float(tol_in) if tol_in.strip() != "" else 0.01
                except ValueError:
                    tol = 0.01

                # Delegate training to the model's training function
                history = model.train_verbose(
                    X_train,
                    y_train,
                    X_val=X_val,
                    y_val=y_val,
                    lr=lr,
                    max_epochs=max_epochs,
                    tol=tol,
                    verbose=True,
                )
                # Keep reference to the last trained model in memory
                last_model = model
                print("Last trained model stored in memory. You can now use option 3 to test debug tiles.")
            # end training subchoices
        elif choice == 3:
            # Test a random debug tile image using the last trained model
            if last_model is None:
                print("No trained model available. Train a model first (option 2 or 5).")
            else:
                try:
                    X_debug, names = load_debug_tiles(normalize=True)
                except Exception as e:
                    print(f"Could not load debug tiles: {e}")
                    continue

                idx = np.random.randint(0, X_debug.shape[0])
                x = X_debug[idx:idx+1]
                name = names[idx]

                # Check if model is LinearMultiClassModel or NeuralNetworkModel
                if isinstance(last_model, LinearMultiClassModel):
                    # Linear model: use numpy-based softmax
                    probs = last_model.softmax(last_model.forward(x))
                    pred = int(np.argmax(probs, axis=1)[0])
                    # show top probabilities
                    top_k = 3
                    top_idx = np.argsort(probs[0])[::-1][:top_k]
                    top_probs = [(int(i), float(probs[0, i])) for i in top_idx]
                else:
                    # NeuralNetworkModel: expects a flat vector and returns logits as list
                    x_flat = x.flatten()
                    logits = last_model.forward(x_flat)
                    # Convert logits to probabilities (softmax)
                    logits_arr = np.array(logits)
                    exp_logits = np.exp(logits_arr - np.max(logits_arr))
                    probs = exp_logits / np.sum(exp_logits)
                    pred = int(np.argmax(probs))
                    # show top probabilities
                    top_k = 3
                    top_idx = np.argsort(probs)[::-1][:top_k]
                    top_probs = [(int(i), float(probs[i])) for i in top_idx]

                print(f"Debug tile: {name}")
                print(f"Predicted digit: {pred}")
                print("Top probabilities:")
                for digit, p in top_probs:
                    print(f"  {digit}: {p:.4f}")
        elif choice == 4:
            for a in range(2):
                models, losses = [], []
                for j in range(30):
                    nn_model = NeuralNetworkModel(784, 10, a + 1, 16)

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
                torch.save(models[minIndex].state_dict(), str(a + 1) + "_hidden.pth")
        elif choice == 5:
            nn_model = NeuralNetworkModel(784, 10, 1, 3)
            nn_model.load_state_dict(torch.load("1_hidden.pth"))
            last_model = nn_model
            print("Neural network model loaded and stored in memory. You can now use option 3 to test debug tiles.")
        elif choice == 0:
            end = True
    print("Shutting down...")

