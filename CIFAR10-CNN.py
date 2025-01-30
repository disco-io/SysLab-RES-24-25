import pickle
import numpy as np
import matplotlib.pyplot as plt
import os

X_train, y_train, X_test, y_test, labels = (
    [],
    [],
    [],
    [],
    [
        "airplane",
        "automobile",
        "bird",
        "cat",
        "deer",
        "dog",
        "frog",
        "horse",
        "ship",
        "truck",
    ],
)
data_path = "cifar-10-batches-py"


def load_data():
    global X_train, y_train, X_test, y_test
    # load training data
    X_train, y_train = [], []
    for i in range(1, 6):
        file_path = os.path.join(data_path, f"data_batch_{i}")
        with open(file_path, "rb") as file:
            batch = pickle.load(file, encoding="bytes")
        X_train.append(batch[b"data"].reshape(-1, 3, 32, 32).transpose(0, 2, 3, 1))
        y_train.append(np.array(batch[b"labels"]))

    X_train = np.concatenate(X_train)
    y_train = np.concatenate(y_train)

    # load testing data
    file_path = os.path.join(data_path, "test_batch")
    with open(file_path, "rb") as file:
        batch = pickle.load(file, encoding="bytes")
    X_test = batch[b"data"].reshape(-1, 3, 32, 32).transpose(0, 2, 3, 1)
    y_test = np.array(batch[b"labels"])


def plot_samples():
    fig, axes = plt.subplots(10, 10, figsize=(15, 15))
    axes = axes.flatten()
    for class_idx in range(10):
        class_images = X_train[y_train == class_idx]
        for i in range(10):
            image = class_images[i]
            axes[class_idx * 10 + i].imshow(image)
            axes[class_idx * 10 + i].axis("off")
            axes[class_idx * 10 + i].set_title(labels[class_idx], fontsize=8)
    plt.tight_layout()
    plt.show()


load_data()

plot_samples()
