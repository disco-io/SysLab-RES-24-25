import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, TensorDataset
import matplotlib.pyplot as plt
import numpy as np
import pickle
import os

DATA_PATH = "cifar-10-batches-py"
CHECKPOINT_FILE = "checkpoint.pkl"
LABELS = [
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
]

# Initialize dataset variables
X_train, y_train, X_test, y_test = [], [], [], []


def load_data():
    global X_train, y_train, X_test, y_test

    X_train, y_train = [], []
    for i in range(1, 6):
        file_path = os.path.join(DATA_PATH, f"data_batch_{i}")
        with open(file_path, "rb") as file:
            batch = pickle.load(file, encoding="bytes")
        X_train.append(batch[b"data"].reshape(-1, 3, 32, 32).transpose(0, 2, 3, 1))
        y_train.append(np.array(batch[b"labels"]))

    X_train = np.concatenate(X_train)
    y_train = np.concatenate(y_train)

    file_path = os.path.join(DATA_PATH, "test_batch")
    with open(file_path, "rb") as file:
        batch = pickle.load(file, encoding="bytes")
    X_test = batch[b"data"].reshape(-1, 3, 32, 32).transpose(0, 2, 3, 1)
    y_test = np.array(batch[b"labels"])


def save_checkpoint(model, optimizer, epoch, train_losses, filename=CHECKPOINT_FILE):
    """Save model, optimizer state, and training loss history."""
    checkpoint = {
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "train_losses": train_losses,  # Save training loss history
    }
    with open(filename, "wb") as f:
        pickle.dump(checkpoint, f)
    print(f"Checkpoint saved at epoch {epoch}!")


def load_checkpoint(model, optimizer, filename=CHECKPOINT_FILE):
    """Load model, optimizer state, and training loss history."""
    if os.path.exists(filename):
        with open(filename, "rb") as f:
            checkpoint = pickle.load(f)
        model.load_state_dict(checkpoint["model_state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        print(f"Resuming from epoch {checkpoint['epoch']}!")
        return checkpoint["epoch"], checkpoint.get(
            "train_losses", []
        )  # Load training loss history if available
    return 0, []  # Start from scratch if no checkpoint exists


def plot_samples():
    fig, axes = plt.subplots(10, 10, figsize=(15, 15))
    axes = axes.flatten()
    for class_idx in range(10):
        class_images = X_train[y_train == class_idx]
        for i in range(10):
            image = class_images[i]
            axes[class_idx * 10 + i].imshow(image)
            axes[class_idx * 10 + i].axis("off")
            axes[class_idx * 10 + i].set_title(LABELS[class_idx], fontsize=8)
    plt.tight_layout()
    plt.show()


load_data()

# Convert np arrays to PyTorch tensors
X_train_tensor = (
    torch.tensor(X_train.transpose(0, 3, 1, 2), dtype=torch.float32) / 255.0
)
y_train_tensor = torch.tensor(y_train, dtype=torch.long)
X_test_tensor = torch.tensor(X_test.transpose(0, 3, 1, 2), dtype=torch.float32) / 255.0
y_test_tensor = torch.tensor(y_test, dtype=torch.long)

# Dataloaders
train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
test_dataset = TensorDataset(X_test_tensor, y_test_tensor)
train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)


# CNN model
class SimpleCNN(nn.Module):
    def __init__(self):
        super(SimpleCNN, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(128 * 4 * 4, 256)
        self.fc2 = nn.Linear(256, 10)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = self.pool(F.relu(self.conv3(x)))
        x = x.view(-1, 128 * 4 * 4)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x


def train_model(model, train_loader, num_epochs=10):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    start_epoch, train_losses = load_checkpoint(model, optimizer)

    print("Training started... (˶˃ ᵕ ˂˶)\n")

    for epoch in range(start_epoch, num_epochs):
        running_loss = 0.0
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

        avg_loss = running_loss / len(train_loader)
        train_losses.append(avg_loss)  # Append to restored loss history
        print(f"Epoch {epoch+1}/{num_epochs}, Loss: {avg_loss:.4f}")

        if (epoch + 1) % 5 == 0:
            save_checkpoint(model, optimizer, epoch + 1, train_losses)

    print("Training completed! ◝(ᵔ ᗜ ᵔ)◜ \n")
    return train_losses


def plot_training_loss(train_losses, num_epochs):
    plt.figure(figsize=(8, 5))
    plt.plot(
        range(1, len(train_losses) + 1),
        train_losses,
        marker="o",
        linestyle="-",
        color="blue",
    )
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.title("Training Loss Over Epochs")
    plt.grid()
    plt.show()
    plt.savefig("training_progress.png")
    plt.pause(3)
    plt.close()


def evaluate_model(model, test_loader):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()

    correct = 0
    total = 0
    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    accuracy = 100 * correct / total
    print(f"Test accuracy: {accuracy:.2f}%")


def main():
    model = SimpleCNN()

    user_choice = (
        input("Resume training from the last checkpoint? (y/n): ").strip().lower()
    )
    if user_choice == "n":
        print("Starting over from scratch...")
        if os.path.exists(CHECKPOINT_FILE):
            os.remove(CHECKPOINT_FILE)

    train_losses = train_model(model, train_loader, num_epochs=10)
    plot_training_loss(train_losses, num_epochs=len(train_losses))
    evaluate_model(model, test_loader)


main()
