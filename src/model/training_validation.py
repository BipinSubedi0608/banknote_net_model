import numpy as np
import torch
import torch.optim as optim
from torch import FloatTensor, LongTensor
from torch.utils.data import TensorDataset, DataLoader
from torch.nn import CrossEntropyLoss
import matplotlib.pyplot as plt

from model.model_definition import CurrencyClassifier
from utils.helpers import get_device
from utils.constants import (
    PROCESSED_DATA_DIR,
    SAVED_MODELS_DIR,
    MODEL_NAME,
    X_TRAIN_FILE,
    X_VAL_FILE,
    Y_TRAIN_FILE,
    Y_VAL_FILE
)

# Constants
BATCH_SIZE = 128
EPOCHS = 20
LEARNING_RATE = 0.01
DROPOUT_RATE = 0.3


def load_data():
    X_train = np.load(PROCESSED_DATA_DIR + X_TRAIN_FILE)
    X_val = np.load(PROCESSED_DATA_DIR + X_VAL_FILE)
    y_train = np.load(PROCESSED_DATA_DIR + Y_TRAIN_FILE)
    y_val = np.load(PROCESSED_DATA_DIR + Y_VAL_FILE)

    return X_train, X_val, y_train, y_val


def create_dataloaders(X_train, X_val, y_train, y_val, batch_size):
    X_train_tensor = FloatTensor(X_train)
    X_val_tensor = FloatTensor(X_val)
    y_train_tensor = LongTensor(y_train)
    y_val_tensor = LongTensor(y_val)

    train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
    val_dataset = TensorDataset(X_val_tensor, y_val_tensor)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    input_dim = X_train_tensor.shape[1]
    output_dim = len(np.unique(y_train))

    return train_loader, val_loader, input_dim, output_dim


def initialize_model(input_dim, output_dim, dropout_rate, device):
    model = CurrencyClassifier(input_dim, output_dim, dropout_rate).to(device)
    criterion = CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

    return model, criterion, optimizer


def train_and_validate(model, criterion, optimizer, train_loader, val_loader, device, epochs):
    train_losses, val_losses = [], []
    train_accuracies, val_accuracies = [], []
    best_val_loss = float('inf')
    best_model_state = None

    for epoch in range(epochs):
        model.train()
        running_loss, correct, total = 0.0, 0, 0
        for X_batch, y_batch in train_loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            optimizer.zero_grad()
            outputs = model(X_batch)
            loss = criterion(outputs, y_batch)
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * X_batch.size(0)
            _, preds = torch.max(outputs, 1)
            correct += (preds == y_batch).sum().item()
            total += y_batch.size(0)
        train_loss = running_loss / total
        train_acc = correct / total
        train_losses.append(train_loss)
        train_accuracies.append(train_acc)

        # Validation
        model.eval()
        val_running_loss, val_correct, val_total = 0.0, 0, 0
        with torch.no_grad():
            for X_batch, y_batch in val_loader:
                X_batch, y_batch = X_batch.to(device), y_batch.to(device)
                outputs = model(X_batch)
                loss = criterion(outputs, y_batch)
                val_running_loss += loss.item() * X_batch.size(0)
                _, preds = torch.max(outputs, 1)
                val_correct += (preds == y_batch).sum().item()
                val_total += y_batch.size(0)
        val_loss = val_running_loss / val_total
        val_acc = val_correct / val_total
        val_losses.append(val_loss)
        val_accuracies.append(val_acc)

        print(f"Epoch {epoch+1}/{epochs} | Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.4f} | Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.4f}")

        # Save best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_model_state = model.state_dict()

    return train_losses, val_losses, train_accuracies, val_accuracies, best_model_state


def plot_curves(train_losses, val_losses, train_accuracies, val_accuracies):
    plt.figure(figsize=(12,5))
    plt.subplot(1,2,1)
    plt.plot(train_losses, label='Train Loss')
    plt.plot(val_losses, label='Val Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.title('Loss Curve')

    plt.subplot(1,2,2)
    plt.plot(train_accuracies, label='Train Acc')
    plt.plot(val_accuracies, label='Val Acc')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.title('Accuracy Curve')
    plt.show()


def save_model(best_model_state):
    path = SAVED_MODELS_DIR + MODEL_NAME
    torch.save(best_model_state, path)


def main():
    device = get_device()
    X_train, X_val, y_train, y_val = load_data()
    train_loader, val_loader, input_dim, output_dim = create_dataloaders(X_train, X_val, y_train, y_val, BATCH_SIZE)
    model, criterion, optimizer = initialize_model(input_dim, output_dim, DROPOUT_RATE, device)
    train_losses, val_losses, train_accuracies, val_accuracies, best_model_state = train_and_validate(
        model, criterion, optimizer, train_loader, val_loader, device, EPOCHS
    )
    plot_curves(train_losses, val_losses, train_accuracies, val_accuracies)
    save_model(best_model_state)


if __name__ == "__main__":
    main()