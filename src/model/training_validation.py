import numpy as np
import torch
import torch.optim as optim
from torch import FloatTensor, LongTensor
from torch.utils.data import TensorDataset, DataLoader
from torch.nn import CrossEntropyLoss
import matplotlib.pyplot as plt

from model.model_definition import CurrencyClassifier

# Constants
BATCH_SIZE = 128
EPOCHS = 20
LEARNING_RATE = 0.01


def create_dataloaders(X_train, X_test, y_train, y_test):
    X_train_tensor = FloatTensor(X_train)
    X_test_tensor = FloatTensor(X_test)
    y_train_tensor = LongTensor(y_train)
    y_test_tensor = LongTensor(y_test)

    train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
    test_dataset = TensorDataset(X_test_tensor, y_test_tensor)

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

    input_dim = X_train_tensor.shape[1]
    output_dim = len(np.unique(y_train))

    return train_loader, test_loader, input_dim, output_dim


def initialize_model(input_dim, output_dim, device):
    model = CurrencyClassifier(input_dim, output_dim).to(device)
    criterion = CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

    return model, criterion, optimizer


def train_and_validate(model, criterion, optimizer, train_loader, test_loader, device):
    train_losses, test_losses = [], []
    train_accuracies, test_accuracies = [], []
    best_test_loss = float('inf')
    best_model_state = None

    for epoch in range(EPOCHS):
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
        test_running_loss, test_correct, test_total = 0.0, 0, 0
        with torch.no_grad():
            for X_batch, y_batch in test_loader:
                X_batch, y_batch = X_batch.to(device), y_batch.to(device)
                outputs = model(X_batch)
                loss = criterion(outputs, y_batch)
                test_running_loss += loss.item() * X_batch.size(0)
                _, preds = torch.max(outputs, 1)
                test_correct += (preds == y_batch).sum().item()
                test_total += y_batch.size(0)
        test_loss = test_running_loss / test_total
        test_acc = test_correct / test_total
        test_losses.append(test_loss)
        test_accuracies.append(test_acc)

        print(f"Epoch {epoch+1}/{EPOCHS} | Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.4f} | Test Loss: {test_loss:.4f} | Test Acc: {test_acc:.4f}")

        # Save best model
        if test_loss < best_test_loss:
            best_test_loss = test_loss
            best_model_state = model.state_dict()

    return train_losses, test_losses, train_accuracies, test_accuracies, best_model_state


def plot_curves(train_losses, test_losses, train_accuracies, test_accuracies):
    plt.figure(figsize=(12,5))
    plt.subplot(1,2,1)
    plt.plot(train_losses, label='Train Loss')
    plt.plot(test_losses, label='Test Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.title('Loss Curve')

    plt.subplot(1,2,2)
    plt.plot(train_accuracies, label='Train Acc')
    plt.plot(test_accuracies, label='Test Acc')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.title('Accuracy Curve')
    plt.show()
