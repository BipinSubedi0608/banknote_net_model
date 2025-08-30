import torch
import torch.optim as optim
from torch import FloatTensor, LongTensor
from torch.utils.data import TensorDataset, DataLoader
from torch.nn import CrossEntropyLoss
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, recall_score, precision_score, f1_score

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

    return train_loader, test_loader


def initialise_model(device):
    model = CurrencyClassifier().to(device)
    criterion = CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

    return model, criterion, optimizer


def train_and_validate(model, criterion, optimizer, train_loader, test_loader, device):
    train_metrics = {
        "losses": [],
        "accuracies": []
    }
    test_metrics = {
        "losses": [],
        "accuracies": [],
        "conf_matrices": [],
        "recalls": [],
        "precisions": [],
        "f1s": []
    }
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
        train_metrics["losses"].append(train_loss)
        train_metrics["accuracies"].append(train_acc)

        # Validation
        model.eval()
        test_running_loss, test_correct, test_total = 0.0, 0, 0
        all_preds, all_targets = [], []
        with torch.no_grad():
            for X_batch, y_batch in test_loader:
                X_batch, y_batch = X_batch.to(device), y_batch.to(device)
                outputs = model(X_batch)
                loss = criterion(outputs, y_batch)
                test_running_loss += loss.item() * X_batch.size(0)
                _, preds = torch.max(outputs, 1)
                test_correct += (preds == y_batch).sum().item()
                test_total += y_batch.size(0)
                all_preds.extend(preds.cpu().numpy())
                all_targets.extend(y_batch.cpu().numpy())
        test_loss = test_running_loss / test_total
        test_acc = test_correct / test_total
        test_metrics["losses"].append(test_loss)
        test_metrics["accuracies"].append(test_acc)

        # Additional metrics for test set
        conf_matrix = confusion_matrix(all_targets, all_preds)
        recall = recall_score(all_targets, all_preds, average='macro')
        precision = precision_score(all_targets, all_preds, average='macro')
        f1 = f1_score(all_targets, all_preds, average='macro')
        test_metrics["conf_matrices"].append(conf_matrix)
        test_metrics["recalls"].append(recall)
        test_metrics["precisions"].append(precision)
        test_metrics["f1s"].append(f1)

        print(f"Epoch {epoch+1}/{EPOCHS} | Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.4f} | Test Loss: {test_loss:.4f} | Test Acc: {test_acc:.4f}")
        print(f"Confusion Matrix:\n{conf_matrix}")
        print(f"Recall: {recall:.4f} | Precision: {precision:.4f} | F1: {f1:.4f}")

        # Save best model
        if test_loss < best_test_loss:
            best_test_loss = test_loss
            best_model_state = model.state_dict()

    return {"train": train_metrics, "test": test_metrics, "best_model_state": best_model_state}


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
