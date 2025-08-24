import torch
import numpy as np

from model.model_definition import CurrencyClassifier
from utils.constants import (
    PROCESSED_DATA_DIR,
    X_TRAIN_FILE,
    X_TEST_FILE,
    Y_TRAIN_FILE,
    Y_TEST_FILE,
    SAVED_MODELS_DIR,
    MODEL_NAME
)


def get_device():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    return device


def save_processed_data(X_train, X_test, y_train, y_test):
    np.save(PROCESSED_DATA_DIR + X_TRAIN_FILE, X_train)
    np.save(PROCESSED_DATA_DIR + X_TEST_FILE, X_test)
    np.save(PROCESSED_DATA_DIR + Y_TRAIN_FILE, y_train)
    np.save(PROCESSED_DATA_DIR + Y_TEST_FILE, y_test)


def load_processed_data():
    X_train = np.load(PROCESSED_DATA_DIR + X_TRAIN_FILE)
    X_test = np.load(PROCESSED_DATA_DIR + X_TEST_FILE)
    y_train = np.load(PROCESSED_DATA_DIR + Y_TRAIN_FILE)
    y_test = np.load(PROCESSED_DATA_DIR + Y_TEST_FILE)

    return X_train, X_test, y_train, y_test


def save_model(best_model_state):
    path = SAVED_MODELS_DIR + MODEL_NAME
    torch.save(best_model_state, path)


def load_model(input_dim, output_dim, device):
    model_path = SAVED_MODELS_DIR + MODEL_NAME
    model = CurrencyClassifier(input_dim, output_dim).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    return model