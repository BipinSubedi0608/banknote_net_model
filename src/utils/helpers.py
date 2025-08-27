import torch

from model.model_definition import CurrencyClassifier
from utils.constants import (
    SAVED_MODELS_DIR,
    CLASSIFIER_MODEL_NAME,
    CURRENCY_LABEL_MAP
)


def get_device():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    return device


def save_model(best_model_state):
    path = SAVED_MODELS_DIR + CLASSIFIER_MODEL_NAME
    torch.save(best_model_state, path)


def load_model(device: torch.device):
    model_path = SAVED_MODELS_DIR + CLASSIFIER_MODEL_NAME
    model = CurrencyClassifier().to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    return model


def get_currency_from_label(label: int):
    for key, value in CURRENCY_LABEL_MAP.items():
        if value == label:
            return key
    return None