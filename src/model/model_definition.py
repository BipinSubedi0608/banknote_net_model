import torch.nn as nn
import pandas as pd
import numpy as np
import joblib
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split

from utils.constants import (
    SAVED_MODELS_DIR,
    PROCESSED_DATA_DIR,
    X_TRAIN_FILE,
    X_VAL_FILE,
    X_TEST_FILE,
    Y_TRAIN_FILE,
    Y_VAL_FILE,
    Y_TEST_FILE,
    SCALER_NAME
)

class CurrencyClassifier(nn.Module):
    def __init__(self, input_size, output_size, dropout_rate):
        super().__init__()

        self.fc1 = nn.Linear(input_size, 192)
        self.bn1 = nn.BatchNorm1d(192)

        self.fc2 = nn.Linear(192, 96)
        self.bn2 = nn.BatchNorm1d(96)

        self.fc3 = nn.Linear(96, 48)
        self.bn3 = nn.BatchNorm1d(48)

        self.fc4 = nn.Linear(48, output_size)

        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout_rate)

    def forward(self, x):
        x = self.dropout(self.relu(self.bn1(self.fc1(x))))
        x = self.relu(self.bn2(self.fc2(x)))
        x = self.dropout(self.relu(self.fc3(x)))
        x = self.fc4(x)

        return x
    

def preprocess_data(df: pd.DataFrame, scaler: StandardScaler):
    df = df.dropna()
    df = df.drop_duplicates()
    
    y = df['Currency']
    currency_encoder = LabelEncoder()
    y = currency_encoder.fit_transform(y)

    X = df.drop(columns=['Currency', 'Denomination'])
    X = scaler.fit_transform(X)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=420, stratify=y
    )

    X_train, X_val, y_train, y_val = train_test_split(
        X_train, y_train, test_size=0.1, random_state=420, stratify=y_train
    )

    return X_train, X_val, X_test, y_train, y_val, y_test


def save_scaler(scaler: StandardScaler):
    path = SAVED_MODELS_DIR + SCALER_NAME
    joblib.dump(scaler, path)


def save_processed_data(X_train, X_val, X_test, y_train, y_val, y_test):
    np.save(PROCESSED_DATA_DIR + X_TRAIN_FILE, X_train)
    np.save(PROCESSED_DATA_DIR + X_VAL_FILE, X_val)
    np.save(PROCESSED_DATA_DIR + X_TEST_FILE, X_test)
    np.save(PROCESSED_DATA_DIR + Y_TRAIN_FILE, y_train)
    np.save(PROCESSED_DATA_DIR + Y_VAL_FILE, y_val)
    np.save(PROCESSED_DATA_DIR + Y_TEST_FILE, y_test)