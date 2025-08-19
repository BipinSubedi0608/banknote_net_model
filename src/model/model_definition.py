import torch.nn as nn
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split


# Constants
DROPOUT_RATE = 0.3

class CurrencyClassifier(nn.Module):
    def __init__(self, input_size, output_size):
        super().__init__()

        self.fc1 = nn.Linear(input_size, 192)
        self.bn1 = nn.BatchNorm1d(192)

        self.fc2 = nn.Linear(192, 96)
        self.bn2 = nn.BatchNorm1d(96)

        self.fc3 = nn.Linear(96, 48)
        self.bn3 = nn.BatchNorm1d(48)

        self.fc4 = nn.Linear(48, output_size)

        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(DROPOUT_RATE)

    def forward(self, x):
        x = self.dropout(self.relu(self.bn1(self.fc1(x))))
        x = self.relu(self.bn2(self.fc2(x)))
        x = self.dropout(self.relu(self.fc3(x)))
        x = self.fc4(x)

        return x
    

def preprocess_data(df: pd.DataFrame):
    df = df.dropna()
    df = df.drop_duplicates()
    
    y = df['Currency']
    currency_encoder = LabelEncoder()
    y = currency_encoder.fit_transform(y)

    X = df.drop(columns=['Currency', 'Denomination'])
    scaler = StandardScaler()
    X = scaler.fit_transform(X)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, random_state=420, stratify=y
    )

    return X_train, X_test, y_train, y_test
