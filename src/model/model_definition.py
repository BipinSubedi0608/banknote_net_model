import torch.nn as nn

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