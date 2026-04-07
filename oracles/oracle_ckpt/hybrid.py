import torch
import torch.nn as nn
import torch.nn.functional as F

class HybridNet(nn.Module):
    def __init__(self):
        super(HybridNet, self).__init__()
        
        # Convolutional layers
        self.conv1 = nn.Conv1d(in_channels=4, out_channels=256, kernel_size=13, stride=1)
        self.relu = nn.ReLU()
        self.maxpool = nn.MaxPool1d(kernel_size=2, stride=2)
        self.dropout1 = nn.Dropout(p=0.3)
        self.batchnorm1 = nn.BatchNorm1d(256)

        self.conv2 = nn.Conv1d(in_channels=256, out_channels=256, kernel_size=13, stride=1)
        self.dropout2 = nn.Dropout(p=0.3)
        self.batchnorm2 = nn.BatchNorm1d(256)

        self.conv3 = nn.Conv1d(in_channels=256, out_channels=256, kernel_size=13, stride=1)
        self.dropout3 = nn.Dropout(p=0.3)
        self.batchnorm3 = nn.BatchNorm1d(256)
        

        # LSTM layer
        self.lstm1 = nn.LSTM(input_size=256, hidden_size=128, batch_first=True)
        self.lstm2 = nn.LSTM(input_size=128, hidden_size=128, batch_first=True)

        # Fully connected layers
        self.fc1 = nn.Linear(in_features=128, out_features=64)
        self.dropout4 = nn.Dropout(p=0.2)
        self.batchnorm4 = nn.BatchNorm1d(64)
        self.fc2 = nn.Linear(in_features=64, out_features=1)

    def forward(self, x):
        # Convolutional layers
        x = x.permute(0,2,1)
        x = self.conv1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        x = self.dropout1(x)
        x = self.batchnorm1(x)

        x = self.conv2(x)
        x = self.relu(x)
        x = self.maxpool(x)
        x = self.dropout2(x)
        x = self.batchnorm2(x)

        x = self.conv3(x)
        x = self.relu(x)
        x = self.maxpool(x)
        x = self.dropout3(x)
        x = self.batchnorm3(x)
        
        # x = self.conv5(x)
        # x = self.relu(x)
        # x = self.maxpool(x)
        # x = self.dropout5(x)
        # x = self.batchnorm5(x)

        # Preparing for LSTM
        x = x.permute(0, 2, 1)  # (batch_size, seq_len, features)

        # LSTM layer
        x, _ = self.lstm1(x)
        x, _ = self.lstm2(x)

        # Taking the last time step output for the fully connected layers
        x = x[:, -1, :]

        # Fully connected layers
        x = self.fc1(x)
        x = self.relu(x)
        x = self.dropout4(x)
        x = self.batchnorm4(x)

        x = self.fc2(x)

        return x