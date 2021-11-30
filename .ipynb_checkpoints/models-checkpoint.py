import torch
import torch.nn as nn
import torch.nn.functional as F


class MnistNet(nn.Module):
    """Simple CNN adapted from Pytorch's 'Basic MNIST Example'."""
    def __init__(self):
        super(MnistNet, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, 3, 1)
        self.conv2 = nn.Conv2d(32, 64, 3, 1)
        self.pool = nn.MaxPool2d(2, 2)
        self.dropout1 = nn.Dropout2d(0.25)
        self.dropout2 = nn.Dropout2d(0.5)
        self.fc1 = nn.Linear(9216, 128)
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        """Compute forward pass."""
        x = F.relu(self.conv1(x))
        x = self.pool(F.relu(self.conv2(x)))
        x = self.dropout1(x)
        x = torch.flatten(x, 1)
        x = F.relu(self.fc1(x))
        x = self.dropout2(x)
        x = self.fc2(x)
        return x
    
class CifarNet(nn.Module):
    def __init__(self):
        super(CifarNet, self).__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 16 * 5 * 5)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x
    
class ToxNet(nn.Module):
    def __init__(self):
        super(ToxNet, self).__init__()
        self.embedding = nn.Embedding(61, 56)
        self.lstm1 = nn.LSTM(56, 48, bidirectional=True, batch_first=True)
        self.lstm2 = nn.LSTM(96, 24, bidirectional=True, batch_first=True)
        self.linear = nn.Linear(24 * 4, 24)
        self.dropout = nn.Dropout()
        self.fc1 = nn.Linear(24, 12)
        self.fc2 = nn.Linear(12, 1)

    def forward(self, x): 
        x = self.embedding(x.long()) 
        h, _ = self.lstm1(x.mean(2)) 
        h, _ = self.lstm2(h) 
        
        h_avg = torch.mean(h, 1)
        h_max, _ = torch.max(h, 1)
        h = torch.cat((h_avg, h_max), 1)
        
        h = F.relu(self.linear(h))
        h = self.dropout(h)
        h = F.relu(self.fc1(h))
        h = self.fc2(h)
        return h.view(-1)
