import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam

class BrainTumorCNN(nn.Module):
    def __init__(self):
        super(BrainTumorCNN, self).__init__()
        # Define the layers of the model
        self.conv1 = nn.Conv2d(3, 32, kernel_size=4, padding='same')  # Assuming input images are RGB
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.dropout1 = nn.Dropout(0.5)
        self.conv2 = nn.Conv2d(32, 32, kernel_size=4, padding='same')
        self.dropout2 = nn.Dropout(0.3)
        self.conv3 = nn.Conv2d(32, 64, kernel_size=4, padding='same')
        self.dropout3 = nn.Dropout(0.3)
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(64 * 32 * 32, 128)  # Adjust the size according to your final feature map size
        self.dropout4 = nn.Dropout(0.15)
        self.fc2 = nn.Linear(128, 1)

    def forward(self, x):
        # Define the forward pass
        x = self.pool(F.relu(self.conv1(x)))
        x = self.dropout1(x)
        x = self.pool(F.relu(self.conv2(x)))
        x = self.dropout2(x)
        x = self.pool(F.relu(self.conv3(x)))
        x = self.dropout3(x)
        x = self.flatten(x)
        x = F.relu(self.fc1(x))
        x = self.dropout4(x)
        x = torch.sigmoid(self.fc2(x))
        return x

def initialize_model():
    model = BrainTumorCNN()
    # Setup the optimizer and loss function
    optimizer = Adam(model.parameters(), lr=0.001)
    loss_fn = nn.BCELoss()  # Binary Cross-Entropy Loss for binary classification
    return model, optimizer, loss_fn

# Usage example
model, optimizer, loss_fn = initialize_model()
