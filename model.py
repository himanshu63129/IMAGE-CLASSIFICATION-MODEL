import torch
import torch.nn as nn
import torch.nn.functional as F

class SimpleCNN(nn.Module):
    def __init__(self):
        super(SimpleCNN, self).__init__()
        # Convolutional layers
        # Input: 3x32x32
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        
        # Max pooling
        self.pool = nn.MaxPool2d(2, 2)
        
        # Fully connected layers
        # After 3 max pools of 2x2, 32x32 -> 16x16 -> 8x8 -> 4x4
        self.fc1 = nn.Linear(128 * 4 * 4, 512)
        self.fc2 = nn.Linear(512, 10) # 10 classes in CIFAR-10
        
        self.dropout = nn.Dropout(0.25)

    def forward(self, x):
        # Apply conv layers with ReLU and MaxPool
        x = self.pool(F.relu(self.conv1(x))) # Output: 32x16x16
        x = self.pool(F.relu(self.conv2(x))) # Output: 64x8x8
        x = self.pool(F.relu(self.conv3(x))) # Output: 128x4x4
        
        # Flatten the feature maps
        x = x.view(-1, 128 * 4 * 4)
        
        # Apply fully connected layers
        x = self.dropout(x)
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        
        return x

if __name__ == "__main__":
    # Test with a dummy input
    model = SimpleCNN()
    dummy_input = torch.randn(1, 3, 32, 32)
    output = model(dummy_input)
    print(f"Model output shape: {output.shape}")
