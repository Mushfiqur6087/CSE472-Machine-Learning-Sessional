import torch 
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

num_epochs = 5
num_classes = 10
batch_size = 100
learning_rate = 0.001

train_dataset = torchvision.datasets.CIFAR10(
    root='./data/',
    train=True,
    transform=transforms.ToTensor(),
    download=True
)

test_dataset = torchvision.datasets.CIFAR10(
    root='./data/',
    train=False,
    transform=transforms.ToTensor()
)

train_loader = torch.utils.data.DataLoader(
    dataset=train_dataset,
    batch_size=batch_size,
    shuffle=True
)

test_loader = torch.utils.data.DataLoader(
    dataset=test_dataset,
    batch_size=batch_size,
    shuffle=False
)


# ==========================================================
# TASK 1: Implement NiNBlock
# ==========================================================
class NiNBlock(nn.Module):
    """
    NiN Block structure:
    Conv(k×k) → ReLU
    Conv(1×1) → ReLU
    Conv(1×1) → ReLU

    Preserves spatial resolution.
    """

    def __init__(self, in_channels, out_channels, k=3):
        super(NiNBlock, self).__init__()

        padding = (k - 1) // 2  # keeps H×W unchanged

        self.conv1 = nn.Conv2d(in_channels, out_channels,
                               kernel_size=k, stride=1, padding=padding)
        self.relu1 = nn.ReLU()

        self.conv2 = nn.Conv2d(out_channels, out_channels,
                               kernel_size=1)
        self.relu2 = nn.ReLU()

        self.conv3 = nn.Conv2d(out_channels, out_channels,
                               kernel_size=1)
        self.relu3 = nn.ReLU()

    def forward(self, x):
        x = self.relu1(self.conv1(x))
        x = self.relu2(self.conv2(x))
        x = self.relu3(self.conv3(x))
        return x


# ==========================================================
# TASK 2: Implement Full NiN Model
# ==========================================================
class NiN3(nn.Module):

    """
    Architecture:
    NiNBlock(3→32)
    ↓ MaxPool

    NiNBlock(32→64)
    ↓ MaxPool

    NiNBlock(64→64)

    1×1 Conv (64→10)
    ↓ Global Average Pool
    """

    def __init__(self, num_classes=10):
        super(NiN3, self).__init__()

        # Block 1
        self.block1 = NiNBlock(3, 32)
        self.pool1 = nn.MaxPool2d(2, 2)

        # Block 2
        self.block2 = NiNBlock(32, 64)
        self.pool2 = nn.MaxPool2d(2, 2)

        # Block 3
        self.block3 = NiNBlock(64, 64)

        # Replace fully connected layer with:
        # 1×1 Conv classifier
        self.classifier = nn.Conv2d(64, num_classes, kernel_size=1)

        # Global Average Pooling
        self.gap = nn.AdaptiveAvgPool2d((1, 1))

    def forward(self, x):

        x = self.pool1(self.block1(x))
        x = self.pool2(self.block2(x))
        x = self.block3(x)

        x = self.classifier(x)
        x = self.gap(x)

        x = torch.flatten(x, 1)
        return x


# Replace demo model with NiN
model = NiN3(num_classes).to(device)


# ==========================================================
# TASK 3: Training (UNCHANGED FROM DEMO)
# ==========================================================

# Loss and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

# Train the model
total_step = len(train_loader)
for epoch in range(num_epochs):
    for i, (images, labels) in enumerate(train_loader):

        images = images.to(device)
        labels = labels.to(device)

        outputs = model(images)
        loss = criterion(outputs, labels)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if (i+1) % 100 == 0:
            print('Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}'
                  .format(epoch+1, num_epochs,
                          i+1, total_step, loss.item()))


# ==========================================================
# Testing (UNCHANGED)
# ==========================================================

# Test the model
model.eval()  # eval mode (batchnorm uses moving mean/variance instead of mini-batch mean/variance)

with torch.no_grad():
    correct = 0
    total = 0
    for images, labels in test_loader:
        images = images.to(device)
        labels = labels.to(device)

        outputs = model(images)
        _, predicted = torch.max(outputs, 1)

        total += labels.size(0)
        correct += (predicted == labels).sum().item()

    print('Test Accuracy of the model on the 10000 test images: {} %'.format(100 * correct / total))

# Save the model checkpoint
torch.save(model.state_dict(), 'model.ckpt')
