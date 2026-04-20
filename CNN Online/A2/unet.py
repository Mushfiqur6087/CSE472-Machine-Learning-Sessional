import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import torchvision
from torch.utils.data import DataLoader


class TinyUNet(nn.Module):
    def __init__(self, in_channels=1, out_channels=1):
        super(TinyUNet, self).__init__()
        # Implement the Unet architecture


    def forward(self, x):
        # Implement the forward pass
        return


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Pad(2),
    transforms.Normalize((0.1307,), (0.3081,))
])

trainset = torchvision.datasets.MNIST(root='./data', train=True, download=True, transform=transform)
from torch.utils.data import Subset

small_trainset = Subset(trainset, range(10000))
trainloader = DataLoader(small_trainset, batch_size=512, shuffle=True)

model = TinyUNet().to(device)
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

print("Starting Training...")
for epoch in range(2):
    running_loss = 0.0
    for i, data in enumerate(trainloader):
        inputs, _ = data
        inputs = inputs.to(device)
        # Implement the Training Step
        # 1. Clear the gradients from the previous iteration

        # 2. Forward pass: Compute the predicted reconstruction

        # 3. Compute loss, perform backpropagation, and update weights


        running_loss += loss.item()
        if i % 10 == 9:
            print(f'[Epoch {epoch + 1}, Batch {i + 1}] Loss: {running_loss / 100:.4f}')
            running_loss = 0.0

print("Finished Training.")