

import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


class FireModule(nn.Module):
    def __init__(self, in_channels, squeeze_channels, expand_channels):
        super(FireModule, self).__init__()

        # TODO: Squeeze layer
        self.squeeze_blocks = nn.Sequential(
            nn.Conv2d(in_channels,squeeze_channels, kernel_size=1, padding=0, stride=1),
            nn.ReLU()
            )
        
        # TODO: Expand layers
        self.conv1=nn.Sequential(
            nn.Conv2d(squeeze_channels,expand_channels, kernel_size=1, padding=0, stride=1),
            nn.ReLU()
            )
        self.conv2=nn.Sequential(
            nn.Conv2d(squeeze_channels,expand_channels, kernel_size=3, padding=1, stride=1),
            nn.ReLU()
            )
        self.expand_blocks=torch.concat((self.conv1,self.conv2),dim=1)
        
        
    

    def forward(self, x):
        # TODO: Implement forward pass for FireModule
        x = self.squeeze_blocks(x)
        x = self.expand_blocks(x)
        return x


class SqueezeLite(nn.Module):
    def __init__(self):
        super(SqueezeLite, self).__init__()
        self.blocks = nn.Sequential(
        nn.Conv2d(1,32, kernel_size=1, padding=0, stride=1),
        nn.ReLU(),
        FireModule(32,8,16),
        nn.MaxPool2d(kernel_size=1,stride=2),
        FireModule(32,16,32),
        nn.Conv2d(64,10,kernel_size=1,stride=1)
        )
        self.globalPool = nn.AdaptiveAvgPool2d((1,1))

        
        # TODO :  SqueezeNet-like architecture for MNIST
        

    def forward(self, x):
       x= self.blocks(x)
       x = self.globalPool(x)
       return x
 


def train_model():

    # MNIST Transform (single channel)
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])

    trainset = torchvision.datasets.MNIST(
        root='./data',
        train=True,
        download=True,
        transform=transform
    )

    trainloader = torch.utils.data.DataLoader(
        trainset,
        batch_size=100,
        shuffle=True
    )

    testset = torchvision.datasets.MNIST(
        root='./data',
        train=False,
        download=True,
        transform=transform
    )

    testloader = torch.utils.data.DataLoader(
        testset,
        batch_size=100,
        shuffle=False
    )

    model = SqueezeLite().to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    # Parameter count
    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total Trainable Parameters: {total_params}")

    print("Starting Training...")

    model.train()
    for epoch in range(5):
        running_loss = 0.0

        for i, data in enumerate(trainloader, 0):
            inputs, labels = data[0].to(device), data[1].to(device)

            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

            if i % 100 == 99:
                print(f'[Epoch {epoch+1}, Batch {i+1}] Loss: {running_loss/100:.3f}')
                running_loss = 0.0

    # Evaluation
    model.eval()
    correct = 0
    total = 0

    with torch.no_grad():
        for data in testloader:
            images, labels = data[0].to(device), data[1].to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)

            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    print(f'Final Test Accuracy: {100 * correct / total:.2f}%')


if __name__ == "__main__":
    train_model()

