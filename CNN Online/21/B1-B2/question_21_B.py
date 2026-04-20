import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import torch.nn as nn
import torch.optim as optim
import numpy as np
import random


from torch.optim.optimizer import Optimizer

class TensorAdaptiveSGD(Optimizer):
    def __init__(self, params, lr, weight_decay=0.0, adapt_const=0.1):

        defaults = dict(lr=lr, weight_decay=weight_decay, adapt_const=adapt_const)
        super().__init__(params, defaults)

    @torch.no_grad()
    def step(self):

        for group in self.param_groups:
            lr = group['lr']
            weight_decay = group['weight_decay']
            adapt_const = group['adapt_const']

            for param in group['params']:
                if param.grad is None:
                    continue
                
                # TODO 1: Compute mean absolute gradient as a tensor
                

                # TODO 2: Compute adaptive learning rate as a tensor
         

                # TODO 3: Update the parameter using weighted SGD



def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


set_seed(42)



# Image transformations
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])


# Load dataset
dataset = datasets.ImageFolder(
    root="images/",
    transform=transform
)

# DataLoader
dataloader = DataLoader(
    dataset,
    batch_size=30,
    shuffle=True,
    num_workers=0
)

images, labels = next(iter(dataloader))

print("Batch image tensor shape:", images.shape)
print("Batch labels tensor shape:", labels.shape)

# Number of classes
num_classes = len(dataset.classes)


class InceptionBlock(nn.Module):
    def __init__(self, in_channels):
        super(InceptionBlock, self).__init__()


        # TODO: define 1×1 conv
        
       
        # TODO: define 1×1 conv 
        # TODO: define 3×3 conv 

       
        # TODO: define 1×1 conv
        # TODO: define 5×5 conv 

        # TODO: define maxpool 
        # TODO: define 1×1 conv 

    
    def forward(self, x):
        # TODO: branch 1 forward


        # TODO: branch 2 forward


        # TODO: branch 3 forward


        # TODO: branch 4 forward


        # TODO: Final output from inception block , 
        # You can use torch.cat to concatenate along channel dimension
        # Syntax: torch.cat([tensor1, tensor2, ...], dim=channel_dimension)


        return x


class MiniInceptionNet(nn.Module):
    def __init__(self):
        super(MiniInceptionNet, self).__init__()


        # TODO: Initial stem convolutional layer
        

        # TODO: First Inception Block 
        

        
        # TODO: Downsampling convolutional layer
        

       

        # TODO: Global average pooling and final FC layer
        

    def forward(self, x):
        
        # TODO : Stem convolutional layer
        

      
        # TODO : First Inception Block
        

        # TODO: Downsampling convolutional layer

        # TODO: Global average pooling and final FC layer
        

        return x
    


device = torch.device("cpu")

model = MiniInceptionNet().to(device)

criterion = nn.BCEWithLogitsLoss()
optimizer = TensorAdaptiveSGD(model.parameters(), lr=0.1, weight_decay=0.01, adapt_const=0.5)

num_epochs = 10

print(f"\nTraining Custom ResNet with {sum(p.numel() for p in model.parameters())} parameters")
print(f"Device: {device}\n")

for epoch in range(num_epochs):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    for images, labels in dataloader:
        images = images.to(device)
        labels = labels.float().to(device)

        # Forward pass
        outputs = model(images).squeeze()
        loss = criterion(outputs, labels)

        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # Statistics
        running_loss += loss.item() * images.size(0)
        
        predicted = (torch.sigmoid(outputs) > 0.5).long()
        total += labels.size(0)
        correct += (predicted == labels.long()).sum().item()

    epoch_loss = running_loss / total
    epoch_acc = 100 * correct / total

    print(f"Epoch [{epoch+1}/{num_epochs}] "
          f"Loss: {epoch_loss:.4f} | Accuracy: {epoch_acc:.2f}%")