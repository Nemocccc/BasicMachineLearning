import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import datasets, transforms
import torch.optim as optim


class SimpleCNN(nn.Module):
    def __init__(self):
        super(SimpleCNN, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=16, kernel_size=3, stride=1, padding=1)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv2 = nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, stride=1, padding=1)
        self.fc1 = nn.Linear(32 * 7 * 7, 128)
        self.fc2 = nn.Linear(128, 10) #output layer

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = self.pool(x)
        x = F.relu(self.conv2(x))
        x = self.pool(x)
        x = x.view(-1, 32 * 7 * 7) #flattening
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x
    
model = SimpleCNN()
print(model)


# dataset pre-process workflow
transform = transforms.Compose([
    transforms.ToTensor(), #The image is converted to Tensor and normalized to the interval [0,1]
    transforms.Normalize((0.1307,), (0.3081,)) #standardize, base on the statistic infomation of MNIST
])

# download and load MNIST dataset.
train_dataset = datasets.MNIST(root='./handwriting_numerals_recognition/data', train=True, download=True, transform=transform)
test_dataset = datasets.MNIST(root='./handwriting_numerals_recognition/data', train=False, download=True, transform=transform)

# view the size of dataset
print("size of train_dataset:", len(train_dataset))
print("size of test_dataset:", len(test_dataset))

# build a DataLoader for batch training and testing.
batch_size = 64
train_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)  #shuffle: data randomness while training.
test_loader = torch.utils.data.DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=False)

#loss function and optimizer
loss_fn = nn.CrossEntropyLoss()  #交叉熵
optimizer = optim.Adam(model.parameters(), lr=0.001) # Adam optimizer, learning rate = 0.001

def train_model(model, dataloader, loss_fn, optimizer, device):
    model.train() #set model as training model
    running_loss = 0
    for inputs, labels in dataloader:
        inputs, labels = inputs.to(device), labels.to(device) # migrate data to GPU

        #forward propagation
        outputs = model(inputs)

        #loss calculate
        loss = loss_fn(outputs, labels)

        #back propagation and optimize
        optimizer.zero_grad() #
        loss.backward()#back propagation
        optimizer.step() #update weight

        running_loss += loss.item()

    return running_loss / len(dataloader) #return average loss

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model.to(device)

num_epochs = 5
for epoch in range(num_epochs):
    train_loss = train_model(model, train_loader, loss_fn, optimizer, device)
    print(f"Epoch {epoch+1}/{num_epochs}, Loss:{train_loss:.4f}")

def evaluate_model(model, dataloader, device):
    model.eval() #set model as evaluate mode
    correct = 0
    total = 0
    with torch.no_grad():
        for images, labels in dataloader:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

    return correct / total

test_accuracy = evaluate_model(model, test_loader, device)
print(f"Test Accuracy: {test_accuracy:.2%}")