import torch
from torchvision import datasets, transforms

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