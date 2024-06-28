"""
Kevin Sani and Basil Reji
CS5330
Project 5 : Recognition using Deep Networks  
Task 4
"""
import torch
import torch.nn as nn
from torchvision.datasets import FashionMNIST
from torchvision.transforms import Compose, ToTensor, Normalize
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import time

#load the custom model
class CustomMNISTNetwork(nn.Module):
    def __init__(self, num_filters=10, kernel_size=5, dropout_rate=0.5):
        super(CustomMNISTNetwork, self).__init__()
        self.network = nn.Sequential(
            nn.Conv2d(1, num_filters, kernel_size=kernel_size, padding=kernel_size//2),  
            nn.MaxPool2d(kernel_size=2),
            nn.ReLU(),
            nn.Conv2d(num_filters, 20, kernel_size=5, padding=2),  
            nn.Dropout2d(dropout_rate),
            nn.MaxPool2d(kernel_size=2),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(20 * 7 * 7, 50),  
            nn.ReLU(),
            nn.Linear(50, 10),
            nn.LogSoftmax(dim=1)
        )

    def forward(self, x):
        return self.network(x)
# train and test the data with multiple configs
def train_and_test(model, train_loader, test_loader, epochs=5):
    criterion = nn.CrossEntropyLoss()
    train_losses, test_losses = [], []
    
    for epoch in range(epochs):
        model.train()
        total_loss = 0
        for data, target in train_loader:
            output = model(data)
            loss = criterion(output, target)
            loss.backward()
            total_loss += loss.item()
        avg_train_loss = total_loss / len(train_loader)
        train_losses.append(avg_train_loss)

        model.eval()
        total_loss = 0
        with torch.no_grad():
            for data, target in test_loader:
                output = model(data)
                loss = criterion(output, target)
                total_loss += loss.item()
        avg_test_loss = total_loss / len(test_loader)
        test_losses.append(avg_test_loss)
        
        print(f'Epoch {epoch+1}/{epochs}, Train Loss: {avg_train_loss:.4f}, Test Loss: {avg_test_loss:.4f}')
    
    return train_losses, test_losses

#looped experiment to run the results
def run_experiment(num_filters, kernel_size, batch_size, dropout_rate):
    start_time = time.time()
    transform = Compose([ToTensor(), Normalize((0.5,), (0.5,))])
    train_set = FashionMNIST(root='./data', train=True, download=True, transform=transform)
    test_set = FashionMNIST(root='./data', train=False, download=True, transform=transform)
    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False)

    model = CustomMNISTNetwork(num_filters=num_filters, kernel_size=kernel_size, dropout_rate=dropout_rate)
    train_losses, test_losses = train_and_test(model, train_loader, test_loader, epochs=5)
    end_time = time.time()  # End timer
    print(f"Experiment with Filters={num_filters}, Kernel={kernel_size}, Batch={batch_size}, Dropout={dropout_rate} completed in {end_time - start_time:.2f} seconds.")

    plt.figure(figsize=(10, 5))
    plt.plot(train_losses, label='Train Loss')
    plt.plot(test_losses, label='Test Loss')
    plt.title(f'Losses: Filters={num_filters}, Kernel={kernel_size}, Batch={batch_size}, Dropout={dropout_rate}')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.show()

# Parameters to vary
num_filters_options = [10, 20]
kernel_size_options = [3, 5]
batch_size_options = [64, 128]
dropout_rate_options = [0.3, 0.5]

# Running the experiment across different configurations
for num_filters in num_filters_options:
    for kernel_size in kernel_size_options:
        for batch_size in batch_size_options:
            for dropout_rate in dropout_rate_options:
                print(f'Running experiment with {num_filters} filters, kernel size {kernel_size}, batch size {batch_size}, dropout rate {dropout_rate}')
                run_experiment(num_filters, kernel_size, batch_size, dropout_rate)
