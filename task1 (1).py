"""
Kevin Sani and Basil Reji
CS5330
Project 5 : Recognition using Deep Networks  
Task 1: A to D
"""
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
from torchvision.datasets import MNIST
from torchvision.transforms import ToTensor
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import torch.nn.functional as F
import matplotlib.pyplot as plt


# Task A: Display MNIST Digits
def display_mnist_digits():
    test_dataset = torchvision.datasets.MNIST(root='./data', train=False, download=True, transform=ToTensor())
    images = [test_dataset[i][0] for i in range(6)]
    labels = [test_dataset[i][1] for i in range(6)]
    
    fig, axs = plt.subplots(1, 6, figsize=(15, 2.5))
    for i, ax in enumerate(axs):
        ax.imshow(images[i].reshape(28, 28), cmap='gray')
        ax.set_title(f'Label: {labels[i]}')
        ax.axis('off')
    plt.show()

# Task B: Train and Test TheMNISTNetwork
class TheMNISTNetwork(nn.Module):
    def __init__(self):
        super(TheMNISTNetwork, self).__init__()
        self.network = nn.Sequential(
            nn.Conv2d(1, 10, kernel_size=5),
            nn.MaxPool2d(kernel_size=2),
            nn.ReLU(),
            nn.Conv2d(10, 20, kernel_size=5),
            nn.Dropout2d(0.5),
            nn.MaxPool2d(kernel_size=2),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(320, 50),
            nn.ReLU(),
            nn.Linear(50, 10),
            nn.LogSoftmax(dim=1)
        )
        
    def forward(self, x):
        return self.network(x)

def task_train_and_test_network():
    # Define parameters
    n_epochs = 5
    batch_size_train = 64
    batch_size_test = 1000

    # Data loading
    train_loader = DataLoader(MNIST('./data', train=True, download=True, transform=ToTensor()), batch_size=batch_size_train, shuffle=True)
    test_loader = DataLoader(MNIST('./data', train=False, download=True, transform=ToTensor()), batch_size=batch_size_test, shuffle=False)

    # Initialize the network and variables to store loss data
    network = TheMNISTNetwork()
    train_losses = []
    train_counter = []
    test_losses = []
    test_counter = []

    # Define the train function
    def train(epoch, network, train_loader, train_losses, train_counter):
        network.train()
        for i, (data, target) in enumerate(train_loader):
            output = network(data)
            loss = F.nll_loss(output, target)
            loss.backward()
            if i % 10 == 0:
                train_losses.append(loss.item())
                train_counter.append(i * len(data) + epoch * len(train_loader.dataset))
            torch.save(network.state_dict(), 'model.pth')

    # Define the test function
    def test(network, test_loader, test_losses, test_counter):
        network.eval()
        test_loss = 0
        correct = 0
        with torch.no_grad():
            for data, target in test_loader:
                output = network(data)
                test_loss += F.nll_loss(output, target, reduction='sum').item()
                pred = output.data.max(1, keepdim=True)[1]
                correct += pred.eq(target.data.view_as(pred)).sum()
        test_loss /= len(test_loader.dataset)
        test_losses.append(test_loss)
        test_counter.append(len(train_loader.dataset) * epoch)  
        print(f'\nTest set: Avg. loss: {test_loss:.4f}, Accuracy: {correct}/{len(test_loader.dataset)} ({100. * correct / len(test_loader.dataset):.0f}%)\n')
    
    # Training and testing loop
    for epoch in range(1, n_epochs + 1):
        train(epoch, network, train_loader,train_losses, train_counter)
        test(network, test_loader, test_losses, test_counter)

    print("Trained network saved to 'model.pth'.")
    # Plotting the training and testing progress
    plt.figure(figsize=(10, 6))
    plt.plot(train_counter, train_losses, color='blue')
    plt.scatter(test_counter, test_losses, color='red')  
    plt.legend(['Train Loss', 'Test Loss'], loc='upper right')
    plt.xlabel('Number of training examples seen')
    plt.ylabel('Negative log likelihood loss')
    plt.show()

# Main Menu as a list to choose from the process when running the program
def main():
    tasks = {
        "A": ("Display MNIST Digits", display_mnist_digits),
        "B": ("Train and Test TheMNISTNetwork and save the Model", task_train_and_test_network),
    }
    
    while True:
        print("\nAvailable Tasks:")
        for task, (description, _) in tasks.items():
            print(f" {task}: {description}")
        choice = input("Please choose a task or Q to quit: ").upper()
        
        if choice == 'Q':
            print("Exiting program.")
            break
        elif choice in tasks:
            print(f"\nExecuting {tasks[choice][0]}...\n")
            tasks[choice][1]()
        else:
            print("Invalid choice. Please try again.")

if __name__ == "__main__":
    main()