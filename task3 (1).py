"""
Kevin Sani and Basil Reji
CS5330
Project 5 : Recognition using Deep Networks  
Task 3
"""
import torch
import torch.nn.functional as F
import torch.nn as nn
import torchvision
from torchvision.datasets import ImageFolder
from torchvision.transforms import Compose, ToTensor, Normalize
from torch.utils.data import DataLoader
from task1 import TheMNISTNetwork 
import matplotlib.pyplot as plt

# greek data set transform
class GreekTransform:
    def __init__(self):
        pass
    def __call__(self, x):
        x = torchvision.transforms.functional.rgb_to_grayscale( x )
        x = torchvision.transforms.functional.affine( x, 0, (0,0), 36/128, 0 )
        x = torchvision.transforms.functional.center_crop( x, (28, 28) )
        return torchvision.transforms.functional.invert( x )

# freezes the parameters for the whole network
def modify_network_for_greek_letters(model):
    for param in model.parameters():
        param.requires_grad = False
    
    num_features = model.network[-2].out_features
    model.network[-1] = nn.Sequential(
        nn.Linear(num_features, 3),  
        nn.LogSoftmax(dim=1)
    )
    return model

#fucntion to run training data on the model
def train(model, train_loader):
    model.train()
    train_losses = []
    train_counter = []
    examples_processed = 0  

    for epoch in range(1, 6):  
        epoch_loss = 0
        for i, (data, target) in enumerate(train_loader):
            output = model(data)
            loss = F.nll_loss(output, target)
            loss.backward()

            epoch_loss += loss.item()
            examples_processed += len(data)  

            if i % 10 == 0:  
                train_losses.append(loss.item())
                train_counter.append(examples_processed)  
                print(f'Epoch {epoch}, Batch {i}, Loss: {loss.item()}')
                
        print(f'Epoch {epoch} complete. Average Loss: {epoch_loss / len(train_loader)}')

    # Plot training loss
    plt.figure(figsize=(10, 6))
    plt.plot(train_counter, train_losses, marker='o', linestyle='-', color='blue')
    plt.legend(['Train Loss'], loc='upper right')
    plt.xlabel('Number of training examples seen')
    plt.ylabel('Negative log likelihood loss')
    plt.title('Training Loss Over Time')
    plt.show()

    return train_losses, train_counter

#fucntion to run testing data on the model
def test_and_display(model, test_loader):
    model.eval()
    with torch.no_grad():
        for data, target in test_loader:
            output = model(data)
            pred = output.argmax(dim=1, keepdim=True)
            fig, axs = plt.subplots(1, len(data), figsize=(15, 2.5))
            for i in range(len(data)):
                axs[i].imshow(data[i].squeeze(), cmap='gray')
                axs[i].set_title(f'Predicted: {pred[i].item()}\nTrue: {target[i]}')
                axs[i].axis('off')
            plt.show()


if __name__ == '__main__':
    training_set_path = r"C:\Users\kevin\Downloads\greek_train\greek_train"  
    test_set_path = r"C:\Users\kevin\Downloads\greektest"

    model = TheMNISTNetwork()
    model = modify_network_for_greek_letters(model)

    # DataLoader for the Greek data training set
    greek_train = DataLoader(
        ImageFolder(training_set_path, transform=Compose([ToTensor(), GreekTransform(), Normalize((0.1307,), (0.3081,))])),
        batch_size=5, 
        shuffle=True
    )
    # DataLoader for the Greek data test set
    greek_test = DataLoader(ImageFolder(test_set_path, transform=Compose([ToTensor(), GreekTransform(), Normalize((0.1307,), (0.3081,))])), batch_size=10, shuffle=False)

    train(model, greek_train)
    test_and_display(model, greek_test)
    print(model)
   