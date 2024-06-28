"""
Kevin Sani and Basil Reji
CS5330
Project 5 : Recognition using Deep Networks  
Task 1:E
"""
import torch
from task1 import TheMNISTNetwork
from torchvision.datasets import MNIST
from torchvision.transforms import ToTensor
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt

#load the model from system
def load_model(model_path='model.pth'):
    model = TheMNISTNetwork()
    model.load_state_dict(torch.load(model_path))
    model.eval()  
    return model

#Running model on test set
def run_on_test_set(model, test_loader):
    data_iter = iter(test_loader)
    images, labels = next(data_iter)
    
    with torch.no_grad():
        outputs = model(images[:10])
    
    return images[:10], outputs, labels[:10]

#Printing predcitions of the etst set
def print_predictions(images, outputs, labels):
    _, predicted = torch.max(outputs, 1)
    
    for i in range(10):
        print(f"Image {i+1}:")
        print(" Output values:", ["{:.2f}".format(value) for value in outputs[i]])
        print(" Predicted:", predicted[i].item(), "Correct Label:", labels[i].item())
        print()

#Plotting images and comparing between ppredicted and actual output 
def plot_images(images, predicted, labels):
    fig, axs = plt.subplots(3, 3, figsize=(10, 10))
    for i, ax in enumerate(axs.flat):
        if i >= 9:  
            break
        ax.imshow(images[i].reshape(28, 28), cmap='gray')
        ax.set_title(f"Pred: {predicted[i].item()} True: {labels[i].item()}")
        ax.axis('off')
    
    plt.show()

if __name__ == '__main__':
    test_loader = DataLoader(MNIST('./data', train=False, download=True, transform=ToTensor()), batch_size=10, shuffle=False)
    model = load_model('model.pth')
    images, outputs, labels = run_on_test_set(model, test_loader)
    print_predictions(images, outputs, labels)
    plot_images(images, outputs.max(1)[1], labels)
