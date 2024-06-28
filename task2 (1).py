"""
Kevin Sani and Basil Reji
CS5330
Project 5 : Recognition using Deep Networks  
Task 2
"""
import torch
from task1 import TheMNISTNetwork
import matplotlib.pyplot as plt
import torchvision.transforms as transforms
from torchvision.datasets import MNIST
from torch.utils.data import DataLoader
import cv2
import numpy as np

#load the model from system
def load_model(model_path='model.pth'):
    model = TheMNISTNetwork()
    model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
    model.eval()
    return model

#Function to visualziae the filters using and the output images too to see the filter used on the image
def visualize_filters_and_effects(model):
    train_loader = DataLoader(MNIST('./data', train=True, download=True, transform=transforms.ToTensor()), batch_size=1, shuffle=True)
    data_iter = iter(train_loader)
    images, _ = next(data_iter)
    
    with torch.no_grad():
        try:
            conv1_weights = model.conv1.weight.data
        except AttributeError:
            conv1_weights = model.network[0].weight.data

    fig, axs = plt.subplots(2, 10, figsize=(20, 4))
    image = images[0].numpy().squeeze()

    for i in range(10):
        filter_weight = conv1_weights[i, 0].numpy()
        filtered_image = cv2.filter2D(image, -1, filter_weight)

        axs[0, i].imshow(filter_weight, cmap='viridis')
        axs[0, i].set_title(f'Filter {i+1}')
        axs[0, i].axis('off')

        axs[1, i].imshow(filtered_image, cmap='gray')
        axs[1, i].set_title(f'Effect {i+1}')
        axs[1, i].axis('off')

    plt.tight_layout()
    plt.show()

if __name__ == '__main__':
    model = load_model('model.pth')
    visualize_filters_and_effects(model)
