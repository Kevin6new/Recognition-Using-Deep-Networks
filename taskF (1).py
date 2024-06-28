"""
Kevin Sani and Basil Reji
CS5330
Project 5 : Recognition using Deep Networks  
Task 1:F
"""
import matplotlib.pyplot as plt
import torch
from torchvision import transforms
from PIL import Image
import os
from task1 import TheMNISTNetwork  

#Convert to greyscale the test images
def preprocess_image(image_path):
    transform = transforms.Compose([
        transforms.Grayscale(),
        transforms.Resize((28, 28)),
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,)),
        transforms.Lambda(lambda x: 1.0 - x)
    ])
    image = Image.open(image_path)
    return transform(image).unsqueeze(0)

#load the model from system
def load_model(model_path='model.pth'):
    model = TheMNISTNetwork()
    model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
    model.eval()
    return model

model = load_model('model.pth')
data_folder = "C:\\Users\\kevin\\Downloads\\mydata"

fig, axs = plt.subplots(2, 5, figsize=(20, 4))  

idx = 0  

#load the image from the new test dataset and compare them ti the actual data and dispalying the output 
for filename in os.listdir(data_folder):
    if filename.endswith(".jpg") or filename.endswith(".png") and idx < 5: 
        image_path = os.path.join(data_folder, filename)
        image = preprocess_image(image_path)
        with torch.no_grad():
            output = model(image)
            prediction = output.argmax(dim=1, keepdim=True).item()

        img = Image.open(image_path)
        axs[0, idx].imshow(img, cmap='gray')
        axs[0, idx].set_title(f"Predicted: {prediction}")
        axs[0, idx].axis('off')

        processed_img = preprocess_image(image_path).squeeze().numpy().squeeze()  
        axs[1, idx].imshow(processed_img, cmap='gray')
        axs[1, idx].set_title("Processed")
        axs[1, idx].axis('off')

        idx += 1  

plt.tight_layout()
plt.show()
