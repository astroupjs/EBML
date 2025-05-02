import os, sys
import torch
from torchvision import models, transforms
from torch import nn
from PIL import Image
import torch.nn.functional as F

import numpy as np
import pathlib

import warnings
warnings.filterwarnings("ignore")

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# Load the model
num_classes = 2  # Replace with the number of classes in your dataset

model = models.resnet50(pretrained=False)
model.fc = nn.Linear(model.fc.in_features, num_classes)
model.load_state_dict(torch.load("model_ResNet.pth", map_location=device))
model = model.to(device)
model.eval()  # Set the model to evaluation mode

transform = transforms.Compose([
    transforms.Resize((224, 224)),  # Resize the image
    transforms.ToTensor(),         # Convert to tensor
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),  # Normalize
])

def preprocess_image(image_path):
    image = Image.open(image_path).convert("RGB")  # Ensure the image is RGB
    image = transform(image)
    return image.unsqueeze(0)  # Add a batch dimension


image_name="" #name of image with transformed light curve

input_image = preprocess_image(image_name).to(device)
output = model(input_image)
_, predicted_class = torch.max(output, 1)  # Get the class index with the highest score
probabilities = F.softmax(output, dim=1)
probabilities = probabilities.detach().cpu().numpy()

print(predicted_class)


