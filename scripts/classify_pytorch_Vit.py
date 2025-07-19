import os, sys
import torch
from torchvision import models, transforms
from torch import nn
from PIL import Image
import torch.nn.functional as F

import numpy as np

import warnings
warnings.filterwarnings("ignore")

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def load_vit_model(model_path, num_classes=2, device=None):
    """
    Load a ViT model from a .pth file.
    Args:
        model_path (str): Path to the .pth file.
        num_classes (int): Number of output classes.
        device (torch.device or None): Device to load the model on.
    Returns:
        model (torch.nn.Module): Loaded model in eval mode.
    """
    from timm import create_model
    if device is None:
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = create_model("vit_base_patch16_224", pretrained=False, num_classes=num_classes)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model = model.to(device)
    model.eval()
    return model

transform = transforms.Compose([
    transforms.Resize((224, 224)),  # Resize the image
    transforms.ToTensor(),         # Convert to tensor
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),  # Normalize
])

def preprocess_image(image_path):
    image = Image.open(image_path).convert("RGB")  # Ensure the image is RGB
    image = transform(image)
    return image.unsqueeze(0)  # Add a batch dimension

def predict_image_vit(model, image_path, device=None):
    """
    Predict the class and probabilities for a single image using a loaded ViT model.
    Args:
        model (torch.nn.Module): Loaded model.
        image_path (str): Path to the image file.
        device (torch.device or None): Device to run prediction on.
    Returns:
        predicted_class (int): Predicted class index.
        probabilities (np.ndarray): Probabilities for each class.
    """
    if device is None:
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    input_image = preprocess_image(image_path).to(device)
    with torch.no_grad():
        output = model(input_image)
        _, predicted_class = torch.max(output, 1)  # Get the class index with the highest score
        probabilities = F.softmax(output, dim=1)
        probabilities = probabilities.detach().cpu().numpy()
    return predicted_class.item(), probabilities[0]
