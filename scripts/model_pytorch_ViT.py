import sys, os
import torch
from torch import nn, optim
import timm
import torchvision.transforms as transforms
import torchvision.datasets as datasets
from torch.utils.data import DataLoader
from tqdm import tqdm
import json

def prepare_training(data_dir, batch_size=32, num_classes=2, num_workers=4):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    mean = [0.5, 0.5, 0.5]
    std = [0.5, 0.5, 0.5]
    data_transforms = {
        'train': transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.ToTensor(),
            transforms.Normalize(mean, std),
        ]),
        'val': transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean, std),
        ]),
    }
    image_datasets = {}
    dataloaders = {}
    for x in ['train', 'val']:
        folder = os.path.join(data_dir, x)
        print(f"Loading images from: {folder}")
        image_datasets[x] = datasets.ImageFolder(root=folder, transform=data_transforms[x])
        dataloaders[x] = DataLoader(image_datasets[x], batch_size=batch_size, shuffle=(x == 'train'), num_workers=num_workers)
        print(f"Loaded {len(image_datasets[x])} images for {x}")
    return dataloaders

def create_vit_model(num_classes=2):
    model = timm.create_model("vit_base_patch16_224", pretrained=True, num_classes=num_classes)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model.to(device)
    return model

def get_loss_optimizer_scheduler(model, lr=1e-4, weight_decay=1e-4, step_size=7, gamma=0.1):
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=step_size, gamma=gamma)
    return criterion, optimizer, scheduler

def train_model(model, dataloaders, criterion, optimizer, scheduler, num_epochs, save_path="training_progress.json"):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    history = {"train_loss": [], "val_loss": [], "train_acc": [], "val_acc": []}
    for epoch in range(num_epochs):
        print(f"Epoch {epoch+1}/{num_epochs}")
        print("-" * 10)
        for phase in ['train', 'val']:
            if phase == 'train':
                model.train()
            else:
                model.eval()
            running_loss = 0.0
            running_corrects = 0
            for inputs, labels in tqdm(dataloaders[phase], desc=f"{phase} phase"):
                inputs, labels = inputs.to(device), labels.to(device)
                optimizer.zero_grad()
                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(inputs)
                    _, preds = torch.max(outputs, 1)
                    loss = criterion(outputs, labels)
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()
                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)
            if phase == 'train' and scheduler is not None:
                scheduler.step()
            epoch_loss = running_loss / len(dataloaders[phase].dataset)
            epoch_acc = running_corrects.double() / len(dataloaders[phase].dataset)
            print(f"{phase} Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}")
            history[f"{phase}_loss"].append(epoch_loss)
            history[f"{phase}_acc"].append(epoch_acc.item())
        with open(save_path, "w") as f:
            json.dump(history, f)
    return model

def evaluate_model(model, dataloader):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model.eval()
    correct = 0
    total = 0
    y_true, y_pred = [], []
    with torch.no_grad():
        for inputs, labels in dataloader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)
            y_true.extend(labels.cpu().numpy())
            y_pred.extend(preds.cpu().numpy())
    print(f"Validation Accuracy: {correct / total:.4f}")
    return y_true, y_pred
