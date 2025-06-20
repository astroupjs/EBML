import torch
from torch import nn, optim
from torch.utils.data import DataLoader
from torchvision import datasets, models, transforms
from tqdm import tqdm
import json
import os


data_dir = ""  # Update with your dataset path

batch_size = 32
num_epochs = 10
num_classes = 2
learning_rate = 0.001
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

print(device)

def prepare_training(data_dir, batch_size=32):
    # Print the resolved data_dir for debugging
    print(f"Resolved data_dir: {os.path.abspath(data_dir)}")
    data_dir = os.path.abspath(data_dir)
    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]
    data_transforms = {
        'train': transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.ToTensor(),
            transforms.Normalize(mean, std),
        ]),
        'val': transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.ToTensor(),
            transforms.Normalize(mean, std),
        ]),
    }
    for x in ['train', 'val']:
        folder = os.path.join(data_dir, x)
        print(f"Checking for folder: {folder} (exists: {os.path.exists(folder)})")
    # Raise an error if the folders do not exist
    for x in ['train', 'val']:
        folder = os.path.join(data_dir, x)
        if not os.path.exists(folder):
            raise FileNotFoundError(f"Expected folder not found: {folder}. Please check your data_dir path and structure.")
    # List subfolders for debugging
    for x in ['train', 'val']:
        folder = os.path.join(data_dir, x)
        print(f"Subfolders in {folder}: {os.listdir(folder) if os.path.exists(folder) else 'N/A'}")
    image_datasets = {
        x: datasets.ImageFolder(root=os.path.join(data_dir, x), transform=data_transforms[x])
        for x in ['train', 'val']
    }
    dataloaders = {
        x: DataLoader(image_datasets[x], batch_size=batch_size, shuffle=(x == 'train'), num_workers=4)
        for x in ['train', 'val']
    }
    return dataloaders


resnet = models.resnet50(pretrained=True)
num_ftrs = resnet.fc.in_features
resnet.fc = nn.Linear(num_ftrs, 2)  # Output 2 logits for binary classification
resnet = resnet.to(device)


criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(resnet.parameters(), lr=learning_rate)
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=7, gamma=0.1)

def train_model(model, dataloaders, criterion, optimizer, scheduler, num_epochs, save_path="training_progress.json"):
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

            for inputs, labels in tqdm(dataloaders[phase]):
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


# Only define functions and variables at the top level. Do not execute any code at import time.
# Remove or comment out any top-level calls to prepare_training, train_model, torch.save, or evaluate_model.

# Example usage (for reference, not to be executed at import):
# dataloaders = prepare_training(data_dir, batch_size)
# trained_model = train_model(resnet, dataloaders, criterion, optimizer, scheduler, num_epochs)
# torch.save(trained_model.state_dict(), "model_ResNet.pth")
# evaluate_model(trained_model, dataloaders['val'])

