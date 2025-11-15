import os
import timm
import torch
import torch.nn as nn
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

# === CONFIG ===
data_dir = r"C:\Users\sabin\Downloads\archive (9)"
img_size = 224
batch_size = 32
epochs = 10
device = 'cuda' if torch.cuda.is_available() else 'cpu'

# === TRANSFORMS ===
transform = transforms.Compose([
    transforms.Resize((img_size, img_size)),
    transforms.ToTensor(),
    transforms.Normalize([0.5]*3, [0.5]*3)
])


# === DATA LOADERS ===
from sklearn.model_selection import train_test_split
from torchvision.datasets import ImageFolder
from torch.utils.data import Subset

# Load full dataset
full_dataset = ImageFolder(data_dir, transform=transform)

# Split into train/val indices
indices = list(range(len(full_dataset)))
train_idx, val_idx = train_test_split(indices, test_size=0.2, stratify=full_dataset.targets, random_state=42)

train_ds = Subset(full_dataset, train_idx)
val_ds = Subset(full_dataset, val_idx)

train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_ds, batch_size=batch_size)

class_names = full_dataset.classes

num_classes = len(class_names)
# === MODEL: ConvNeXt-Tiny ===
model = timm.create_model('convnext_tiny', pretrained=True, num_classes=num_classes)
model.to(device)

# === OPTIMIZER AND LOSS ===
optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
criterion = nn.CrossEntropyLoss()

# === TRAINING LOOP ===
for epoch in range(epochs):
    model.train()
    correct, total = 0, 0
    for images, labels in train_loader:
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        preds = torch.argmax(outputs, dim=1)
        correct += (preds == labels).sum().item()
        total += labels.size(0)

    train_acc = correct / total

    # === VALIDATION ===
    model.eval()
    correct, total = 0, 0
    all_preds = []
    all_labels = []
    with torch.no_grad():
        for images, labels in val_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            preds = torch.argmax(outputs, dim=1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            correct += (preds == labels).sum().item()
            total += labels.size(0)

    val_acc = correct / total
    print(f"Epoch {epoch+1}/{epochs} - Train Acc: {train_acc:.3f}, Val Acc: {val_acc:.3f}")

# === SAVE MODEL ===
torch.save(model.state_dict(), r"D:\CS\AI\PROJECT\ML_agri project\new_models\disease_detection\convnext_disease.pth")
