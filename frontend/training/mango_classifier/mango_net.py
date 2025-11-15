import os
import timm
import torch
import torch.nn as nn
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix
import numpy as np

def main():
    # === Config ===
    data_dir = r"D:\CS\AI\PROJECT\ML_agri project\Mango Variety\mango images\mango images"
    img_size = 224
    batch_size = 32
    epochs = 5
    num_workers = 2
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # === Transforms ===
    transform = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(10),
        transforms.ToTensor(),
        transforms.Normalize([0.5]*3, [0.5]*3)
    ])

    # === Datasets ===
    train_dataset = datasets.ImageFolder(data_dir, transform=transform)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)

    # === Model: RepVGG ===
    model = timm.create_model('repvgg_a0', pretrained=True, num_classes=len(train_dataset.classes))
    model.to(device)

    # === Training setup ===
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

    # === Training loop ===
    train_acc = []
    train_loss = []

    for epoch in range(epochs):
        model.train()
        total, correct, running_loss = 0, 0, 0.0

        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()

            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            _, predicted = torch.max(outputs, 1)
            correct += (predicted == labels).sum().item()
            total += labels.size(0)

        acc = correct / total
        train_acc.append(acc)
        train_loss.append(running_loss / len(train_loader))

        print(f"Epoch {epoch+1}/{epochs} - Loss: {train_loss[-1]:.4f} - Accuracy: {acc:.4f}")

    # === Save model ===
    torch.save(model.state_dict(), r'D:\CS\AI\PROJECT\ML_agri project\new_models\repvgg_mango_classifier.pth')

    # === Plot Training Loss and Accuracy ===
    plt.plot(train_acc, label='Accuracy')
    plt.plot(train_loss, label='Loss')
    plt.title('RepVGG Training')
    plt.xlabel('Epoch')
    plt.legend()
    plt.show()

    # === Evaluate Model ===
    model.eval()  # Set the model to evaluation mode
    all_labels = []
    all_preds = []

    with torch.no_grad():  # Disable gradient calculation for inference
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, preds = torch.max(outputs, 1)
            all_labels.extend(labels.cpu().numpy())
            all_preds.extend(preds.cpu().numpy())

    # === Classification Report ===
    print("\nClassification Report:")
    print(classification_report(all_labels, all_preds, target_names=train_dataset.classes))

    # === Confusion Matrix ===
    cm = confusion_matrix(all_labels, all_preds)
    plt.figure(figsize=(10, 7))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=train_dataset.classes, yticklabels=train_dataset.classes)
    plt.title('Confusion Matrix')
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.show()

if __name__ == '__main__':
    main()
