import os
import torch
import timm
import random
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from PIL import Image
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Subset
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.metrics import classification_report, confusion_matrix, cohen_kappa_score, accuracy_score
from lime import lime_image
from skimage.segmentation import mark_boundaries
from torch import nn, optim
from tqdm import tqdm

# === Configuration ===
DATA_DIR = r"D:\CS\AI\PROJECT\ML_agri project\Mango Variety and Grading Dataset\Dataset\Grading_dataset"
MODEL_PATH = r"D:\AMRITA\sem 3\semVI_project\machine_learning\coat_mango_model.pt"
BATCH_SIZE = 32
IMAGE_SIZE = 224
NUM_CLASSES = 3
EPOCHS = 5
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# === Transformations ===
transform = transforms.Compose([
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# === Load Dataset ===
dataset = datasets.ImageFolder(DATA_DIR, transform=transform)
class_names = dataset.classes

# === Stratified Split ===
targets = [sample[1] for sample in dataset.samples]
sss = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
train_idx, test_idx = next(sss.split(np.zeros(len(targets)), targets))

train_dataset = Subset(dataset, train_idx)
test_dataset = Subset(dataset, test_idx)

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

# === Load Pretrained CoaT ===
model = timm.create_model("coat_lite_small", pretrained=True, num_classes=NUM_CLASSES)
model.to(DEVICE)

# === Loss & Optimizer ===
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=1e-4)

# === Training Loop ===
for epoch in range(EPOCHS):
    model.train()
    total_loss = 0
    correct = 0
    for images, labels in tqdm(train_loader, desc=f"Epoch {epoch + 1}/{EPOCHS}", leave=False):
        images, labels = images.to(DEVICE), labels.to(DEVICE)
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        total_loss += loss.item()
        loss.backward()
        optimizer.step()
        preds = outputs.argmax(1)
        correct += (preds == labels).sum().item()

    acc = correct / len(train_loader.dataset)
    print(f"Epoch {epoch + 1}, Loss: {total_loss:.4f}, Accuracy: {acc:.4f}")

# === Save Model ===
os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)
torch.save(model.state_dict(), MODEL_PATH)
print(f"Model saved to {MODEL_PATH}")

# === Evaluation ===
model.eval()
y_true, y_pred = [], []

with torch.no_grad():
    for images, labels in test_loader:
        images, labels = images.to(DEVICE), labels.to(DEVICE)
        outputs = model(images)
        preds = outputs.argmax(1)
        y_true.extend(labels.cpu().numpy())
        y_pred.extend(preds.cpu().numpy())

print("\n=== Evaluation Results ===")
print("Classification Report:")
print(classification_report(y_true, y_pred, target_names=class_names))

print("\nConfusion Matrix:")
cm = confusion_matrix(y_true, y_pred)
print(cm)

print("\nCohen’s Kappa:", cohen_kappa_score(y_true, y_pred))
print("Accuracy:", accuracy_score(y_true, y_pred))

# Optional Confusion Matrix Heatmap
plt.figure(figsize=(6, 5))
sns.heatmap(cm, annot=True, fmt="d", xticklabels=class_names, yticklabels=class_names, cmap="Blues")
plt.xlabel("Predicted")
plt.ylabel("True")
plt.title("Confusion Matrix")
plt.show()


# === LIME Explanation ===
def get_lime_explanation(img_path, model, class_names):
    model.eval()
    explainer = lime_image.LimeImageExplainer()

    def batch_predict(images):
        batch = torch.stack([
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])(
                transforms.ToTensor()(Image.fromarray(img))
            ).to(DEVICE) for img in images
        ])
        with torch.no_grad():
            logits = model(batch)
        return logits.cpu().numpy()

    img = Image.open(img_path).convert('RGB').resize((IMAGE_SIZE, IMAGE_SIZE))
    img_np = np.array(img)

    explanation = explainer.explain_instance(
        img_np, batch_predict, top_labels=1, hide_color=0, num_samples=1000
    )
    temp, mask = explanation.get_image_and_mask(
        explanation.top_labels[0], positive_only=True, num_features=5, hide_rest=False
    )
    plt.imshow(mark_boundaries(temp / 255.0, mask))
    plt.title(f"LIME Explanation: {os.path.basename(img_path)}")
    plt.axis('off')
    plt.show()


# Example usage (provide an image path):
get_lime_explanation(r"C:\Users\sabin\Downloads\download (1).jpeg", model, class_names)
