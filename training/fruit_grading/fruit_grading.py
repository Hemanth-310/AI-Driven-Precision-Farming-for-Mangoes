# Hybrid Mango Grading with CoaT + Handcrafted Features + Random Forest

import os
import numpy as np
import torch
import timm
import cv2
import mahotas

from PIL import Image
from tqdm import tqdm
from torchvision import transforms, datasets
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix

# Configuration
DATA_DIR = r"D:\CS\AI\PROJECT\ML_agri project\Mango Variety and Grading Dataset\Dataset\Grading_dataset"  # Directory with subfolders for each class
IMAGE_SIZE = 224
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Transform for CoaT
transform = transforms.Compose([
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# Load Pretrained CoaT
coat = timm.create_model("coat_lite_small", pretrained=True)
coat.head = torch.nn.Identity()
coat.eval()
coat.to(DEVICE)

# Handcrafted Feature Extractors
def color_histogram(img):
    hsv = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
    hist = cv2.calcHist([hsv], [0, 1, 2], None, [8, 8, 8], [0,180,0,256,0,256])
    return cv2.normalize(hist, hist).flatten()

def texture_features(img_gray):
    return mahotas.features.haralick(img_gray).mean(axis=0)

def shape_features(img_gray):
    moments = cv2.moments(img_gray)
    return cv2.HuMoments(moments).flatten()

def extract_cnn_features(image_tensor):
    with torch.no_grad():
        image_tensor = image_tensor.unsqueeze(0).to(DEVICE)
        features = coat(image_tensor)
        return features.cpu().numpy().flatten()

def extract_all_features(img_path):
    img_pil = Image.open(img_path).convert("RGB").resize((IMAGE_SIZE, IMAGE_SIZE))
    img_np = np.array(img_pil)
    img_gray = cv2.cvtColor(img_np, cv2.COLOR_RGB2GRAY)

    handcrafted = np.concatenate([
        color_histogram(img_np),
        texture_features(img_gray),
        shape_features(img_gray)
    ])

    img_tensor = transform(img_pil)
    cnn_feat = extract_cnn_features(img_tensor)

    return np.concatenate([cnn_feat, handcrafted])

# Load and process dataset
def load_dataset():
    X, y = [], []
    class_map = {}
    label = 0

    for class_name in sorted(os.listdir(DATA_DIR)):
        class_path = os.path.join(DATA_DIR, class_name)
        if not os.path.isdir(class_path):
            continue
        class_map[label] = class_name
        for img_file in tqdm(os.listdir(class_path), desc=f"Processing {class_name}"):
            img_path = os.path.join(class_path, img_file)
            try:
                features = extract_all_features(img_path)
                X.append(features)
                y.append(label)
            except:
                print(f"Failed to process {img_path}")
        label += 1

    return np.array(X), np.array(y), class_map

# Train and evaluate classifier
import joblib

def train_classifier(X, y, class_map):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)

    clf = RandomForestClassifier(n_estimators=100, random_state=42)
    clf.fit(X_train, y_train)

    y_pred = clf.predict(X_test)
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred, target_names=[class_map[i] for i in sorted(class_map.keys())]))

    print("Confusion Matrix:")
    print(confusion_matrix(y_test, y_pred))

    # 🔽 Save the model
    joblib.dump(clf, r"D:\CS\AI\PROJECT\ML_agri project\new_models\fruit_grading\random_forest_model.pkl")
    print("Model saved as models/random_forest_model.pkl")

    return clf


if __name__ == '__main__':
    print("\nExtracting features and loading dataset...")
    X, y, class_map = load_dataset()

    print("\nTraining classifier...")
    clf = train_classifier(X, y, class_map)

    print("\nDone.")