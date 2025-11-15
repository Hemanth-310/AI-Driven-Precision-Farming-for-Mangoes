import cv2
import numpy as np
import torch
import timm
from PIL import Image
import mahotas
import joblib
from torchvision import transforms
from lime import lime_image
from skimage.segmentation import mark_boundaries
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
import io

# Load Random Forest model
rf_model = joblib.load(r"D:\CS\AI\PROJECT\ML_agri project\new_models\fruit_grading\random_forest_model.pkl")

# Define class names
class_names = {0: 'Undamaged', 1: 'Mild Damage', 2: 'Severe Damage'}

# Setup CoaT
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
coat = timm.create_model("coat_lite_small", pretrained=True)
coat.head = torch.nn.Identity()
coat.eval()
coat.to(DEVICE)

# Transform for CoaT
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# Feature extractors
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
    img_pil = Image.open(img_path).convert("RGB").resize((224, 224))
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

# Prediction function
def predict_damage(image_path):
    features = extract_all_features(image_path)
    pred_rf = rf_model.predict([features])[0]
    
    img = cv2.imread(image_path)
    if img is None:
        raise ValueError("Image could not be read.")
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    return img_rgb, class_names[pred_rf], pred_rf
