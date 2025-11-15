from pytorch_grad_cam import LayerCAM
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
from pytorch_grad_cam.utils.image import show_cam_on_image
import matplotlib.pyplot as plt
import timm
import torch
from torchvision import transforms
import numpy as np
from PIL import Image
import os
import tempfile
import streamlit as st

@st.cache_resource
def load_disease_model():
    model_path = r"D:\CS\AI\PROJECT\ML_agri project\new_models\disease_detection\convnext_disease.pth"
    class_names = [
        'Anthracnose', 'Bacterial Canker', 'Cutting Weevil', 'Die Back',
        'Gall Midge', 'Healthy', 'Powdery Mildew', 'Sooty Mould'
    ]
    model = timm.create_model('convnext_tiny', pretrained=False, num_classes=len(class_names))
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval().to(device)
    return model, class_names, device

def predict_disease(image_file):
    model, class_names, device = load_disease_model()
    img_size = 224

    # Load image
    pil_img = Image.open(image_file).convert("RGB")
    rgb_img = np.array(pil_img.resize((img_size, img_size))) / 255.0

    transform = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        # ⚠️ Change normalization if you trained with ImageNet stats
        transforms.Normalize([0.5]*3, [0.5]*3)
    ])
    input_tensor = transform(pil_img).unsqueeze(0).to(device)

    # Prediction
    with torch.no_grad():
        output = model(input_tensor)
        pred_class = torch.argmax(output, dim=1).item()
        confidence = torch.softmax(output, dim=1)[0][pred_class].item()
        label = class_names[pred_class]

    # Grad-CAM
    last_block = model.stages[-1].blocks[-1]
    if hasattr(last_block, "depthwise_conv"):
        target_layers = [last_block.depthwise_conv]
    else:
        target_layers = [last_block]  # fallback

    cam = LayerCAM(model=model, target_layers=target_layers)
    grayscale_cam = cam(input_tensor=input_tensor, targets=[ClassifierOutputTarget(pred_class)])[0]
    heatmap = show_cam_on_image(rgb_img, grayscale_cam, use_rgb=True)

    return label, confidence, heatmap
