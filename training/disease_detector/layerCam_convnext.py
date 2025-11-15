'''
import os
import torch
import timm
import cv2
import numpy as np
from PIL import Image
from torchvision import transforms
from sklearn.metrics import jaccard_score
from pytorch_grad_cam import LayerCAM
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget

# === CONFIG ===
img_size = 224
conf_thresh = 0.5
iou_thresh = 0.4
device = 'cuda' if torch.cuda. is_available() else 'cpu'

# === PATHS ===
image_folder = r"D:\CS\AI\PROJECT\ML_agri project\new_models\disease_detection\images"
mask_folder = r"D:\CS\AI\PROJECT\ML_agri project\new_models\disease_detection\masks"
model_path = r"D:\CS\AI\PROJECT\ML_agri project\new_models\disease_detection\convnext_disease.pth"

class_names = [
    'Anthracnose', 'Bacterial Canker', 'Cutting Weevil', 'Die Back',
    'Gall Midge', 'Healthy', 'Powdery Mildew', 'Sooty Mould'
]

# === Load Model ===
model = timm.create_model('convnext_tiny', pretrained=False, num_classes=len(class_names))
model.load_state_dict(torch.load(model_path, map_location=device))
model.eval().to(device)
target_layers = [model.stages[-1].blocks[-1].norm]

# === Transform ===
transform = transforms.Compose([
    transforms.Resize((img_size, img_size)),
    transforms.ToTensor(),
    transforms.Normalize([0.5]*3, [0.5]*3)
])

# === Results Log ===
results = []

# === Process Each Image ===
image_files = sorted([f for f in os.listdir(image_folder) if f.endswith(('.png', '.jpg', '.jpeg'))])

for filename in image_files:
    img_path = os.path.join(image_folder, filename)
    base_name = os.path.splitext(filename)[0]
    mask_name = f"{base_name}_mask.png"
    mask_path = os.path.join(mask_folder, mask_name)

    # Skip if mask doesn't exist
    if not os.path.exists(mask_path):
        print(f"[!] Missing mask for {filename}")
        continue

    # Load image and mask
    pil_img = Image.open(img_path).convert("RGB")
    input_tensor = transform(pil_img).unsqueeze(0).to(device)
    original_np = np.array(pil_img.resize((img_size, img_size))) / 255.0
    gt_mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
    gt_mask = cv2.resize(gt_mask, (img_size, img_size))
    gt_binary = (gt_mask > 127).astype(np.uint8)

    # Predict
    with torch.no_grad():
        output = model(input_tensor)
        pred_class = torch.argmax(output, dim=1).item()
        confidence = torch.softmax(output, dim=1)[0][pred_class].item()

    # Layer-CAM
    cam = LayerCAM(model=model, target_layers=target_layers)
    grayscale_cam = cam(input_tensor=input_tensor, targets=[ClassifierOutputTarget(pred_class)])[0]
    cam_thresh = np.percentile(grayscale_cam, 75)
    cam_binary = (grayscale_cam >= cam_thresh).astype(np.uint8)

    # IoU
    iou = jaccard_score(gt_binary.flatten(), cam_binary.flatten())

    # Decision
    if confidence > conf_thresh and iou > iou_thresh:
        decision = f"✅ {class_names[pred_class]}"
    else:
        decision = "⚠️ Uncertain"

    print(f"{filename} | Class: {class_names[pred_class]} | Conf: {confidence:.4f} | IoU: {iou:.4f} | {decision}")

    # Store result
    results.append({
        "Image": filename,
        "Class": class_names[pred_class],
        "Confidence": round(confidence, 4),
        "IoU": round(iou, 4),
        "Decision": decision
    })

# === Summary Table ===
print("\n📊 Summary:")
for r in results:
    print(f"{r['Image']:<20} | {r['Class']:<20} | Conf: {r['Confidence']:<6} | IoU: {r['IoU']:<6} | {r['Decision']}")
'''
import torch
import timm
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from torchvision import transforms
from pytorch_grad_cam import LayerCAM, GradCAM
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
from pytorch_grad_cam.utils.image import show_cam_on_image

# === CONFIG ===
img_path = r"C:\Users\sabin\Downloads\images (1).jpg"
model_path = r"D:\CS\AI\PROJECT\ML_agri project\new_models\disease_detection\convnext_disease.pth"
img_size = 224
class_names = [
    'Anthracnose', 'Bacterial Canker', 'Cutting Weevil', 'Die Back',
    'Gall Midge', 'Healthy', 'Powdery Mildew', 'Sooty Mould'
]

device = 'cuda' if torch.cuda.is_available() else 'cpu'

# === LOAD MODEL ===
model = timm.create_model('convnext_tiny', pretrained=False, num_classes=len(class_names))
model.load_state_dict(torch.load(model_path, map_location=device))
model.eval().to(device)

# === PICK TARGET LAYER SAFELY ===
last_block = model.stages[-1].blocks[-1]
if hasattr(last_block, "depthwise_conv"):
    target_layers = [last_block.depthwise_conv]
else:
    target_layers = [last_block]  # fallback

# === LOAD IMAGE ===
pil_img = Image.open(img_path).convert("RGB")

transform = transforms.Compose([
    transforms.Resize((img_size, img_size)),
    transforms.ToTensor(),
    # 🔄 Use ImageNet normalization (default for timm models)
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225])
    # If you trained with [0.5, 0.5, 0.5], replace with:
    # transforms.Normalize([0.5]*3, [0.5]*3)
])

input_tensor = transform(pil_img).unsqueeze(0).to(device)
rgb_img = np.array(pil_img.resize((img_size, img_size))) / 255.0  # for visualization

# === PREDICT CLASS ===
with torch.no_grad():
    output = model(input_tensor)
    pred_class = torch.argmax(output, dim=1).item()
    confidence = torch.softmax(output, dim=1)[0][pred_class].item()
    label = class_names[pred_class]

# === GENERATE CAM ===
# Start with LayerCAM; switch to GradCAM if heatmaps look strange
cam = LayerCAM(model=model, target_layers=target_layers)
# cam = GradCAM(model=model, target_layers=target_layers, use_cuda=(device == 'cuda'))

grayscale_cam = cam(input_tensor=input_tensor,
                    targets=[ClassifierOutputTarget(pred_class)])[0]

heatmap = show_cam_on_image(rgb_img, grayscale_cam, use_rgb=True)

# === DISPLAY RESULT ===
plt.figure(figsize=(8, 6))
plt.imshow(heatmap)
plt.title(f"Predicted: {label} ({confidence*100:.2f}%)", fontsize=16)
plt.axis('off')
plt.tight_layout()
plt.show()
