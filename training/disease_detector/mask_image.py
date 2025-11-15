'''
import os
import numpy as np
import PIL.Image
from labelme import utils
import json
import cv2
# === CONFIG ===
json_path = r"D:\CS\AI\PROJECT\ML_agri project\new_models\disease_detection\sooty mould.json"
output_dir = os.path.splitext(json_path)[0] + "_output"

import os
import numpy as np
import PIL.Image
from labelme import utils
import json
import cv2  # Ensure OpenCV is imported



# === LOAD LABELME JSON ===
with open(json_path) as f:
    data = json.load(f)

imageData = data.get("imageData")

if not imageData:
    # Load the image file directly if imageData is missing
    imagePath = os.path.join(os.path.dirname(json_path), data["imagePath"])
    with open(imagePath, "rb") as img_f:
        imageData = utils.img_to_base64(PIL.Image.open(img_f)).decode("utf-8")

# === PROCESS TO LABEL MASK ===
img = utils.img_b64_to_arr(imageData)

# === Handle the shapes ===
shapes = data.get("shapes", [])
lbl_names = []
# Convert shapes to labels, focusing on polygons
labels = np.zeros(img.shape[:2], dtype=np.uint8)  # Create an empty mask (background)

for i, shape in enumerate(shapes):
    label = shape["label"]  # Get label name
    lbl_names.append(label)  # Add label name to lbl_names list
    points = np.array(shape["points"], dtype=np.int32)  # Convert polygon points to numpy array

    print(f"Shape {i}: Label = {label}, Points = {points}")  # Debugging info to verify points

    # Fill the polygon area with label id (1 for disease)
    cv2.fillPoly(labels, [points], 1)  # Note: Use '1' for disease label (you can change if needed)

# Debugging: Check if the mask is being filled
if np.sum(labels) == 0:
    print("Warning: The label mask is empty. No areas are being filled.")
else:
    print("Label mask has been filled with non-zero values.")

# === SAVE OUTPUT ===
os.makedirs(output_dir, exist_ok=True)
PIL.Image.fromarray(img).save(os.path.join(output_dir, "img.png"))
PIL.Image.fromarray(labels.astype(np.uint8)).save(os.path.join(output_dir, "label.png"))

# Save label names (for mapping purposes)
with open(os.path.join(output_dir, "label_names.txt"), "w") as f:
    for name in lbl_names:
        f.write(f"{name}\n")

print(f"✅ Saved to: {output_dir}")
'''
'''
import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import jaccard_score

# === CONFIG ===
img_size = 224
mask_folder = r"D:\CS\AI\PROJECT\ML_agri project\new_models\disease_detection\masks"
cam_root = r"D:\CS\AI\PROJECT\ML_agri project\new_models\disease_detection\cams"
cam_types = ['grad_cam', 'eigen_cam', 'layer_cam']
DEBUG_VISUALS = True     # Show mask + CAM overlays
USE_FIXED_THRESHOLD = True
CAM_THRESHOLD = 0.3      # Used if USE_FIXED_THRESHOLD = True
DILATE_MASK = False      # Optional: dilate to tolerate spatial offset

# === IoU FUNCTION ===
def compute_iou(mask, cam, debug=False):
    # Resize and binarize ground truth mask
    mask = cv2.resize(mask, (img_size, img_size), interpolation=cv2.INTER_NEAREST)
    mask_binary = (mask > 0).astype(np.uint8)

    # Optional: dilate the mask to allow looser overlap (tunable)
    if DILATE_MASK:
        kernel = np.ones((5, 5), np.uint8)
        mask_binary = cv2.dilate(mask_binary, kernel, iterations=1)

    # Convert CAM to grayscale and normalize
    cam_gray = cv2.cvtColor(cam, cv2.COLOR_BGR2GRAY)
    cam_gray = cv2.resize(cam_gray, (img_size, img_size), interpolation=cv2.INTER_CUBIC)
    cam_gray = cam_gray / 255.0

    # Threshold CAM
    if USE_FIXED_THRESHOLD:
        cam_binary = (cam_gray >= CAM_THRESHOLD).astype(np.uint8)
        thresh_val = CAM_THRESHOLD
    else:
        thresh_val = np.percentile(cam_gray, 75)
        cam_binary = (cam_gray >= thresh_val).astype(np.uint8)

    # Optional: dilate CAM for tolerance
    if DILATE_MASK:
        kernel = np.ones((5, 5), np.uint8)
        cam_binary = cv2.dilate(cam_binary, kernel, iterations=1)

    # Debug and visualization
    if debug:
        print(f"Mask unique values: {np.unique(mask)}")
        print(f"CAM min/max: {cam_gray.min():.3f}/{cam_gray.max():.3f}")
        print(f"Threshold value used: {thresh_val:.3f}")
        visualize(mask_binary, cam_gray, cam_binary)

    # Flatten both
    gt = mask_binary.flatten()
    pred = cam_binary.flatten()

    # Handle edge cases
    if np.sum(gt) == 0 and np.sum(pred) == 0:
        return 1.0
    if np.sum(gt) == 0 or np.sum(pred) == 0:
        return 0.0

    return jaccard_score(gt, pred)

# === VISUALIZER ===
def visualize(mask_binary, cam_gray, cam_binary):
    plt.figure(figsize=(15, 4))

    plt.subplot(1, 4, 1)
    plt.imshow(mask_binary, cmap='gray')
    plt.title('GT Mask')
    plt.axis('off')

    plt.subplot(1, 4, 2)
    plt.imshow(cam_gray, cmap='jet')
    plt.title('CAM Grayscale')
    plt.axis('off')

    plt.subplot(1, 4, 3)
    plt.imshow(cam_binary, cmap='gray')
    plt.title('CAM Binary')
    plt.axis('off')

    plt.subplot(1, 4, 4)
    # Red: mask, Green: CAM, Yellow: overlap
    overlay = np.zeros((img_size, img_size, 3), dtype=np.uint8)
    overlay[..., 1] = cam_binary * 255  # Green = CAM
    overlay[..., 0] = mask_binary * 255  # Red = GT mask
    plt.imshow(overlay)
    plt.title('Overlay: Mask (Red), CAM (Green)')
    plt.axis('off')

    plt.tight_layout()
    plt.show()

# === MAIN EVALUATION LOOP ===
results = {}

mask_files = sorted([f for f in os.listdir(mask_folder) if f.endswith(".png")])
for mask_file in mask_files:
    image_id = os.path.splitext(mask_file)[0].replace("_mask", "")
    results[image_id] = {}

    mask_path = os.path.join(mask_folder, mask_file)
    gt_mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)

    for cam_type in cam_types:
        cam_path = os.path.join(cam_root, cam_type, f"{image_id}_cam.png")
        if not os.path.exists(cam_path):
            print(f"[!] Missing CAM for {image_id} in {cam_type}")
            results[image_id][cam_type] = None
            continue

        cam_img = cv2.imread(cam_path)
        iou = compute_iou(gt_mask, cam_img, debug=DEBUG_VISUALS)
        results[image_id][cam_type] = iou
        print(f"{image_id} | {cam_type} | IoU: {iou:.3f}")

# === SUMMARY REPORT ===
print("\n🧾 Average IoU Scores:")
for cam_type in cam_types:
    ious = [v[cam_type] for v in results.values() if v[cam_type] is not None]
    avg = np.mean(ious) if ious else 0.0
    print(f"{cam_type}: {avg:.3f}")
from sklearn.metrics import jaccard_score, f1_score

def compute_metrics(mask, cam, debug=False):
    # Resize and binarize mask
    mask = cv2.resize(mask, (img_size, img_size), interpolation=cv2.INTER_NEAREST)
    mask_binary = (mask > 0).astype(np.uint8)

    # CAM to grayscale
    cam_gray = cv2.cvtColor(cam, cv2.COLOR_BGR2GRAY)
    cam_gray = cv2.resize(cam_gray, (img_size, img_size), interpolation=cv2.INTER_CUBIC)
    cam_gray = cam_gray / 255.0

    # Threshold CAM
    cam_binary = (cam_gray >= CAM_THRESHOLD).astype(np.uint8)

    if debug:
        print(f"Dice - CAM min/max: {cam_gray.min():.3f}/{cam_gray.max():.3f}")
        visualize(mask_binary, cam_gray, cam_binary)

    gt = mask_binary.flatten()
    pred = cam_binary.flatten()

    if np.sum(gt) == 0 and np.sum(pred) == 0:
        return 1.0, 1.0
    if np.sum(gt) == 0 or np.sum(pred) == 0:
        return 0.0, 0.0

    iou = jaccard_score(gt, pred)
    dice = f1_score(gt, pred)
    return iou, dice
'''
'''
import os
import torch
import timm
import numpy as np
import cv2
from PIL import Image
from torchvision import transforms
from pytorch_grad_cam import GradCAM, EigenCAM, LayerCAM
from pytorch_grad_cam.utils.image import show_cam_on_image
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget

# === CONFIG ===
img_size = 224
device = 'cuda' if torch.cuda.is_available() else 'cpu'
model_path = r"D:\CS\AI\PROJECT\ML_agri project\new_models\disease_detection\convnext_disease.pth"
image_folder = r"D:\CS\AI\PROJECT\ML_agri project\new_models\disease_detection\images"  # original leaf images

# === OUTPUT FOLDERS ===
output_root = r"D:\CS\AI\PROJECT\ML_agri project\new_models\disease_detection\cams"
os.makedirs(output_root, exist_ok=True)
cam_methods = {
    'grad_cam': GradCAM,
    'eigen_cam': EigenCAM,
    'layer_cam': LayerCAM
}
for cam_type in cam_methods:
    os.makedirs(os.path.join(output_root, cam_type), exist_ok=True)

# === MODEL ===
model = timm.create_model('convnext_tiny', pretrained=False, num_classes=8)
model.load_state_dict(torch.load(model_path, map_location=device))
model.eval().to(device)
target_layers = [model.stages[-1].blocks[-1].norm]

# === TRANSFORM ===
transform = transforms.Compose([
    transforms.Resize((img_size, img_size)),
    transforms.ToTensor(),
    transforms.Normalize([0.5]*3, [0.5]*3)
])

# === LOOP OVER IMAGES ===
for filename in os.listdir(image_folder):
    if not filename.lower().endswith(('.jpg', '.jpeg', '.png')):
        continue

    image_path = os.path.join(image_folder, filename)
    pil_img = Image.open(image_path).convert('RGB')
    rgb_img = np.array(pil_img.resize((img_size, img_size))) / 255.0
    input_tensor = transform(pil_img).unsqueeze(0).to(device)

    with torch.no_grad():
        outputs = model(input_tensor)
        pred_class = torch.argmax(outputs, dim=1).item()

    for name, cam_cls in cam_methods.items():
        cam = cam_cls(model=model, target_layers=target_layers)
        grayscale_cam = cam(input_tensor=input_tensor, targets=[ClassifierOutputTarget(pred_class)])
        heatmap = show_cam_on_image(rgb_img, grayscale_cam[0], use_rgb=True)

        save_path = os.path.join(output_root, name, f"{os.path.splitext(filename)[0]}_cam.png")
        cv2.imwrite(save_path, cv2.cvtColor(heatmap, cv2.COLOR_RGB2BGR))

    print(f"[✓] Saved CAMs for {filename}")
'''
'''
import os
import time
import torch
import timm
import numpy as np
import cv2
from PIL import Image
from torchvision import transforms
from pytorch_grad_cam import GradCAM, EigenCAM, LayerCAM
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget

# === CONFIG ===
image_folder = r"D:\CS\AI\PROJECT\ML_agri project\new_models\disease_detection\images"
model_path = r"D:\CS\AI\PROJECT\ML_agri project\new_models\disease_detection\convnext_disease.pth"
img_size = 224
device = 'cuda' if torch.cuda.is_available() else 'cpu'
threshold_percentile = 75

# === MODEL ===
model = timm.create_model('convnext_tiny', pretrained=False, num_classes=8)
model.load_state_dict(torch.load(model_path, map_location=device))
model.eval().to(device)
target_layers = [model.stages[-1].blocks[-1].norm]

# === TRANSFORM ===
transform = transforms.Compose([
    transforms.Resize((img_size, img_size)),
    transforms.ToTensor(),
    transforms.Normalize([0.5]*3, [0.5]*3)
])

# === CAM METHODS ===
cam_methods = {
    "Grad-CAM": GradCAM,
    "Eigen-CAM": EigenCAM,
    "Layer-CAM": LayerCAM
}

# === RESULT HOLDERS ===
fidelity_scores = {name: [] for name in cam_methods}
timing = {name: [] for name in cam_methods}

# === LOOP THROUGH IMAGES ===
image_files = sorted([f for f in os.listdir(image_folder) if f.endswith(('.png', '.jpg', '.jpeg'))])

for filename in image_files:
    print(f"\n🖼️ Processing: {filename}")
    img_path = os.path.join(image_folder, filename)

    pil_img = Image.open(img_path).convert("RGB")
    original_np = np.array(pil_img.resize((img_size, img_size))) / 255.0
    input_tensor = transform(pil_img).unsqueeze(0).to(device)

    # Original prediction and confidence
    with torch.no_grad():
        output = model(input_tensor)
        pred_class = torch.argmax(output, dim=1).item()
        original_conf = torch.softmax(output, dim=1)[0][pred_class].item()

    for name, cam_class in cam_methods.items():
        cam = cam_class(model=model, target_layers=target_layers)

        # Measure time
        start = time.perf_counter()
        grayscale_cam = cam(input_tensor=input_tensor, targets=[ClassifierOutputTarget(pred_class)])[0]
        end = time.perf_counter()
        elapsed = end - start
        timing[name].append(elapsed)

        # Mask out CAM region
        cam_mask = (grayscale_cam >= np.percentile(grayscale_cam, threshold_percentile)).astype(np.uint8)
        masked_img = np.array(original_np * (1 - cam_mask[..., None]))  # RGB mask
        masked_img = (masked_img * 255).astype(np.uint8)
        masked_pil = Image.fromarray(masked_img)
        masked_tensor = transform(masked_pil).unsqueeze(0).to(device)

        # Get masked confidence
        with torch.no_grad():
            masked_output = model(masked_tensor)
            masked_conf = torch.softmax(masked_output, dim=1)[0][pred_class].item()

        fidelity = original_conf - masked_conf
        fidelity_scores[name].append(fidelity)

        print(f"{name:<10} | Time: {elapsed:.4f}s | Fidelity: {fidelity:.4f}")

# === AVERAGE RESULTS ===
print("\n📊 Average Computational Time and Fidelity:")
for name in cam_methods:
    avg_time = np.mean(timing[name])
    avg_fidelity = np.mean(fidelity_scores[name])
    print(f"{name:<10} | Avg Time: {avg_time:.4f}s | Avg Fidelity: {avg_fidelity:.4f}")
'''