'''
import torch
import timm
import cv2
import numpy as np
import matplotlib.pyplot as plt
from torchvision import transforms
from PIL import Image
from pytorch_grad_cam import GradCAM, EigenCAM, LayerCAM
from pytorch_grad_cam.utils.image import show_cam_on_image
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget


# === CONFIG ===
model_name = 'convnext_tiny'
model_path = r
num_classes = 8  # Adjust as needed
img_size = 224
device = 'cuda' if torch.cuda.is_available() else 'cpu'

# === TRANSFORMS ===
transform = transforms.Compose([
    transforms.Resize((img_size, img_size)),
    transforms.ToTensor(),
    transforms.Normalize([0.5]*3, [0.5]*3)
])

# === LOAD MODEL ===
model = timm.create_model(model_name, pretrained=False, num_classes=num_classes)
model.load_state_dict(torch.load(model_path, map_location=device))
model.eval().to(device)

# === LOAD IMAGE ===
img_path =  # replace with your own
pil_img = Image.open(img_path).convert('RGB')
input_tensor = transform(pil_img).unsqueeze(0).to(device)

# === Preprocessing for visualization ===
rgb_img = np.array(pil_img.resize((img_size, img_size))) / 255.0

# === GET TARGET LAYER FOR CONVNEXT ===
# Usually the last feature block
target_layers = [model.stages[-1].blocks[-1].norm]

# === DEFINE CAM METHODS ===
methods = {
    "Grad-CAM": GradCAM(model=model, target_layers=target_layers, use_cuda=device=='cuda'),
    "Eigen-CAM": EigenCAM(model=model, target_layers=target_layers, use_cuda=device=='cuda'),
    "Layer-CAM": LayerCAM(model=model, target_layers=target_layers, use_cuda=device=='cuda'),
}

# === PREDICTION ===
with torch.no_grad():
    outputs = model(input_tensor)
    pred_class = outputs.argmax(dim=1).item()

# === RUN CAMS ===
for name, cam_method in methods.items():
    grayscale_cam = cam_method(input_tensor=input_tensor, targets=[ClassifierOutputTarget(pred_class)])
    cam_image = show_cam_on_image(rgb_img, grayscale_cam[0], use_rgb=True)

    # Show and save
    plt.figure(figsize=(5, 5))
    plt.imshow(cam_image)
    plt.title(f"{name} — Predicted: {pred_class}")
    plt.axis('off')
    plt.tight_layout()
    plt.savefig(f"{name.replace('-', '_').lower()}_cam_output.png")
    plt.show()
'''
import torch
import timm
import numpy as np
import matplotlib.pyplot as plt
from torchvision import transforms
from PIL import Image
import time
import psutil
from pytorch_grad_cam import GradCAM, EigenCAM, LayerCAM
from pytorch_grad_cam.utils.image import show_cam_on_image
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
from ptflops import get_model_complexity_info

# === CONFIG ===
model_name = 'convnext_tiny'
model_path = r"D:\CS\AI\PROJECT\ML_agri project\new_models\disease_detection\convnext_disease.pth"
num_classes = 8
img_size = 224
device = 'cpu'  # Using CPU for this comparison

# === TRANSFORMS ===
transform = transforms.Compose([
    transforms.Resize((img_size, img_size)),
    transforms.ToTensor(),
    transforms.Normalize([0.5]*3, [0.5]*3)
])

# === LOAD MODEL ===
model = timm.create_model(model_name, pretrained=False, num_classes=num_classes)
model.load_state_dict(torch.load(model_path, map_location=device))
model.eval().to(device)

# === LOAD IMAGE ===
img_path = r"C:\Users\sabin\Downloads\mango_disease test.jpg"  # Replace with your image path
pil_img = Image.open(img_path).convert('RGB')
input_tensor = transform(pil_img).unsqueeze(0).to(device)

# === Preprocessing for visualization ===
rgb_img = np.array(pil_img.resize((img_size, img_size))) / 255.0

# === GET TARGET LAYER FOR CONVNEXT ===
target_layers = [model.stages[-1].blocks[-1].norm]

# === DEFINE CAM METHODS ===
methods = {
    "Grad-CAM": GradCAM(model=model, target_layers=target_layers),
    "Eigen-CAM": EigenCAM(model=model, target_layers=target_layers),
    "Layer-CAM": LayerCAM(model=model, target_layers=target_layers),
}

# === TRACKING FLOPs and Parameters ===
macs, params = get_model_complexity_info(
    model, (3, img_size, img_size), as_strings=True, print_per_layer_stat=False
)
print(f"FLOPs: {macs}, Parameters: {params}")

# === TRACKING MEMORY USAGE ===
process = psutil.Process()

# === TRACKING TIME FOR CAM METHODS ===
cam_times = {}

for name, cam_method in methods.items():
    # Start timer for CAM computation
    start_time = time.time()

    # Track memory usage before CAM computation
    initial_memory = process.memory_info().rss / 1024**2  # in MB

    # Run CAM computation
    grayscale_cam = cam_method(input_tensor=input_tensor, targets=[ClassifierOutputTarget(0)])

    # Track memory usage after CAM computation
    final_memory = process.memory_info().rss / 1024**2  # in MB

    # End timer for CAM computation
    end_time = time.time()

    # Calculate time taken
    cam_time = end_time - start_time
    cam_times[name] = cam_time

    # Calculate memory usage increase
    memory_used = final_memory - initial_memory

    print(f"{name} - Time: {cam_time:.4f}s, Memory Used: {memory_used:.2f}MB")

    # Visualize CAM
    cam_image = show_cam_on_image(rgb_img, grayscale_cam[0], use_rgb=True)
    plt.figure(figsize=(5, 5))
    plt.imshow(cam_image)
    plt.title(f"{name} — Predicted: {0}")  # Assuming class 0 for this example
    plt.axis('off')
    plt.tight_layout()
    plt.show()

# === Final Comparison Summary ===
print("\n===== CAM Methods Comparison =====")
for name, cam_time in cam_times.items():
    print(f"{name} - Time: {cam_time:.4f} seconds")


