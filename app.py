# --- app.py ---
import os
import io
import time
import torch
import numpy as np
from PIL import Image
from flask import Flask, request, render_template, send_file, jsonify
from torchvision import transforms
import matplotlib
matplotlib.use('Agg')  # Use a non-GUI backend
import matplotlib.pyplot as plt

from model.hybrid_model import HybridGCNGAN
from model.utils import visualize_predictions_with_overlay_single_only

# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Class names (index 0 is background)
class_names = [
    "__background__",
    "Boiled Egg", "Chayote", "Chicken", "Egg Sunny Side Up",
    "Green Leaf Vegetable", "Pasta", "Pork", "Potato", "Rice", "Scrambled Egg"
]
num_classes = len(class_names)
skip_channels = 24

# Model checkpoint path
checkpoint_path = "bestFinal_hybridModel.pth"
model = None  # Global instance

# Model loading function
def load_model():
    global model
    if model is None:
        print("Loading model...")
        model_instance = HybridGCNGAN(num_classes=num_classes, skip_channels=skip_channels)
        checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
        model_instance.load_state_dict(checkpoint["model_state_dict"])
        print(f"Loaded checkpoint from epoch {checkpoint['epoch'] + 1} | mIoU: {checkpoint['best_miou']:.4f}, Acc: {checkpoint['best_acc']:.4f}")
        model_instance.to(device).eval()
        model = model_instance
    return model

# Image transform
transform = transforms.Compose([
    transforms.Resize((320, 320)),
    transforms.ToTensor(),
])

# Initialize Flask app
app = Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload():
    if 'file' not in request.files:
        return jsonify({"status": "error", "message": "No file provided"}), 400

    file = request.files['file']
    if file.filename == '':
        return jsonify({"status": "error", "message": "No file selected"}), 400

    image = Image.open(file.stream).convert('RGB')
    image_tensor = transform(image).unsqueeze(0).to(device)

    model_instance = load_model()

    # Inference
    start_time = time.time()
    with torch.no_grad():
        pred = model_instance(image_tensor)
        seg_probs = torch.softmax(pred, dim=1).cpu().numpy()[0]  # [C, H, W]
        pred_mask = seg_probs.argmax(axis=0)  # [H, W]
    print("UPLOAD prediction took", round(time.time() - start_time, 2), "seconds")

    # Visualization
    fig = visualize_predictions_with_overlay_single_only(
        image_tensor.squeeze(0).cpu(),
        torch.zeros_like(torch.from_numpy(pred_mask)),  # placeholder for gt_mask
        torch.from_numpy(pred_mask),
        torch.from_numpy(seg_probs),
        class_names
    )

    buf = io.BytesIO()
    fig.savefig(buf, format='png', bbox_inches='tight', dpi=100)
    buf.seek(0)
    plt.close(fig)

    return send_file(buf, mimetype='image/png')

@app.route('/predict_live', methods=['POST'])
def predict_live():
    if 'frame' not in request.files:
        return "No frame uploaded", 400

    file = request.files['frame']
    image = Image.open(file.stream).convert('RGB')
    image_tensor = transform(image).unsqueeze(0).to(device)

    model_instance = load_model()

    start_time = time.time()
    with torch.no_grad():
        pred = model_instance(image_tensor)
        seg_probs = torch.softmax(pred, dim=1).cpu().numpy()[0]
        pred_mask = seg_probs.argmax(axis=0)
    print("LIVE prediction took", round(time.time() - start_time, 2), "seconds")

    fig = visualize_predictions_with_overlay_single_only(
        image_tensor.squeeze(0).cpu(),
        torch.zeros_like(torch.from_numpy(pred_mask)),
        torch.from_numpy(pred_mask),
        torch.from_numpy(seg_probs),
        class_names
    )

    buf = io.BytesIO()
    fig.savefig(buf, format='png', bbox_inches='tight', dpi=100)
    buf.seek(0)
    plt.close(fig)

    return send_file(buf, mimetype='image/png')

if __name__ == '__main__':
    app.run(debug=True)
