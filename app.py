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
import matplotlib.patches as mpatches
from model.hybrid_model import HybridGCNGAN
from model.utils import visualize_predictions_pil_overlay

# Set up device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Define number of classes
num_classes = 11
class_idx_to_name = {
    1: "Boiled Egg",
    2: "Chayote",
    3: "Chicken",
    4: "Egg Sunny Side Up",
    5: "Green Leaf Vegetable",
    6: "Pasta",
    7: "Pork",
    8: "Potato",
    9: "Rice",
    10: "Scrambled Egg"
}
skip_channels = 24

# Lazy-loaded model
model = None
checkpoint_path = "bestFinal_hybridModel.pth"

def load_model():
    global model
    if model is None:
        print("Loading model...")
        model_instance = HybridGCNGAN(num_classes=num_classes, skip_channels=skip_channels)
        if not hasattr(np, "scalar"):
            np.scalar = np.generic
        checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
        model_instance.load_state_dict(checkpoint["model_state_dict"])
        print(f"Best training from epoch {checkpoint['epoch'] + 1} with best mIoU: {checkpoint['best_miou']:.4f}, best accuracy: {checkpoint['best_acc']:.4f}")
        model_instance.to(device)
        model_instance.eval()
        model = model_instance
    return model

# Transform
transform = transforms.Compose([
    transforms.Resize((320, 320)),
    transforms.ToTensor(),
])

# Flask app
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

    start_time = time.time()
    with torch.no_grad():
        pred = model_instance(image_tensor)
        seg_probs = torch.softmax(pred, dim=1).cpu().numpy()[0]
        pred_mask = seg_probs.argmax(axis=0)
    print("UPLOAD prediction took", round(time.time() - start_time, 2), "seconds")

    overlay_img =  visualize_predictions_pil_overlay(
        image_tensor.squeeze(0).cpu(),
        torch.from_numpy(pred_mask),
        torch.from_numpy(seg_probs),
        class_idx_to_name
    )

    buf = io.BytesIO()
    # fig.savefig(buf, format='png', bbox_inches='tight', dpi=80)
    overlay_img.save(buf, format='PNG')
    buf.seek(0)
    # plt.close(fig)

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

    overlay_img =  visualize_predictions_pil_overlay(
        image_tensor.squeeze(0).cpu(),
        torch.from_numpy(pred_mask),
        torch.from_numpy(seg_probs),
        class_idx_to_name
    )

    buf = io.BytesIO()
    # fig.savefig(buf, format='png', bbox_inches='tight', dpi=80)
    overlay_img.save(buf, format='PNG')
    buf.seek(0)
    # plt.close(fig)

    return send_file(buf, mimetype='image/png')

if __name__ == '__main__':
    app.run(debug=True)
