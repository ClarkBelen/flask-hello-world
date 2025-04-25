# --- app.py ---
import os
import io
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
from model.utils import visualize_predictions_with_overlay_single_only

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

# Load model
model = HybridGCNGAN(num_classes=num_classes, skip_channels=skip_channels)
checkpoint_path = "bestFinal_hybridModel.pth"

if not hasattr(np, "scalar"):
    np.scalar = np.generic

# torch.serialization.add_safe_globals([np.scalar])
checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)

model.load_state_dict(checkpoint["model_state_dict"])
start_epoch = checkpoint["epoch"] + 1
best_miou = checkpoint["best_miou"]
best_acc = checkpoint["best_acc"]
print(f"Best training from epoch {start_epoch} with best mIoU: {best_miou:.4f}, best accuracy: {best_acc:.4f}")
    
model.to(device)
model.eval()

# Transform
transform = transforms.Compose([
    transforms.Resize((480, 480)),
    transforms.ToTensor(),
])

# Flask app
app = Flask(__name__)
# os.makedirs("static", exist_ok=True)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
# def upload():
#     if 'file' not in request.files:
#         return jsonify({"status": "error", "message": "No file provided"}), 400

#     file = request.files['file']
#     if file.filename == '':
#         return jsonify({"status": "error", "message": "No file selected"}), 400

#     image = Image.open(file.stream).convert('RGB')
#     image_tensor = transform(image).unsqueeze(0).to(device)

#     with torch.no_grad():
#         pred = model(image_tensor)
#         seg_probs = torch.softmax(pred, dim=1).cpu().numpy()[0]
#         pred_mask = seg_probs.argmax(axis=0)

#     fig = visualize_predictions_with_overlay_single_only(
#         image_tensor.squeeze(0).cpu(),
#         torch.zeros_like(torch.from_numpy(pred_mask)),
#         torch.from_numpy(pred_mask),
#         torch.from_numpy(seg_probs),
#         class_idx_to_name
#     )

#     output_path = os.path.join('static', 'predicted_result.png')
#     fig.savefig(output_path, format='png', bbox_inches='tight')
#     plt.close(fig)

#     return jsonify({"status": "success", "prediction_path": "/static/predicted_result.png"})
@app.route('/upload', methods=['POST'])
def upload():
    if 'file' not in request.files:
        return jsonify({"status": "error", "message": "No file provided"}), 400

    file = request.files['file']
    if file.filename == '':
        return jsonify({"status": "error", "message": "No file selected"}), 400

    image = Image.open(file.stream).convert('RGB')
    image_tensor = transform(image).unsqueeze(0).to(device)

    with torch.no_grad():
        pred = model(image_tensor)
        seg_probs = torch.softmax(pred, dim=1).cpu().numpy()[0]
        pred_mask = seg_probs.argmax(axis=0)

    fig = visualize_predictions_with_overlay_single_only(
        image_tensor.squeeze(0).cpu(),
        torch.zeros_like(torch.from_numpy(pred_mask)),
        torch.from_numpy(pred_mask),
        torch.from_numpy(seg_probs),
        class_idx_to_name
    )

    # Instead of saving to static folder, send directly to client
    buf = io.BytesIO()
    fig.savefig(buf, format='png', bbox_inches='tight')
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

    with torch.no_grad():
        pred = model(image_tensor)
        seg_probs = torch.softmax(pred, dim=1).cpu().numpy()[0]
        pred_mask = seg_probs.argmax(axis=0)

    fig = visualize_predictions_with_overlay_single_only(
        image_tensor.squeeze(0).cpu(),
        torch.zeros_like(torch.from_numpy(pred_mask)),
        torch.from_numpy(pred_mask),
        torch.from_numpy(seg_probs),
        class_idx_to_name
    )

    buf = io.BytesIO()
    fig.savefig(buf, format='png', bbox_inches='tight')
    buf.seek(0)
    plt.close(fig)

    return send_file(buf, mimetype='image/png')

if __name__ == '__main__':
    app.run(debug=True)
