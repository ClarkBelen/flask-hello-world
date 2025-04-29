# --- utils.py ---
import matplotlib
matplotlib.use('Agg')  # Use a non-GUI backend
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np

def visualize_predictions_with_overlay_single_only(image_tensor, gt_mask, pred_mask, seg_probs, class_names):
    num_classes = len(class_names)
    cmap = plt.get_cmap('tab20', num_classes)

    image = image_tensor.permute(1, 2, 0).cpu().numpy()
    pred_mask = pred_mask.cpu().numpy()
    gt_mask = gt_mask.cpu().numpy()
    seg_probs = seg_probs.cpu().numpy()

    unique_classes = np.unique(pred_mask)
    legend_entries = []
    for cls_id in unique_classes:
        if cls_id == 0:  # Skip background
            continue
        class_name = class_names[cls_id]
        class_mask = (pred_mask == cls_id)
        if class_mask.sum() == 0:
            continue
        class_conf = seg_probs[cls_id][class_mask].mean().item()
        label_str = f"{class_name} ({class_conf:.2f})"
        patch = mpatches.Patch(color=cmap(cls_id), label=label_str)
        legend_entries.append(patch)

    fig, axs = plt.subplots(1, 1, figsize=(6, 6))
    axs.imshow(image, alpha=0.7)
    axs.imshow(pred_mask, cmap=cmap, alpha=0.5, vmin=0, vmax=num_classes - 1)
    axs.set_title("Predicted Overlay + Confidence")
    axs.axis('off')

    fig.legend(handles=legend_entries, loc='lower center', ncol=3, bbox_to_anchor=(0.5, -0.05))
    plt.tight_layout()

    return fig
