from PIL import Image, ImageDraw, ImageFont
import numpy as np
import torchvision.transforms.functional as TF

def visualize_predictions_pil_overlay(image_tensor, pred_mask, seg_probs, class_names):
    image = TF.to_pil_image(image_tensor).convert("RGBA")
    width, height = image.size

    # Soft, pastel-style RGBA colors
    pastel_colors = {
        1: (230, 200, 100, 80),   # Boiled Egg - soft golden yellow
        2: (140, 200, 140, 80),   # Chayote - earthy green
        3: (210, 160, 120, 80),   # Chicken - warm beige
        4: (255, 240, 140, 80),   # Egg Sunny Side Up - light yolk yellow
        5: (100, 160, 100, 80),   # Green Leaf Vegetable - forest green
        6: (240, 210, 150, 80),   # Pasta - golden tan
        7: (170, 100, 70, 80),    # Pork - muted reddish brown
        8: (200, 160, 90, 80),    # Potato - golden brown
        9: (240, 240, 230, 80),   # Rice - subtle ivory white
        10: (255, 220, 130, 80),  # Scrambled Egg - soft amber yellow
    }

    overlay = Image.new("RGBA", (width, height))
    pred_mask_np = pred_mask.numpy()
    seg_probs_np = seg_probs.numpy()
    unique_classes = np.unique(pred_mask_np)

    for cls_id in unique_classes:
        if cls_id == 0:
            continue
        mask = (pred_mask_np == cls_id)
        if mask.sum() == 0:
            continue
        color = pastel_colors.get(cls_id, (200, 200, 200, 80))  # fallback light gray
        rgba = np.zeros((height, width, 4), dtype=np.uint8)
        rgba[mask] = color
        overlay = Image.alpha_composite(overlay, Image.fromarray(rgba, mode="RGBA"))

    blended = Image.alpha_composite(image, overlay)

    # Prepare labels
    try:
        font = ImageFont.truetype("arial.ttf", 16)
    except IOError:
        font = ImageFont.load_default()

    label_items = []
    for cls_id in unique_classes:
        if cls_id == 0:
            continue
        mask = (pred_mask_np == cls_id)
        if mask.sum() == 0:
            continue
        conf = seg_probs_np[cls_id][mask].mean()
        label = f"{class_names.get(cls_id, f'Class {cls_id}')} ({conf:.2f})"
        color = pastel_colors.get(cls_id, (220, 220, 220, 100))[:3]
        label_items.append((label, color))

    # Wrapping layout
    box_size = 20
    spacing = 10
    padding = 10
    line_height = box_size + 5
    max_width = width - 2 * padding

    x = 0
    rows = 1
    for text, _ in label_items:
        text_width = font.getlength(text)
        item_width = box_size + spacing + text_width + spacing
        if x + item_width > max_width:
            rows += 1
            x = 0
        x += item_width

    footer_height = rows * line_height + 2 * padding
    final = Image.new("RGB", (width, height + footer_height), (255, 255, 255))
    final.paste(blended.convert("RGB"), (0, 0))
    draw = ImageDraw.Draw(final)

    # Draw labels
    x_offset = padding
    y_offset = height + padding

    for label, color in label_items:
        text_width = font.getlength(label)
        item_width = box_size + spacing + text_width + spacing

        if x_offset + item_width > width - padding:
            x_offset = padding
            y_offset += line_height

        draw.rectangle(
            [x_offset, y_offset, x_offset + box_size, y_offset + box_size],
            fill=color
        )
        draw.text(
            (x_offset + box_size + spacing, y_offset),
            label,
            fill=(0, 0, 0),
            font=font
        )
        x_offset += item_width

    return final
