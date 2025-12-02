import os
import random
from collections import Counter

import numpy as np
import pandas as pd
import sys
import matplotlib.pyplot as plt
from PIL import Image


# ================== CONFIG ==================
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(CURRENT_DIR)
sys.path.append(PROJECT_ROOT)

IMAGE_DIR = os.path.join(PROJECT_ROOT, "data", "training")
CSV_PATH = os.path.join(IMAGE_DIR, "training_ground_truth.csv")

OUTPUT_DIR = os.path.join(CURRENT_DIR, "output")
os.makedirs(OUTPUT_DIR, exist_ok=True)

# How many images to use for the pixel-intensity histogram & sample grid
SAMPLE_FOR_HIST = 200
NUM_ROWS = 5
NUM_COLS = 5


# ================== LOAD METADATA ==================

df = pd.read_csv(CSV_PATH)

# Some versions name the column "image" instead of "image_id"
if "image" in df.columns and "image_id" not in df.columns:
    df = df.rename(columns={"image": "image_id"})

class_cols = [c for c in df.columns if c != "image_id"]
num_classes = len(class_cols)

num_labeled = len(df)

# ================== CHECK FILES ==================

expected_files = set(df["image_id"].astype(str) + ".jpg")
actual_files = {f for f in os.listdir(IMAGE_DIR) if f.lower().endswith(".jpg")}

missing_images = sorted(expected_files - actual_files)
extra_images = sorted(actual_files - expected_files)

num_missing = len(missing_images)
num_extra = len(extra_images)
num_image_files = len(actual_files)


# ================== CLASS DISTRIBUTION ==================

# Each class column is one-hot; sum gives counts
class_counts = df[class_cols].sum().astype(int)
class_percents = class_counts / class_counts.sum() * 100

# Plot class distribution
plt.figure(figsize=(8, 5))
plt.bar(class_cols, class_counts.values)
plt.xticks(rotation=45, ha="right")
plt.ylabel("Number of images")
plt.title("HAM10000 Class Distribution")
plt.tight_layout()
class_dist_path = os.path.join(OUTPUT_DIR, "class_distribution.png")
plt.savefig(class_dist_path)
plt.close()


# ================== IMAGE & PIXEL STATS ==================

image_sizes = Counter()
pixel_sum = 0.0
pixel_sq_sum = 0.0
pixel_min = float("inf")
pixel_max = float("-inf")
total_pixels = 0

image_ids = df["image_id"].tolist()

# For histogram, sample a subset of images
sample_ids_for_hist = random.sample(image_ids, min(SAMPLE_FOR_HIST, len(image_ids)))
pixel_values_for_hist = []

for idx, image_id in enumerate(image_ids):
    img_path = os.path.join(IMAGE_DIR, image_id + ".jpg")

    if not os.path.exists(img_path):
        continue

    img = Image.open(img_path).convert("RGB")
    np_img = np.asarray(img, dtype=np.float32) / 255.0  # normalize 0–1

    h, w, c = np_img.shape
    image_sizes[(h, w)] += 1

    # Flatten for stats
    flat = np_img.reshape(-1)

    total_pixels += flat.size
    pixel_sum += flat.sum()
    pixel_sq_sum += (flat ** 2).sum()
    pixel_min = min(pixel_min, flat.min())
    pixel_max = max(pixel_max, flat.max())

    if image_id in sample_ids_for_hist:
        pixel_values_for_hist.append(flat)

# Combine sampled pixel values
if pixel_values_for_hist:
    pixel_values_for_hist = np.concatenate(pixel_values_for_hist)
else:
    pixel_values_for_hist = np.array([])

# Compute mean/std
if total_pixels > 0:
    pixel_mean = pixel_sum / total_pixels
    pixel_var = pixel_sq_sum / total_pixels - pixel_mean ** 2
    pixel_std = float(np.sqrt(max(pixel_var, 0.0)))
else:
    pixel_mean = pixel_std = pixel_min = pixel_max = float("nan")

# Pixel histogram from sampled images
if pixel_values_for_hist.size > 0:
    plt.figure(figsize=(8, 5))
    plt.hist(pixel_values_for_hist, bins=50, alpha=0.7, edgecolor="black")
    plt.xlabel("Pixel Intensity (0–1)")
    plt.ylabel("Frequency")
    plt.title(f"Pixel Intensity Distribution (sample of {len(sample_ids_for_hist)} images)")
    plt.tight_layout()
    pixel_hist_path = os.path.join(OUTPUT_DIR, "pixel_histogram.png")
    plt.savefig(pixel_hist_path)
    plt.close()
else:
    pixel_hist_path = "No histogram (no pixels collected)."


# ================== SAMPLE IMAGE GRID ==================

sample_ids_for_grid = random.sample(image_ids, min(NUM_ROWS * NUM_COLS, len(image_ids)))

fig, axes = plt.subplots(NUM_ROWS, NUM_COLS, figsize=(2.5 * NUM_COLS, 2.5 * NUM_ROWS))

for ax, img_id in zip(axes.flatten(), sample_ids_for_grid):
    img_path = os.path.join(IMAGE_DIR, img_id + ".jpg")
    img = Image.open(img_path).convert("RGB")

    # Get label name
    row = df.loc[df["image_id"] == img_id, class_cols].iloc[0]
    class_index = int(np.argmax(row.values))
    class_name = class_cols[class_index]

    ax.imshow(img)
    ax.set_title(f"{img_id}\n{class_name}", fontsize=6)
    ax.axis("off")

for ax in axes.flatten()[len(sample_ids_for_grid):]:
    ax.axis("off")

plt.suptitle("Sample HAM10000 Images", fontsize=14)
plt.tight_layout(rect=[0, 0, 1, 0.97])
sample_grid_path = os.path.join(OUTPUT_DIR, "sample_images.png")
plt.savefig(sample_grid_path, dpi=200)
plt.close()


# ================== REPORT TEXT ==================

# Most common image sizes
size_strings = [
    f"{h}x{w}: {count} images"
    for (h, w), count in image_sizes.most_common()
]

report_text = f"""
HAM10000 Dataset Quality Report
===============================

1. Dataset Overview
   - Number of labeled entries in CSV: {num_labeled}
   - Number of .jpg image files in folder: {num_image_files}
   - Number of expected images missing from folder: {num_missing}
   - Number of extra images not listed in CSV: {num_extra}

2. Class Distribution
"""

for cls, cnt, pct in zip(class_cols, class_counts.values, class_percents.values):
    report_text += f"   - {cls}: {cnt} images ({pct:.2f}%)\n"

report_text += f"""
   - Class distribution plot saved to: {class_dist_path}

3. Image Properties
   - Number of distinct image sizes: {len(image_sizes)}
"""

for line in size_strings[:10]:  # show top 10 sizes
    report_text += f"   - {line}\n"
if len(size_strings) > 10:
    report_text += "   - ... (only top 10 sizes shown)\n"

report_text += f"""
4. Pixel Intensity Statistics (all images, RGB normalized to 0–1)
   - Mean intensity: {pixel_mean:.4f}
   - Standard deviation: {pixel_std:.4f}
   - Minimum intensity: {pixel_min:.4f}
   - Maximum intensity: {pixel_max:.4f}
   - Pixel intensity histogram saved to: {pixel_hist_path}

5. Sample Images
   - Random sample grid saved to: {sample_grid_path}

6. Potential Issues
   - Missing images referenced in CSV: {num_missing}
   - Extra images not referenced in CSV: {num_extra}
"""

# Save report
report_path = os.path.join(OUTPUT_DIR, "ham10000_report.txt")
with open(report_path, "w") as f:
    f.write(report_text)

print(f"Report generated: {report_path}")
print(f"Class distribution figure: {class_dist_path}")
print(f"Pixel histogram figure: {pixel_hist_path}")
print(f"Sample image grid: {sample_grid_path}")
