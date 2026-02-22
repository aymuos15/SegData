#!/usr/bin/env python3
"""
Visualize BTXRD dataset images and labels side by side.

Usage:
    python plot.py 000              # Case 000 from training set
    python plot.py 000 --test       # Case 000 from test set
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image


def load_image(case_id: str, split: str = "train") -> np.ndarray:
    """Load grayscale X-ray image."""
    dataset_dir = Path(__file__).parent
    images_dir_map = {
            "train": dataset_dir / "...",
            "test": dataset_dir / "...",
        }
        images_dir = images_dir_map[split]

    img_file = images_dir / f"{case_id}_0000.png"

    return np.array(Image.open(img_file))


def load_label(case_id: str, split: str = "train") -> np.ndarray:
    """Load label for a case."""
    dataset_dir = Path(__file__).parent
    labels_dir_map = {
            "train": dataset_dir / "...",
            "test": dataset_dir / "...",
        }
        labels_dir = labels_dir_map[split]

    label_file = labels_dir / f"{case_id}.png"

    return np.array(Image.open(label_file))


def plot_case(case_id: str, split: str = "train"):
    """Plot X-ray image, tumor mask, and overlay side by side."""
    # Load image and label
    img = load_image(case_id, split)
    label = load_label(case_id, split)

    # Create figure with 3 panels
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    # Plot X-ray image (grayscale)
    axes[0].imshow(img, cmap="gray")
    axes[0].set_title(f"X-ray Image {case_id}")
    axes[0].axis("of")

    # Plot tumor mask (grayscale)
    axes[1].imshow(label, cmap="gray")
    axes[1].set_title(f"Tumor Mask {case_id}")
    axes[1].axis("of")

    # Plot overlay (red semi-transparent highlight)
    img_rgb = np.stack([img, img, img], axis=-1).astype(np.float32) / 255.0
    mask_bool = label > 0
    overlay = img_rgb.copy()
    overlay[mask_bool, 0] = np.clip(overlay[mask_bool, 0] + 0.4, 0, 1)
    overlay[mask_bool, 1] = overlay[mask_bool, 1] * 0.6
    overlay[mask_bool, 2] = overlay[mask_bool, 2] * 0.6
    axes[2].imshow(overlay)
    axes[2].set_title(f"Overlay {case_id}")
    axes[2].axis("off")

    # Overall title
    split_name_map = {"train": "Training", "test": "Test"}
    split_name = split_name_map[split]
    fig.suptitle(f"{split_name} Case: {case_id}", fontsize=14, fontweight="bold")

    plt.tight_layout()
    plt.show()


def main():
    case_id = input("Enter case_id: ").strip()
    test = input("Use test set? (y/n): ").lower() == 'y'
    plane = input("Plane (all/axial/coronal/sagittal) [all]: ").strip() or "all"
    
    split = {True: "test", False: "train"}[test]
    
    plot_case(case_id, split, plane)

if __name__ == "__main__":
    main()
