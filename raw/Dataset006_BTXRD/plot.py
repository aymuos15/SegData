#!/usr/bin/env python3
"""
Visualize BTXRD dataset images and labels side by side.

Usage:
    python plot.py 000              # Case 000 from training set
    python plot.py 000 --test       # Case 000 from test set
"""

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image


def load_image(case_id: str, split: str = "train") -> np.ndarray:
    """Load grayscale X-ray image."""
    dataset_dir = Path(__file__).parent
    if split == "train":
        images_dir = dataset_dir / "imagesTr"
    else:
        images_dir = dataset_dir / "imagesTs"

    img_file = images_dir / f"{case_id}_0000.png"
    if not img_file.exists():
        raise FileNotFoundError(f"Image not found: {img_file}")

    return np.array(Image.open(img_file))


def load_label(case_id: str, split: str = "train") -> np.ndarray:
    """Load label for a case."""
    dataset_dir = Path(__file__).parent
    if split == "train":
        labels_dir = dataset_dir / "labelsTr"
    else:
        labels_dir = dataset_dir / "labelsTs"

    label_file = labels_dir / f"{case_id}.png"
    if not label_file.exists():
        raise FileNotFoundError(f"Label not found: {label_file}")

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
    axes[0].axis("off")

    # Plot tumor mask (grayscale)
    axes[1].imshow(label, cmap="gray")
    axes[1].set_title(f"Tumor Mask {case_id}")
    axes[1].axis("off")

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
    split_name = "Training" if split == "train" else "Test"
    fig.suptitle(f"{split_name} Case: {case_id}", fontsize=14, fontweight="bold")

    plt.tight_layout()
    plt.show()


def main():
    parser = argparse.ArgumentParser(
        description="Visualize BTXRD dataset images and labels"
    )
    parser.add_argument(
        "case_id",
        help="Case ID to visualize (e.g., 000, 001, 042)",
    )
    parser.add_argument(
        "--test",
        action="store_true",
        help="Use test set (default: training set)",
    )

    args = parser.parse_args()

    split = "test" if args.test else "train"

    try:
        plot_case(args.case_id, split)
    except FileNotFoundError as e:
        print(f"Error: {e}")
        exit(1)


if __name__ == "__main__":
    main()
