#!/usr/bin/env python3
"""
Visualize 3D FLAIR dataset volumes with mid-slices in axial, coronal, sagittal planes.

Usage:
    python plot.py 000              # Case 000 from training set
    python plot.py 000 --test       # Case 000 from test set
    python plot.py 000 --plane axial  # Show only axial plane
"""

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import nibabel as nib
import numpy as np


def load_flair(case_id: str, split: str = "train") -> np.ndarray:
    """Load FLAIR from a 3D volume."""
    dataset_dir = Path(__file__).parent
    if split == "train":
        images_dir = dataset_dir / "imagesTr"
    else:
        images_dir = dataset_dir / "imagesTs"

    img_file = images_dir / f"{case_id}_0000.nii.gz"
    if not img_file.exists():
        raise FileNotFoundError(f"Image not found: {img_file}")

    nib_img = nib.load(img_file)
    return np.array(nib_img.dataobj)


def load_label(case_id: str, split: str = "train") -> np.ndarray:
    """Load label for a case."""
    dataset_dir = Path(__file__).parent
    if split == "train":
        labels_dir = dataset_dir / "labelsTr"
    else:
        labels_dir = dataset_dir / "labelsTs"

    label_file = labels_dir / f"{case_id}.nii.gz"
    if not label_file.exists():
        raise FileNotFoundError(f"Label not found: {label_file}")

    nib_label = nib.load(label_file)
    return np.array(nib_label.dataobj)


def get_mid_slices(volume: np.ndarray):
    """Extract mid-slices in three planes."""
    d, h, w = volume.shape
    mid_d = d // 2
    mid_h = h // 2
    mid_w = w // 2

    axial = volume[mid_d, :, :]  # Z plane
    coronal = volume[:, mid_h, :]  # Y plane
    sagittal = volume[:, :, mid_w]  # X plane

    return axial, coronal, sagittal


def normalize_image(img: np.ndarray) -> np.ndarray:
    """Normalize image to 0-1 range."""
    img_min = np.percentile(img, 2)
    img_max = np.percentile(img, 98)
    img_normalized = np.clip((img - img_min) / (img_max - img_min), 0, 1)
    return img_normalized


def plot_case(case_id: str, split: str = "train", plane: str = "all"):
    """Plot 3D FLAIR slices in axial, coronal, sagittal planes with lesion overlay."""

    split_name = "Training" if split == "train" else "Test"

    # Load FLAIR and label
    try:
        flair = load_flair(case_id, split)
        label = load_label(case_id, split)
    except FileNotFoundError as e:
        print(f"Error: {e}")
        raise

    # Extract mid-slices
    flair_slices = get_mid_slices(flair)
    label_slices = get_mid_slices(label)

    # Map plane names to indices
    plane_map = {"axial": 0, "coronal": 1, "sagittal": 2}

    if plane == "all":
        # Create 2×3 grid: FLAIR and FLAIR+Lesion in 3 planes
        fig, axes = plt.subplots(2, 3, figsize=(15, 10))
        fig.suptitle(f"{split_name} Case {case_id} - FLAIR", fontsize=16, fontweight="bold")

        # Row 0: FLAIR only
        for col, plane_name in enumerate(["axial", "coronal", "sagittal"]):
            plane_idx = plane_map[plane_name]
            slice_data = flair_slices[plane_idx]
            img_normalized = normalize_image(slice_data)
            axes[0, col].imshow(img_normalized, cmap="gray")
            axes[0, col].set_title(f"FLAIR ({plane_name})")
            axes[0, col].axis("off")

        # Row 1: FLAIR with lesion overlay
        for col, plane_name in enumerate(["axial", "coronal", "sagittal"]):
            plane_idx = plane_map[plane_name]
            flair_slice = flair_slices[plane_idx]
            label_slice = label_slices[plane_idx]

            img_normalized = normalize_image(flair_slice)
            axes[1, col].imshow(img_normalized, cmap="gray")

            # Overlay lesion mask contour
            if label_slice.sum() > 0:
                contours = np.where(label_slice > 0)
                axes[1, col].scatter(contours[1], contours[0], c="red", s=1, alpha=0.5)

            axes[1, col].set_title(f"FLAIR + Lesion ({plane_name})")
            axes[1, col].axis("off")

    else:
        # Show only specified plane: FLAIR and FLAIR+Lesion side by side
        plane_idx = plane_map.get(plane, 0)
        fig, axes = plt.subplots(1, 2, figsize=(12, 5))
        fig.suptitle(f"{split_name} Case {case_id} - FLAIR {plane.capitalize()} Plane", fontsize=14, fontweight="bold")

        # Load and display
        slice_flair = flair_slices[plane_idx]
        slice_label = label_slices[plane_idx]

        axes[0].imshow(normalize_image(slice_flair), cmap="gray")
        axes[0].set_title("FLAIR")
        axes[0].axis("off")

        axes[1].imshow(normalize_image(slice_flair), cmap="gray")
        if slice_label.sum() > 0:
            contours = np.where(slice_label > 0)
            axes[1].scatter(contours[1], contours[0], c="red", s=1, alpha=0.5)
        axes[1].set_title("FLAIR + Lesion")
        axes[1].axis("off")

    plt.tight_layout()
    plt.show()


def main():
    parser = argparse.ArgumentParser(
        description="Visualize 3D dataset volumes"
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
    parser.add_argument(
        "--plane",
        choices=["all", "axial", "coronal", "sagittal"],
        default="all",
        help="Plane(s) to display (default: all)",
    )

    args = parser.parse_args()

    split = "test" if args.test else "train"

    try:
        plot_case(args.case_id, split, args.plane)
    except FileNotFoundError as e:
        print(f"Error: {e}")
        exit(1)


if __name__ == "__main__":
    main()
