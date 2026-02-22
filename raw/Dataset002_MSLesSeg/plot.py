#!/usr/bin/env python3
"""
Visualize 3D FLAIR dataset volumes with mid-slices in axial, coronal, sagittal planes.

Usage:
    python plot.py 000              # Case 000 from training set
    python plot.py 000 --test       # Case 000 from test set
    python plot.py 000 --plane axial  # Show only axial plane
"""

from pathlib import Path

import matplotlib.pyplot as plt
import nibabel as nib
import numpy as np


def load_flair(case_id: str, split: str = "train") -> np.ndarray:
    """Load FLAIR from a 3D volume."""
    dataset_dir = Path(__file__).parent
    images_dirs = {
        "train": dataset_dir / "imagesTr",
        "test": dataset_dir / "imagesTs",
    }
    images_dir = images_dirs[split]

    img_file = images_dir / f"{case_id}_0000.nii.gz"
    nib_img = nib.load(img_file)
    return np.array(nib_img.dataobj)


def load_label(case_id: str, split: str = "train") -> np.ndarray:
    """Load label for a case."""
    dataset_dir = Path(__file__).parent
    labels_dirs = {
        "train": dataset_dir / "labelsTr",
        "test": dataset_dir / "labelsTs",
    }
    labels_dir = labels_dirs[split]

    label_file = labels_dir / f"{case_id}.nii.gz"
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


def _plot_all_planes(case_id: str, split: str, split_name: str, flair_slices, label_slices):
    """Plot all planes (2×3 grid)."""
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    fig.suptitle(f"{split_name} Case {case_id} - FLAIR", fontsize=16, fontweight="bold")

    plane_map = {"axial": 0, "coronal": 1, "sagittal": 2}

    # Row 0: FLAIR only
    for col, plane_name in enumerate(["axial", "coronal", "sagittal"]):
        plane_idx = plane_map[plane_name]
        slice_data = flair_slices[plane_idx]
        img_normalized = normalize_image(slice_data)
        axes[0, col].imshow(img_normalized, cmap="gray")
        axes[0, col].set_title(f"FLAIR ({plane_name})")
        axes[0, col].axis("of")

    # Row 1: FLAIR with lesion overlay
    for col, plane_name in enumerate(["axial", "coronal", "sagittal"]):
        plane_idx = plane_map[plane_name]
        flair_slice = flair_slices[plane_idx]
        label_slice = label_slices[plane_idx]

        img_normalized = normalize_image(flair_slice)
        axes[1, col].imshow(img_normalized, cmap="gray")

        # Overlay lesion mask contour
        contours = np.where(label_slice > 0)
        axes[1, col].scatter(contours[1], contours[0], c="red", s=1, alpha=0.5)

        axes[1, col].set_title(f"FLAIR + Lesion ({plane_name})")
        axes[1, col].axis("of")


def _plot_single_plane(case_id: str, split: str, split_name: str, plane: str, flair_slices, label_slices):
    """Plot single plane (1×2 side by side)."""
    plane_map = {"axial": 0, "coronal": 1, "sagittal": 2}
    plane_idx = plane_map[plane]

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    fig.suptitle(f"{split_name} Case {case_id} - FLAIR {plane.capitalize()} Plane", fontsize=14, fontweight="bold")

    slice_flair = flair_slices[plane_idx]
    slice_label = label_slices[plane_idx]

    axes[0].imshow(normalize_image(slice_flair), cmap="gray")
    axes[0].set_title("FLAIR")
    axes[0].axis("of")

    axes[1].imshow(normalize_image(slice_flair), cmap="gray")
    contours = np.where(slice_label > 0)
    axes[1].scatter(contours[1], contours[0], c="red", s=1, alpha=0.5)
    axes[1].set_title("FLAIR + Lesion")
    axes[1].axis("of")


def plot_case(case_id: str, split: str = "train", plane: str = "all"):
    """Plot 3D FLAIR slices in axial, coronal, sagittal planes with lesion overlay."""

    split_name_map = {"train": "Training", "test": "Test"}
    split_name = split_name_map[split]

    # Load FLAIR and label
    flair = load_flair(case_id, split)
    label = load_label(case_id, split)

    # Extract mid-slices
    flair_slices = get_mid_slices(flair)
    label_slices = get_mid_slices(label)

    # Dispatch based on plane
    plot_funcs = {
        "all": lambda: _plot_all_planes(case_id, split, split_name, flair_slices, label_slices),
        "axial": lambda: _plot_single_plane(case_id, split, split_name, "axial", flair_slices, label_slices),
        "coronal": lambda: _plot_single_plane(case_id, split, split_name, "coronal", flair_slices, label_slices),
        "sagittal": lambda: _plot_single_plane(case_id, split, split_name, "sagittal", flair_slices, label_slices),
    }
    plot_funcs[plane]()

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
