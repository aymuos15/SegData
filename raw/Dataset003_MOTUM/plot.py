#!/usr/bin/env python3
"""
Visualize 3D MOTUM dataset volumes with mid-slices in axial, coronal, sagittal planes.

Usage:
    python plot.py 000                           # Case 000 from training set, FLAIR channel
    python plot.py 000 --test                    # Case 000 from test set
    python plot.py 000 --plane axial             # Show only axial plane
    python plot.py 000 --channel 3               # Show T1ce channel instead of FLAIR
    python plot.py 000 --channel 0,1,2           # Show first 3 channels
"""

from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.patches as patches
import nibabel as nib
import numpy as np


CHANNEL_NAMES = {
    0: "FLAIR",
    1: "T1",
    2: "T2",
    3: "T1ce",
}

LABEL_NAMES = {
    0: "background",
    1: "tumor_flair",
    2: "tumor_enhancing",
}

LABEL_COLORS = {
    0: None,  # transparent
    1: "red",
    2: "blue",
}


def load_channel(case_id: str, channel: int, split: str = "train") -> np.ndarray:
    """Load a single channel (0=FLAIR, 1=T1, 2=T2, 3=T1ce) from a 3D volume."""
    dataset_dir = Path(__file__).parent
    images_dirs = {
        "train": dataset_dir / "imagesTr",
        "test": dataset_dir / "imagesTs",
    }
    images_dir = images_dirs[split]

    img_file = images_dir / f"{case_id}_{channel:04d}.nii.gz"
    nib_img = nib.load(img_file)
    return np.array(nib_img.dataobj)


def load_label(case_id: str, split: str = "train") -> np.ndarray:
    """Load 3-class label for a case."""
    dataset_dir = Path(__file__).parent
    labels_dirs = {
        "train": dataset_dir / "labelsTr",
        "test": dataset_dir / "labelsTs",
    }
    labels_dir = labels_dirs[split]

    label_file = labels_dir / f"{case_id}.nii.gz"
    nib_label = nib.load(label_file)
    return np.array(nib_label.dataobj).astype(np.uint8)


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
    """Normalize image to 0-1 range using percentiles."""
    img_min = np.percentile(img, 2)
    img_max = np.percentile(img, 98)
    img_normalized = np.clip((img - img_min) / (img_max - img_min), 0, 1)
    return img_normalized


def plot_case(
    case_id: str,
    split: str = "train",
    plane: str = "all",
    channels: list = None,
):
    """Plot 3D volume slices in axial, coronal, sagittal planes with label overlay."""

    if channels is None:
        channels = [0]  # Default to FLAIR

    split_name_map = {"train": "Training", "test": "Test"}
    split_name = split_name_map[split]
    channel_str = ", ".join(CHANNEL_NAMES.get(c, str(c)) for c in channels)

    # Load channels and label
    volume_data = []
    for ch in channels:
        vol = load_channel(case_id, ch, split)
        volume_data.append(vol)
    label = load_label(case_id, split)

    # For display, use first channel if multiple
    display_channel = volume_data[0]

    # Extract mid-slices
    display_slices = get_mid_slices(display_channel)
    label_slices = get_mid_slices(label)

    # Map plane names to indices
    plane_map = {"axial": 0, "coronal": 1, "sagittal": 2}

    if plane == "all":
        # Create 2×3 grid: image and image+label overlay in 3 planes
        fig, axes = plt.subplots(2, 3, figsize=(15, 10))
        fig.suptitle(f"{split_name} Case {case_id} - {channel_str}", fontsize=16, fontweight="bold")

        # Row 0: Image only
        for col, plane_name in enumerate(["axial", "coronal", "sagittal"]):
            plane_idx = plane_map[plane_name]
            slice_data = display_slices[plane_idx]
            img_normalized = normalize_image(slice_data)
            axes[0, col].imshow(img_normalized, cmap="gray")
            axes[0, col].set_title(f"{CHANNEL_NAMES.get(channels[0], 'Channel')} ({plane_name})")
            axes[0, col].axis("of")

        # Row 1: Image with label overlay
        for col, plane_name in enumerate(["axial", "coronal", "sagittal"]):
            plane_idx = plane_map[plane_name]
            img_slice = display_slices[plane_idx]
            label_slice = label_slices[plane_idx]

            img_normalized = normalize_image(img_slice)
            axes[1, col].imshow(img_normalized, cmap="gray")

            # Overlay labels with different colors
            for label_class in [2, 1]:  # Draw class 2 first, then class 1 on top
                color = LABEL_COLORS[label_class]
                contours = np.where(label_slice == label_class)
                axes[1, col].scatter(
                    contours[1],
                    contours[0],
                    c=color,
                    s=1,
                    alpha=0.6,
                    label=LABEL_NAMES[label_class] if col == 0 else None,
                )

            axes[1, col].set_title(f"{CHANNEL_NAMES.get(channels[0], 'Channel')} + Labels ({plane_name})")
            axes[1, col].axis("of")

        # Add legend to first subplot
        handles = [
            patches.Patch(facecolor="red", label="tumor_flair"),
            patches.Patch(facecolor="blue", label="tumor_enhancing"),
        ]
        axes[1, 0].legend(handles=handles, loc="upper left", fontsize=8)

    else:
        # Show only specified plane: Image and Image+Label side by side
        plane_idx = plane_map.get(plane, 0)
        fig, axes = plt.subplots(1, 2, figsize=(12, 5))
        fig.suptitle(
            f"{split_name} Case {case_id} - {channel_str} {plane.capitalize()} Plane",
            fontsize=14,
            fontweight="bold",
        )

        # Load and display
        slice_img = display_slices[plane_idx]
        slice_label = label_slices[plane_idx]

        axes[0].imshow(normalize_image(slice_img), cmap="gray")
        axes[0].set_title(CHANNEL_NAMES.get(channels[0], "Channel"))
        axes[0].axis("of")

        axes[1].imshow(normalize_image(slice_img), cmap="gray")
        for label_class in [2, 1]:  # Draw class 2 first, then class 1 on top
            color = LABEL_COLORS[label_class]
            contours = np.where(slice_label == label_class)
            axes[1].scatter(
                contours[1],
                contours[0],
                c=color,
                s=1,
                alpha=0.6,
                label=LABEL_NAMES[label_class],
            )
        axes[1].set_title(f"{CHANNEL_NAMES.get(channels[0], 'Channel')} + Labels")
        axes[1].legend(fontsize=8)
        axes[1].axis("of")

    plt.tight_layout()
    plt.show()


def main():
    case_id = input("Enter case_id: ").strip()
    test = input("Use test set? (y/n): ").lower() == 'y'
    plane = input("Plane (all/axial/coronal/sagittal) [all]: ").strip() or "all"
    channel_str = input("Channel(s) (comma-separated, e.g. 0 or 0,1,2) [0]: ").strip() or "0"
    
    split = {True: "test", False: "train"}[test]
    channels = [int(c.strip()) for c in channel_str.split(",")]
    
    plot_case(case_id, split, plane, channels)

if __name__ == "__main__":
    main()
