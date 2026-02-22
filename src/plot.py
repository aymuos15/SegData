#!/usr/bin/env python3
"""
Unified dataset visualization script.

Auto-detects format (PNG/NIfTI) and dimensionality (2D/3D) from dataset.json.
Displays images and labels side by side with interactive case selection.

Usage:
    python plot.py <dataset_path>
    python plot.py raw/Dataset003_MOTUM
    python plot.py Datset001_Cellpose
"""

import json
import sys
from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
import matplotlib.patches as patches
import nibabel as nib
import numpy as np
from PIL import Image


def load_dataset_meta(dataset_dir: Path) -> Dict:
    """Load dataset metadata from dataset.json."""
    dataset_json = dataset_dir / "dataset.json"
    with open(dataset_json) as f:
        return json.load(f)


def list_case_ids(dataset_dir: Path, file_ending: str, split: str) -> List[str]:
    """Get all case IDs for a split."""
    split_dirs = {
        "train": ("imagesTr", "labelsTr"),
        "test": ("imagesTs", "labelsTs"),
    }
    images_dir_name, _ = split_dirs[split]
    images_dir = dataset_dir / images_dir_name

    if file_ending in (".nii.gz", ".nii"):
        # NIfTI: find *_0000.nii.gz files
        pattern = f"*_0000{file_ending}"
        files = sorted(images_dir.glob(pattern))
        return [f.name.replace(f"_0000{file_ending}", "") for f in files]
    else:
        # PNG: find *_0000.png files
        pattern = f"*_0000{file_ending}"
        files = sorted(images_dir.glob(pattern))
        return [f.name.replace(f"_0000{file_ending}", "") for f in files]


def load_image_2d(dataset_dir: Path, case_id: str, split: str, channel: int, file_ending: str) -> np.ndarray:
    """Load a single 2D image channel."""
    split_dirs = {
        "train": ("imagesTr", "labelsTr"),
        "test": ("imagesTs", "labelsTs"),
    }
    images_dir_name, _ = split_dirs[split]
    images_dir = dataset_dir / images_dir_name

    img_file = images_dir / f"{case_id}_{channel:04d}{file_ending}"
    return np.array(Image.open(img_file))


def load_label_2d(dataset_dir: Path, case_id: str, split: str, file_ending: str) -> np.ndarray:
    """Load a 2D label."""
    split_dirs = {
        "train": ("imagesTr", "labelsTr"),
        "test": ("imagesTs", "labelsTs"),
    }
    _, labels_dir_name = split_dirs[split]
    labels_dir = dataset_dir / labels_dir_name

    label_file = labels_dir / f"{case_id}{file_ending}"
    return np.array(Image.open(label_file))


def load_volume(path: Path) -> np.ndarray:
    """Load a 3D NIfTI volume."""
    nib_img = nib.load(path)
    return np.array(nib_img.dataobj)


def get_mid_slices(volume: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Extract mid-slices in axial, coronal, sagittal planes."""
    d, h, w = volume.shape
    mid_d = d // 2
    mid_h = h // 2
    mid_w = w // 2

    axial = volume[mid_d, :, :]    # Z plane
    coronal = volume[:, mid_h, :]  # Y plane
    sagittal = volume[:, :, mid_w] # X plane

    return axial, coronal, sagittal


def normalize_image(img: np.ndarray) -> np.ndarray:
    """Normalize image to 0-1 range using percentile clipping."""
    img_min = np.percentile(img, 1)
    img_max = np.percentile(img, 99)
    img_normalized = np.clip((img - img_min) / (img_max - img_min + 1e-8), 0, 1)
    return img_normalized


def plot_case_2d(dataset_dir: Path, meta: Dict, case_id: str, split: str):
    """Plot 2D image(s) with label."""
    file_ending = meta["file_ending"]
    channel_names = meta.get("channel_names", {})
    labels = meta.get("labels", {})
    n_channels = len(channel_names)

    # Load label
    label = load_label_2d(dataset_dir, case_id, split, file_ending)

    split_name_map = {"train": "Training", "test": "Test"}
    split_name = split_name_map[split]

    if n_channels == 1:
        # Single channel: 1×2 layout
        img = load_image_2d(dataset_dir, case_id, split, 0, file_ending)
        fig, axes = plt.subplots(1, 2, figsize=(12, 5))

        axes[0].imshow(img, cmap="gray")
        axes[0].set_title(f"Image {case_id} ({channel_names.get('0', 'Channel 0')})")
        axes[0].axis("off")

        axes[1].imshow(label, cmap="gray")
        axes[1].set_title(f"Label {case_id}")
        axes[1].axis("off")

    elif n_channels == 3:
        # RGB composite: 1×2 layout
        channels = [load_image_2d(dataset_dir, case_id, split, ch, file_ending) for ch in range(3)]
        rgb = np.stack(channels, axis=-1)
        rgb_normalized = rgb.astype(float) / np.max(rgb) if np.max(rgb) > 0 else rgb.astype(float)

        fig, axes = plt.subplots(1, 2, figsize=(12, 5))

        axes[0].imshow(rgb_normalized)
        axes[0].set_title(f"Image {case_id} (RGB)")
        axes[0].axis("off")

        axes[1].imshow(label, cmap="gray")
        axes[1].set_title(f"Label {case_id}")
        axes[1].axis("off")

    else:
        # Multi-channel: 1×(n+1) layout
        channels = [load_image_2d(dataset_dir, case_id, split, ch, file_ending) for ch in range(n_channels)]
        fig, axes = plt.subplots(1, n_channels + 1, figsize=(4 * (n_channels + 1), 4))

        for ch in range(n_channels):
            axes[ch].imshow(channels[ch], cmap="gray")
            axes[ch].set_title(f"{channel_names.get(str(ch), f'Channel {ch}')}")
            axes[ch].axis("off")

        axes[-1].imshow(label, cmap="gray")
        axes[-1].set_title("Label")
        axes[-1].axis("off")

    fig.suptitle(f"{split_name} Case: {case_id}", fontsize=14, fontweight="bold")
    plt.tight_layout()
    plt.show()


def plot_case_3d(dataset_dir: Path, meta: Dict, case_id: str, split: str, plane: str = "all"):
    """Plot 3D volume mid-slices with label."""
    file_ending = meta["file_ending"]
    channel_names = meta.get("channel_names", {})
    labels = meta.get("labels", {})
    num_labels = meta.get("numLabels", 2)
    n_channels = len(channel_names)

    split_dirs = {
        "train": ("imagesTr", "labelsTr"),
        "test": ("imagesTs", "labelsTs"),
    }
    images_dir_name, labels_dir_name = split_dirs[split]
    images_dir = dataset_dir / images_dir_name
    labels_dir = dataset_dir / labels_dir_name

    # Load first channel and label
    img_file = images_dir / f"{case_id}_0000{file_ending}"
    label_file = labels_dir / f"{case_id}{file_ending}"

    display_volume = load_volume(img_file)
    label_volume = load_volume(label_file)

    # Extract mid-slices
    display_slices = get_mid_slices(display_volume)
    label_slices = get_mid_slices(label_volume)

    split_name_map = {"train": "Training", "test": "Test"}
    split_name = split_name_map[split]
    channel_str = channel_names.get("0", "Channel 0")

    plane_map = {"axial": 0, "coronal": 1, "sagittal": 2}
    plane_names = ["axial", "coronal", "sagittal"]

    if plane == "all":
        # 3×2 grid (3 planes × [image + image+label])
        fig, axes = plt.subplots(3, 2, figsize=(12, 12))
        fig.suptitle(f"{split_name} Case {case_id} - {channel_str}", fontsize=16, fontweight="bold")

        for row, plane_name in enumerate(plane_names):
            plane_idx = plane_map[plane_name]
            img_slice = display_slices[plane_idx]
            label_slice = label_slices[plane_idx]
            img_norm = normalize_image(img_slice)

            # Left: image only
            axes[row, 0].imshow(img_norm, cmap="gray")
            axes[row, 0].set_title(f"{channel_str} ({plane_name})")
            axes[row, 0].axis("off")

            # Right: image + label overlay
            axes[row, 1].imshow(img_norm, cmap="gray")

            if num_labels == 2:
                # Binary: simple overlay
                mask = label_slice > 0
                axes[row, 1].contourf(mask.astype(float), levels=[0.5, 1.5], colors=["red"], alpha=0.5)
            else:
                # Multi-class: use tab10 colormap
                from matplotlib.cm import get_cmap
                cmap = get_cmap("tab10")
                for label_class in range(1, num_labels):
                    mask = label_slice == label_class
                    if np.any(mask):
                        color = cmap(label_class % 10)
                        axes[row, 1].contourf(mask.astype(float), levels=[0.5, 1.5], colors=[color], alpha=0.6)

            axes[row, 1].set_title(f"{channel_str} + Labels ({plane_name})")
            axes[row, 1].axis("off")

        # Add legend
        if num_labels > 2:
            handles = [
                patches.Patch(facecolor=plt.cm.tab10(i % 10), label=labels.get(str(i), f"Class {i}"))
                for i in range(1, num_labels)
            ]
            axes[0, 1].legend(handles=handles, loc="upper right", fontsize=8)

    else:
        # Single plane: 1×2 side by side
        plane_idx = plane_map[plane]
        img_slice = display_slices[plane_idx]
        label_slice = label_slices[plane_idx]
        img_norm = normalize_image(img_slice)

        fig, axes = plt.subplots(1, 2, figsize=(12, 5))
        fig.suptitle(f"{split_name} Case {case_id} - {channel_str} {plane.capitalize()} Plane",
                     fontsize=14, fontweight="bold")

        axes[0].imshow(img_norm, cmap="gray")
        axes[0].set_title(channel_str)
        axes[0].axis("off")

        axes[1].imshow(img_norm, cmap="gray")

        if num_labels == 2:
            mask = label_slice > 0
            axes[1].contourf(mask.astype(float), levels=[0.5, 1.5], colors=["red"], alpha=0.5)
        else:
            from matplotlib.cm import get_cmap
            cmap = get_cmap("tab10")
            for label_class in range(1, num_labels):
                mask = label_slice == label_class
                if np.any(mask):
                    color = cmap(label_class % 10)
                    axes[1].contourf(mask.astype(float), levels=[0.5, 1.5], colors=[color], alpha=0.6)

        axes[1].set_title(f"{channel_str} + Labels")
        axes[1].axis("off")

        # Add legend
        if num_labels > 2:
            handles = [
                patches.Patch(facecolor=plt.cm.tab10(i % 10), label=labels.get(str(i), f"Class {i}"))
                for i in range(1, num_labels)
            ]
            axes[1].legend(handles=handles, loc="upper right", fontsize=8)

    plt.tight_layout()
    plt.show()


def main():
    """Main entry point."""
    if len(sys.argv) < 2:
        print("Usage: python plot.py <dataset_path>")
        print("Example: python plot.py raw/Dataset003_MOTUM")
        sys.exit(1)

    dataset_arg = sys.argv[1]

    # Resolve dataset path
    dataset_dir = Path(dataset_arg)
    if not dataset_dir.is_absolute():
        dataset_dir = Path.cwd() / dataset_dir

    if not dataset_dir.exists():
        print(f"Error: Dataset directory not found: {dataset_dir}")
        sys.exit(1)

    # Load metadata
    try:
        meta = load_dataset_meta(dataset_dir)
    except Exception as e:
        print(f"Error loading dataset.json: {e}")
        sys.exit(1)

    # Determine format
    is_3d = meta.get("tensorImageSize") == "3D"
    file_ending = meta.get("file_ending", ".png")

    # List available cases
    try:
        available_ids = list_case_ids(dataset_dir, file_ending, "train")
        print(f"Available case IDs (training): {', '.join(available_ids[:10])}")
        if len(available_ids) > 10:
            print(f"  ... and {len(available_ids) - 10} more")
    except Exception as e:
        print(f"Warning: Could not list case IDs: {e}")

    # Interactive prompts
    case_id = input("Enter case_id: ").strip()
    test_str = input("Use test set? (y/n): ").strip().lower()
    split = "test" if test_str == "y" else "train"

    # For 3D datasets, ask about plane
    if is_3d:
        plane = input("Plane (axial/coronal/sagittal/all) [all]: ").strip() or "all"
    else:
        plane = "all"

    # Plot
    try:
        if is_3d:
            plot_case_3d(dataset_dir, meta, case_id, split, plane)
        else:
            plot_case_2d(dataset_dir, meta, case_id, split)
    except Exception as e:
        print(f"Error plotting case: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
