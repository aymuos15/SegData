#!/usr/bin/env python3
"""
Visualize dataset images and labels side by side.

Usage:
    python plot.py 000              # Case 000 from training set
    python plot.py 000 --test       # Case 000 from test set
    python plot.py 000 --channel 1  # Show specific channel
"""

from pathlib import Path
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt


def load_image(case_id: str, split: str = "train", channel: int = 0) -> np.ndarray:
    """Load a specific channel from an image."""
    dataset_dir = Path(__file__).parent
    images_dirs = {
        "train": dataset_dir / "imagesTr",
        "test": dataset_dir / "imagesTs",
    }
    images_dir = images_dirs[split]

    img_file = images_dir / f"{case_id}_{channel:04d}.png"
    return np.array(Image.open(img_file))


def load_label(case_id: str, split: str = "train") -> np.ndarray:
    """Load label for a case."""
    dataset_dir = Path(__file__).parent
    labels_dirs = {
        "train": dataset_dir / "labelsTr",
        "test": dataset_dir / "labelsTs",
    }
    labels_dir = labels_dirs[split]

    label_file = labels_dir / f"{case_id}.png"
    return np.array(Image.open(label_file))


def load_rgb_image(case_id: str, split: str = "train") -> np.ndarray:
    """Load all 3 channels and stack as RGB."""
    channels = [load_image(case_id, split, ch) for ch in range(3)]
    return np.stack(channels, axis=-1)


def _plot_single_channel(case_id: str, split: str, channel: int, label: np.ndarray):
    """Plot single channel with label."""
    img = load_image(case_id, split, channel)
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    axes[0].imshow(img, cmap="gray")
    axes[0].set_title(f"Image {case_id} (Channel {channel})")
    axes[0].axis("of")
    axes[1].imshow(label, cmap="gray")
    axes[1].set_title(f"Label {case_id}")
    axes[1].axis("of")
    return fig


def _plot_rgb(case_id: str, split: str, label: np.ndarray):
    """Plot RGB with label."""
    img = load_rgb_image(case_id, split)
    img_normalized = img.astype(float) / 255.0
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    axes[0].imshow(img_normalized)
    axes[0].set_title(f"Image {case_id} (RGB)")
    axes[0].axis("of")
    axes[1].imshow(label, cmap="gray")
    axes[1].set_title(f"Label {case_id}")
    axes[1].axis("of")
    return fig


def plot_case(case_id: str, split: str = "train", channel: int = None):
    """Plot image and label side by side."""
    label = load_label(case_id, split)

    plot_funcs = {
        True: lambda: _plot_single_channel(case_id, split, channel, label),
        False: lambda: _plot_rgb(case_id, split, label),
    }
    fig = plot_funcs[channel is not None]()

    split_name_map = {"train": "Training", "test": "Test"}
    fig.suptitle(f"{split_name_map[split]} Case: {case_id}", fontsize=14, fontweight="bold")
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
