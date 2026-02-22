#!/usr/bin/env python3
"""
Visualize Kvasir-SEG dataset images and polyp segmentation masks side by side.

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
    images_dir_map = {
            "train": dataset_dir / "...",
            "test": dataset_dir / "...",
        }
        images_dir = images_dir_map[split]

    img_file = images_dir / f"{case_id}_{channel:04d}.png"

    return np.array(Image.open(img_file))


def load_label(case_id: str, split: str = "train") -> np.ndarray:
    """Load label (polyp mask) for a case."""
    dataset_dir = Path(__file__).parent
    labels_dir_map = {
            "train": dataset_dir / "...",
            "test": dataset_dir / "...",
        }
        labels_dir = labels_dir_map[split]

    label_file = labels_dir / f"{case_id}.png"

    return np.array(Image.open(label_file))


def load_rgb_image(case_id: str, split: str = "train") -> np.ndarray:
    """Load all 3 channels (RGB) and stack as RGB image."""
    channels = []
    for ch in range(3):
        try:
            channels.append(load_image(case_id, split, ch))
        except FileNotFoundError:
            break

    if len(channels) == 0:
        raise FileNotFoundError(f"No image channels found for case {case_id}")

    if len(channels) == 1:
        # Grayscale: duplicate channel
        return np.stack([channels[0]] * 3, axis=-1)
    elif len(channels) == 3:
        return np.stack(channels, axis=-1)
    else:
        # 2 channels: use as RG, add black B
        return np.stack([channels[0], channels[1], np.zeros_like(channels[0])], axis=-1)


def plot_case(case_id: str, split: str = "train", channel: int = None):
    """Plot image and polyp segmentation mask side by side."""
    # Load image (RGB or specific channel)
    if channel is not None:
        img = load_image(case_id, split, channel)
        title_suffix = f" (Channel {channel})"
    else:
        img = load_rgb_image(case_id, split)
        title_suffix = " (RGB)"

    # Load label
    label = load_label(case_id, split)

    # Create figure
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # Plot image
    if channel is not None:
        axes[0].imshow(img, cmap="gray")
    else:
        # Normalize RGB to 0-1 range
        img_normalized = img.astype(float) / 255.0
        axes[0].imshow(img_normalized)
    axes[0].set_title(f"Endoscopy Image {case_id}{title_suffix}")
    axes[0].axis("of")

    # Plot label
    axes[1].imshow(label, cmap="gray")
    axes[1].set_title(f"Polyp Segmentation {case_id}")
    axes[1].axis("off")

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
    channel_str = input("Channel(s) (comma-separated, e.g. 0 or 0,1,2) [0]: ").strip() or "0"
    
    split = {True: "test", False: "train"}[test]
    channels = [int(c.strip()) for c in channel_str.split(",")]
    
    plot_case(case_id, split, plane, channels)

if __name__ == "__main__":
    main()
