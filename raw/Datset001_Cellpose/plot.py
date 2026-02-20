#!/usr/bin/env python3
"""
Visualize dataset images and labels side by side.

Usage:
    python plot.py 000              # Case 000 from training set
    python plot.py 000 --test       # Case 000 from test set
    python plot.py 000 --channel 1  # Show specific channel
"""

import argparse
from pathlib import Path
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt


def load_image(case_id: str, split: str = "train", channel: int = 0) -> np.ndarray:
    """Load a specific channel from an image."""
    dataset_dir = Path(__file__).parent
    if split == "train":
        images_dir = dataset_dir / "imagesTr"
    else:
        images_dir = dataset_dir / "imagesTs"

    img_file = images_dir / f"{case_id}_{channel:04d}.png"
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


def load_rgb_image(case_id: str, split: str = "train") -> np.ndarray:
    """Load all 3 channels and stack as RGB."""
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
    """Plot image and label side by side."""
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
    axes[0].set_title(f"Image {case_id}{title_suffix}")
    axes[0].axis("off")

    # Plot label
    axes[1].imshow(label, cmap="gray")
    axes[1].set_title(f"Label {case_id}")
    axes[1].axis("off")

    # Overall title
    split_name = "Training" if split == "train" else "Test"
    fig.suptitle(f"{split_name} Case: {case_id}", fontsize=14, fontweight="bold")

    plt.tight_layout()
    plt.show()


def main():
    parser = argparse.ArgumentParser(
        description="Visualize dataset images and labels"
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
        "--channel",
        type=int,
        default=None,
        help="Specific channel to display (default: RGB from all 3 channels)",
    )

    args = parser.parse_args()

    split = "test" if args.test else "train"

    try:
        plot_case(args.case_id, split, args.channel)
    except FileNotFoundError as e:
        print(f"Error: {e}")
        exit(1)


if __name__ == "__main__":
    main()
