#!/usr/bin/env python3
"""
Setup script to convert Cellpose dataset to nnUNet format.
- Extracts zips
- Converts instance labels to semantic (binary) labels
- Saves in nnUNet format (PNG)
"""

import zipfile
from pathlib import Path

import numpy as np
from PIL import Image
from tqdm import tqdm


def setup_dataset():
    """Convert Cellpose zips to nnUNet format."""

    dataset_dir = Path(__file__).parent

    # nnUNet structure (in-place)
    images_tr = dataset_dir / "imagesTr"
    labels_tr = dataset_dir / "labelsTr"
    images_ts = dataset_dir / "imagesTs"
    labels_ts = dataset_dir / "labelsTs"

    # Create directories
    for d in [images_tr, labels_tr, images_ts, labels_ts]:
        d.mkdir(parents=True, exist_ok=True)

    # Extract zips
    print("Extracting zip files...")
    with zipfile.ZipFile(dataset_dir / "train.zip", "r") as z:
        z.extractall(dataset_dir / "temp_train")

    with zipfile.ZipFile(dataset_dir / "test.zip", "r") as z:
        z.extractall(dataset_dir / "temp_test")

    # Process training data
    print("Processing training data...")
    train_dir = dataset_dir / "temp_train" / "train"
    img_files = sorted(train_dir.glob("*_img.png"))

    for img_file in tqdm(img_files):
        sample_id = img_file.stem.replace("_img", "")
        mask_file = img_file.parent / f"{sample_id}_masks.png"

        # Read image and split into separate channel files
        img = Image.open(img_file).convert("RGB")
        img_array = np.array(img)

        # Save separate channel files (nnUNet format)
        for ch in range(img_array.shape[2]):
            channel_img = Image.fromarray(img_array[:, :, ch])
            channel_img.save(images_tr / f"{sample_id}_{ch:04d}.png")

        # Read mask and binarize (instance -> semantic)
        mask = np.array(Image.open(mask_file))
        mask_binary = (mask > 0).astype(np.uint8) * 255  # Binarize to 0/255
        Image.fromarray(mask_binary).save(labels_tr / f"{sample_id}.png")

    # Process test data
    print("Processing test data...")
    test_dir = dataset_dir / "temp_test" / "test"
    img_files = sorted(test_dir.glob("*_img.png"))

    for img_file in tqdm(img_files):
        sample_id = img_file.stem.replace("_img", "")
        mask_file = img_file.parent / f"{sample_id}_masks.png"

        # Read image and split into separate channel files
        img = Image.open(img_file).convert("RGB")
        img_array = np.array(img)

        # Save separate channel files (nnUNet format)
        for ch in range(img_array.shape[2]):
            channel_img = Image.fromarray(img_array[:, :, ch])
            channel_img.save(images_ts / f"{sample_id}_{ch:04d}.png")

        # Save label
        mask = np.array(Image.open(mask_file))
        mask_binary = (mask > 0).astype(np.uint8) * 255
        Image.fromarray(mask_binary).save(labels_ts / f"{sample_id}.png")

    # Cleanup temp directories
    print("Cleaning up...")
    import shutil

    for temp_dir in [dataset_dir / "temp_train", dataset_dir / "temp_test"]:
        shutil.rmtree(temp_dir, ignore_errors=True)

    # Update dataset.json
    import json

    dataset_json_path = dataset_dir / "dataset.json"
    # Count unique samples (images have _0000, _0001, _0002 for channels)
    train_files = sorted([f.stem.replace("_0000", "") for f in images_tr.glob("*_0000.png")])
    test_files = sorted([f.stem.replace("_0000", "") for f in images_ts.glob("*_0000.png")])

    dataset_json = {
        "name": "Cellpose 2D Cell Segmentation",
        "description": "2D cell segmentation dataset from Cellpose. Instance labels converted to binary semantic segmentation.",
        "reference": "http://cellpose.org/",
        "citation": "Stringer, C., Wang, T., Michaelos, M. et al. Cellpose: a generalist algorithm for cellular segmentation. Nat Methods 18, 100–106 (2021). https://doi.org/10.1038/s41592-020-01018-x",
        "license": "MIT",
        "release": "1.0",
        "tensorImageSize": "2D",
        "file_ending": ".png",
        "channel_names": {"0": "red", "1": "green", "2": "blue"},
        "labels": {"background": 0, "cell": 1},
        "numTraining": len(train_files),
        "numTest": len(test_files),
        "numLabels": 2,
    }

    with open(dataset_json_path, "w") as f:
        json.dump(dataset_json, f, indent=2)

    print(f"✓ Dataset converted to nnUNet format at {dataset_dir}")
    print(f"  - Training images: {len(train_files)} files")
    print(f"  - Training labels: {len(list(labels_tr.glob('*.png')))} files")
    print(f"  - Test images: {len(test_files)} files")
    print(f"  - Test labels: {len(list(labels_ts.glob('*.png')))} files")
    print("  - dataset.json updated")


if __name__ == "__main__":
    setup_dataset()
