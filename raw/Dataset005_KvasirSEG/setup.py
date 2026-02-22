#!/usr/bin/env python3
"""
Setup script to convert Kvasir-SEG dataset to nnUNet format.
- Extracts zip file
- Applies 80/20 train/test split
- Saves images as separate channel PNGs
- Binarizes masks to 0/255
- Saves in nnUNet format
"""

import zipfile
from pathlib import Path
import numpy as np
from PIL import Image
from tqdm import tqdm


def setup_dataset():
    """Convert Kvasir-SEG zip to nnUNet format."""

    dataset_dir = Path(__file__).parent

    # nnUNet structure (in-place)
    images_tr = dataset_dir / "imagesTr"
    labels_tr = dataset_dir / "labelsTr"
    images_ts = dataset_dir / "imagesTs"
    labels_ts = dataset_dir / "labelsTs"

    # Create directories
    for d in [images_tr, labels_tr, images_ts, labels_ts]:
        d.mkdir(parents=True, exist_ok=True)

    # Extract zip
    print("Extracting kvasir-seg.zip...")
    zip_path = dataset_dir / "kvasir-seg.zip"
    temp_dir = dataset_dir / "temp"
    with zipfile.ZipFile(zip_path, 'r') as z:
        z.extractall(temp_dir)

    # Get image and mask directories
    images_dir = temp_dir / "Kvasir-SEG" / "images"
    masks_dir = temp_dir / "Kvasir-SEG" / "masks"

    # Get sorted image files
    image_files = sorted([f for f in images_dir.glob("*.jpg")])
    print(f"Found {len(image_files)} images")

    # Apply 80/20 split (first 800 train, last 200 test)
    split_idx = 800
    train_images = image_files[:split_idx]
    test_images = image_files[split_idx:]

    print(f"Train: {len(train_images)}, Test: {len(test_images)}")

    # Process training data
    print("Processing training data...")
    for idx, img_file in enumerate(tqdm(train_images)):
        case_id = f"{idx:03d}"

        # Read image and save as separate channels
        img = Image.open(img_file).convert('RGB')
        img_array = np.array(img)

        # Save separate channel files (nnUNet format)
        for ch in range(3):
            channel_img = Image.fromarray(img_array[:, :, ch])
            channel_img.save(images_tr / f"{case_id}_{ch:04d}.png")

        # Read and binarize mask (masks have same filename as images in masks/ directory)
        mask_file = masks_dir / (img_file.stem + ".jpg")
        mask = np.array(Image.open(mask_file).convert('L'))
        mask_binary = (mask > 127).astype(np.uint8) * 255
        Image.fromarray(mask_binary).save(labels_tr / f"{case_id}.png")

    # Process test data
    print("Processing test data...")
    for idx, img_file in enumerate(tqdm(test_images)):
        case_id = f"{(split_idx + idx):03d}"

        # Read image and save as separate channels
        img = Image.open(img_file).convert('RGB')
        img_array = np.array(img)

        # Save separate channel files (nnUNet format)
        for ch in range(3):
            channel_img = Image.fromarray(img_array[:, :, ch])
            channel_img.save(images_ts / f"{case_id}_{ch:04d}.png")

        # Read and binarize mask (masks have same filename as images in masks/ directory)
        mask_file = masks_dir / (img_file.stem + ".jpg")
        mask = np.array(Image.open(mask_file).convert('L'))
        mask_binary = (mask > 127).astype(np.uint8) * 255
        Image.fromarray(mask_binary).save(labels_ts / f"{case_id}.png")

    # Cleanup temp directory
    print("Cleaning up...")
    import shutil
    shutil.rmtree(temp_dir, ignore_errors=True)

    # Update dataset.json
    import json
    dataset_json_path = dataset_dir / "dataset.json"
    # Count unique samples (images have _0000, _0001, _0002 for channels)
    train_files = sorted([f.stem.replace("_0000", "") for f in images_tr.glob("*_0000.png")])
    test_files = sorted([f.stem.replace("_0000", "") for f in images_ts.glob("*_0000.png")])

    dataset_json = {
        "name": "Kvasir-SEG Polyp Segmentation",
        "description": "2D gastrointestinal polyp segmentation from colonoscopy. 1000 images with binary masks. 80/20 train/test split applied.",
        "reference": "https://datasets.simula.no/kvasir-seg/",
        "citation": "Jha, D. et al. Kvasir-SEG: A Segmented Polyp Dataset. MMM 2020. https://doi.org/10.1007/978-3-030-37734-2_37",
        "license": "Research/Educational use only - https://datasets.simula.no/kvasir-seg/",
        "release": "1.0",
        "tensorImageSize": "2D",
        "file_ending": ".png",
        "channel_names": {
            "0": "red",
            "1": "green",
            "2": "blue"
        },
        "labels": {
            "0": "background",
            "1": "polyp"
        },
        "numTraining": len(train_files),
        "numTest": len(test_files),
        "numLabels": 2
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
