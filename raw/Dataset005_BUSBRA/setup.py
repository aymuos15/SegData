#!/usr/bin/env python3
"""
Setup script to convert BUS-BRA breast ultrasound dataset to nnUNet format.

Prerequisites:
- Place the downloaded BUSBRA.zip in this directory

Processing:
- Extracts zip to temporary directory
- Recursively discovers all image files (excluding files with 'mask' in name)
- For each image, locates corresponding mask via multiple strategies:
  * Same name with '_mask' suffix
  * Parallel 'masks/' or 'Mask/' directory with same filename
  * Sibling 'Mask' subdirectory
- Converts images to grayscale PNG if needed (handles BMP, color JPG input)
- Binarizes masks to 0/255
- Applies 80/20 train/test split
- Saves in nnUNet format

Usage:
    python setup.py

Output:
    imagesTr/      → Training ultrasound images (grayscale, single channel)
    labelsTr/      → Training segmentation masks (binary 0/255)
    imagesTs/      → Test ultrasound images (grayscale, single channel)
    labelsTs/      → Test segmentation masks (binary 0/255)
    dataset.json   → Updated metadata with actual case counts
"""

import json
import shutil
import zipfile
from pathlib import Path
from typing import List, Tuple, Optional

import numpy as np
from PIL import Image
from tqdm import tqdm


def find_matching_mask(image_path: Path, extracted_dir: Path) -> Optional[Path]:
    """
    Locate the mask file corresponding to an image file.
    Tries multiple strategies:
    1. Same directory with '_mask' suffix
    2. Parallel 'masks/', 'Masks/', or 'Mask/' directory with same filename
    3. Parallel 'masks/', 'Masks/', or 'Mask/' directory with different prefix (e.g., 'mask_' for 'bus_')
    4. Sibling 'Mask/' subdirectory
    """
    image_stem = image_path.stem
    image_parent = image_path.parent

    # Strategy 1: Same directory with _mask suffix
    for ext in ['.png', '.jpg', '.jpeg', '.bmp', '.tif', '.tiff']:
        mask_file = image_parent / f"{image_stem}_mask{ext}"
        if mask_file.exists():
            return mask_file

    # Strategy 2: Parallel masks/ or Mask/ directory with exact stem match
    for masks_dir_name in ['masks', 'Masks', 'Mask']:
        masks_dir = image_parent.parent / masks_dir_name
        if masks_dir.exists():
            for ext in ['.png', '.jpg', '.jpeg', '.bmp', '.tif', '.tiff']:
                mask_file = masks_dir / f"{image_stem}{ext}"
                if mask_file.exists():
                    return mask_file

    # Strategy 3: Parallel masks/ or Mask/ directory with different prefix
    # Extract the numeric/ID part from image filename (e.g., "0001-l" from "bus_0001-l")
    # and look for mask_XXXX pattern
    for masks_dir_name in ['masks', 'Masks', 'Mask']:
        masks_dir = image_parent.parent / masks_dir_name
        if masks_dir.exists():
            # Try to find mask by extracting the ID part from image name
            # For example: bus_0001-l.png -> look for mask_0001-l.png
            for mask_file in masks_dir.iterdir():
                if mask_file.is_file() and mask_file.suffix.lower() in ['.png', '.jpg', '.jpeg', '.bmp', '.tif', '.tiff']:
                    # Extract the ID part (e.g., "0001-l" from both "bus_0001-l" and "mask_0001-l")
                    image_id_part = image_stem.split('_')[-1] if '_' in image_stem else image_stem
                    mask_id_part = mask_file.stem.split('_')[-1] if '_' in mask_file.stem else mask_file.stem
                    if image_id_part == mask_id_part:
                        return mask_file

    # Strategy 4: Sibling Mask/ subdirectory
    mask_subdir = image_parent / 'Mask'
    if mask_subdir.exists():
        for ext in ['.png', '.jpg', '.jpeg', '.bmp', '.tif', '.tiff']:
            mask_file = mask_subdir / f"{image_stem}{ext}"
            if mask_file.exists():
                return mask_file

    return None


def discover_images_and_masks(extracted_dir: Path) -> List[Tuple[Path, Path]]:
    """
    Recursively discover all image-mask pairs in extracted directory.
    Excludes files with 'mask' in the name from image search.
    """
    image_extensions = {'.png', '.jpg', '.jpeg', '.bmp', '.tif', '.tiff'}
    image_files = []

    # Recursively find all image files (excluding those with 'mask' in name)
    for img_file in extracted_dir.rglob('*'):
        if img_file.is_file() and img_file.suffix.lower() in image_extensions:
            if 'mask' not in img_file.name.lower():
                image_files.append(img_file)

    # Sort for consistency
    image_files.sort()

    # Find matching masks
    image_mask_pairs = []
    for img_file in image_files:
        mask_file = find_matching_mask(img_file, extracted_dir)
        if mask_file:
            image_mask_pairs.append((img_file, mask_file))

    return image_mask_pairs


def convert_to_grayscale_png(image_path: Path) -> Image.Image:
    """Convert image to grayscale PNG format."""
    img = Image.open(image_path)
    # Convert to grayscale
    if img.mode != 'L':
        img = img.convert('L')
    return img


def binarize_mask(mask_path: Path, threshold: int = 127) -> Image.Image:
    """Binarize mask to 0/255."""
    mask = Image.open(mask_path).convert('L')
    mask_array = np.array(mask)
    mask_binary = (mask_array > threshold).astype(np.uint8) * 255
    return Image.fromarray(mask_binary)


def setup_dataset():
    """Convert BUS-BRA dataset to nnUNet format."""

    dataset_dir = Path(__file__).parent

    # nnUNet structure (in-place)
    images_tr = dataset_dir / "imagesTr"
    labels_tr = dataset_dir / "labelsTr"
    images_ts = dataset_dir / "imagesTs"
    labels_ts = dataset_dir / "labelsTs"

    # Create directories
    for d in [images_tr, labels_tr, images_ts, labels_ts]:
        d.mkdir(parents=True, exist_ok=True)

    # Find and extract zip
    zip_files = list(dataset_dir.glob("*.zip"))
    if not zip_files:
        print("Error: No .zip file found in dataset directory")
        return

    zip_path = zip_files[0]
    print(f"Extracting {zip_path.name}...")

    temp_dir = dataset_dir / "temp_extract"
    if temp_dir.exists():
        shutil.rmtree(temp_dir)

    with zipfile.ZipFile(zip_path, 'r') as z:
        z.extractall(temp_dir)

    # Discover images and masks
    print("Discovering images and masks...")
    image_mask_pairs = discover_images_and_masks(temp_dir)

    if not image_mask_pairs:
        print("Error: No image-mask pairs found")
        shutil.rmtree(temp_dir)
        return

    print(f"Found {len(image_mask_pairs)} image-mask pairs")

    # Sort pairs for consistency
    image_mask_pairs.sort(key=lambda x: x[0].name)

    # Apply 80/20 split
    split_idx = int(0.8 * len(image_mask_pairs))
    train_pairs = image_mask_pairs[:split_idx]
    test_pairs = image_mask_pairs[split_idx:]

    print(f"Split: {len(train_pairs)} train, {len(test_pairs)} test")

    # Process training data
    print("Processing training data...")
    for idx, (img_path, mask_path) in enumerate(tqdm(train_pairs)):
        case_id = f"{idx:03d}"

        try:
            # Convert image to grayscale PNG
            img = convert_to_grayscale_png(img_path)
            img.save(images_tr / f"{case_id}_0000.png")

            # Binarize mask
            mask = binarize_mask(mask_path)
            mask.save(labels_tr / f"{case_id}.png")
        except Exception as e:
            print(f"Warning: Error processing training case {case_id}: {e}")

    # Process test data
    print("Processing test data...")
    for idx, (img_path, mask_path) in enumerate(tqdm(test_pairs)):
        case_id = f"{(split_idx + idx):03d}"

        try:
            # Convert image to grayscale PNG
            img = convert_to_grayscale_png(img_path)
            img.save(images_ts / f"{case_id}_0000.png")

            # Binarize mask
            mask = binarize_mask(mask_path)
            mask.save(labels_ts / f"{case_id}.png")
        except Exception as e:
            print(f"Warning: Error processing test case {case_id}: {e}")

    # Cleanup temp directory
    print("Cleaning up...")
    shutil.rmtree(temp_dir)

    # Update dataset.json
    print("Updating dataset.json...")
    dataset_json_path = dataset_dir / "dataset.json"

    with open(dataset_json_path, "r") as f:
        dataset_json = json.load(f)

    num_train = len(list(images_tr.glob("*_0000.png")))
    num_test = len(list(images_ts.glob("*_0000.png")))

    dataset_json["numTraining"] = num_train
    dataset_json["numTest"] = num_test

    with open(dataset_json_path, "w") as f:
        json.dump(dataset_json, f, indent=2)

    print(f"\n✓ Dataset converted to nnUNet format at {dataset_dir}")
    print(f"  - Training images: {num_train}")
    print(f"  - Training labels: {len(list(labels_tr.glob('*.png')))}")
    print(f"  - Test images: {num_test}")
    print(f"  - Test labels: {len(list(labels_ts.glob('*.png')))}")
    print(f"  - dataset.json updated")


if __name__ == "__main__":
    setup_dataset()
