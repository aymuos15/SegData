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

from PIL import Image
from tqdm import tqdm


def discover_images_and_masks(root_dir: Path) -> list[tuple[Path, Path]]:
    """
    Discover all image-mask pairs recursively.

    Strategies:
    1. Same name with '_mask' suffix
    2. Parallel 'masks/' or 'Mask/' directory with same filename
    3. Sibling 'Mask' subdirectory
    """
    image_extensions = {".bmp", ".jpg", ".jpeg", ".png", ".tiff"}
    pairs = []

    # Find all image files (excluding files with 'mask' in name)
    for img_path in root_dir.rglob("*"):
        if img_path.suffix.lower() not in image_extensions:
            continue
        if "mask" in img_path.name.lower():
            continue

        # Try to find corresponding mask
        mask_path = _find_mask_for_image(img_path)
        if mask_path is not None:
            pairs.append((img_path, mask_path))

    return pairs


def _find_mask_for_image(img_path: Path) -> Path | None:
    """Find mask file for a given image."""
    base_name = img_path.stem

    # Strategy 1: Same directory, '_mask' suffix
    candidate = img_path.parent / f"{base_name}_mask{img_path.suffix}"
    if candidate.exists():
        return candidate

    # Strategy 2: Parallel 'masks/' or 'Mask/' directory
    for mask_dir_name in ["masks", "Masks", "mask", "Mask"]:
        mask_dir = img_path.parent.parent / mask_dir_name
        candidate = mask_dir / img_path.name
        if candidate.exists():
            return candidate

    # Strategy 3: Sibling 'Mask' subdirectory
    mask_dir = img_path.parent / "Mask"
    candidate = mask_dir / img_path.name
    if candidate.exists():
        return candidate

    return None


def convert_to_grayscale_png(img_path: Path) -> Image.Image:
    """Convert image to grayscale PNG."""
    img = Image.open(img_path)
    if img.mode != "L":
        img = img.convert("L")
    return img


def binarize_mask(mask_path: Path) -> Image.Image:
    """Binarize mask to 0/255."""
    mask = Image.open(mask_path)
    if mask.mode != "L":
        mask = mask.convert("L")
    binary_array = (Image.open(mask_path).convert("L")) > 127
    return Image.fromarray((binary_array.astype(int) * 255).astype("uint8"))


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
    zip_path = zip_files[0]
    print(f"Extracting {zip_path.name}...")

    temp_dir = dataset_dir / "temp_extract"
    shutil.rmtree(temp_dir, ignore_errors=True)

    with zipfile.ZipFile(zip_path, "r") as z:
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
    print("  - dataset.json updated")


if __name__ == "__main__":
    setup_dataset()
