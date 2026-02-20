#!/usr/bin/env python3
"""
Setup script to convert BTXRD bone tumor X-ray dataset to nnUNet format.

Prerequisites:
- Place the downloaded BTXRD zip in this directory

Processing:
- Extracts zip to temporary directory
- Reads dataset.csv to identify tumor images (benign + malignant, excludes normal)
- For each tumor image, rasterizes COCO-style polygon annotations to binary masks
- Converts JPEG images to grayscale PNG
- Applies 80/20 train/test split by sorted filename
- Saves in nnUNet format

Usage:
    python setup.py

Output:
    imagesTr/      -> Training X-ray images (grayscale, single channel)
    labelsTr/      -> Training segmentation masks (binary 0/255)
    imagesTs/      -> Test X-ray images (grayscale, single channel)
    labelsTs/      -> Test segmentation masks (binary 0/255)
    dataset.json   -> Updated metadata with actual case counts
"""

import csv
import json
import shutil
import zipfile
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
from PIL import Image, ImageDraw
from tqdm import tqdm


def find_data_root(temp_dir: Path) -> Optional[Path]:
    """Find the root directory containing 'images/' and 'Annotations/' folders.

    Handles both flat extraction and nested directory structures.
    """
    # Check temp_dir itself
    if (temp_dir / "images").exists() and (temp_dir / "Annotations").exists():
        return temp_dir

    # Check one level of nesting
    for child in temp_dir.iterdir():
        if child.is_dir():
            if (child / "images").exists() and (child / "Annotations").exists():
                return child

    return None


def find_csv_file(data_root: Path) -> Optional[Path]:
    """Find the dataset CSV file."""
    for name in ["dataset.csv", "Dataset.csv"]:
        csv_path = data_root / name
        if csv_path.exists():
            return csv_path
    # Search one level up in case CSV is at zip root
    for name in ["dataset.csv", "Dataset.csv"]:
        csv_path = data_root.parent / name
        if csv_path.exists():
            return csv_path
    return None


def get_tumor_filenames(csv_path: Path) -> List[str]:
    """Read CSV and return filenames of non-normal images.

    Dynamically discovers the filename and label columns.
    """
    with open(csv_path, "r", newline="") as f:
        reader = csv.DictReader(f)
        fieldnames = reader.fieldnames
        if not fieldnames:
            raise ValueError(f"CSV file {csv_path} has no header row")

        # Find filename column
        filename_col = None
        for candidate in ["file_name", "filename", "image_name", "ID", "File_name", "FileName"]:
            if candidate in fieldnames:
                filename_col = candidate
                break
        if filename_col is None:
            raise ValueError(
                f"Could not find filename column in CSV. Columns: {fieldnames}"
            )

        # Find label column
        label_col = None
        for candidate in ["Label", "label", "class", "Class", "category", "Category"]:
            if candidate in fieldnames:
                label_col = candidate
                break
        if label_col is None:
            raise ValueError(
                f"Could not find label column in CSV. Columns: {fieldnames}"
            )

        tumor_files = []
        for row in reader:
            label = row[label_col].strip().lower()
            if label != "normal":
                tumor_files.append(row[filename_col].strip())

    return tumor_files


def find_image_file(images_dir: Path, base_name: str) -> Optional[Path]:
    """Find an image file trying multiple extensions."""
    for ext in [".jpeg", ".jpg", ".png", ".JPEG", ".JPG", ".PNG"]:
        img_path = images_dir / f"{base_name}{ext}"
        if img_path.exists():
            return img_path
    # Try the base_name as-is (it may already include extension)
    img_path = images_dir / base_name
    if img_path.exists():
        return img_path
    return None


def rasterize_coco_annotations(json_path: Path, fallback_image_path: Optional[Path] = None) -> Optional[Image.Image]:
    """Rasterize COCO-style polygon annotations to a binary mask.

    Returns a PIL Image with 0 (background) and 255 (tumor), or None on failure.
    """
    with open(json_path, "r") as f:
        coco_data = json.load(f)

    # Get image dimensions
    width, height = None, None
    if "images" in coco_data and len(coco_data["images"]) > 0:
        img_info = coco_data["images"][0]
        width = img_info.get("width")
        height = img_info.get("height")

    # Fallback to reading actual image dimensions
    if width is None or height is None:
        if fallback_image_path and fallback_image_path.exists():
            with Image.open(fallback_image_path) as img:
                width, height = img.size
        else:
            return None

    # Create blank mask
    mask = Image.new("L", (width, height), 0)
    draw = ImageDraw.Draw(mask)

    # Draw all annotation polygons
    annotations = coco_data.get("annotations", [])
    for ann in annotations:
        segmentation = ann.get("segmentation", [])
        for polygon_flat in segmentation:
            if len(polygon_flat) < 6:
                # Skip degenerate polygons (fewer than 3 points)
                continue
            # Convert flat list [x1,y1,x2,y2,...] to list of (x,y) tuples
            points = list(zip(polygon_flat[0::2], polygon_flat[1::2]))
            draw.polygon(points, fill=255)

    return mask


def setup_dataset():
    """Convert BTXRD dataset to nnUNet format."""

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

    with zipfile.ZipFile(zip_path, "r") as z:
        z.extractall(temp_dir)

    # Find data root (handles nested directories)
    data_root = find_data_root(temp_dir)
    if data_root is None:
        print("Error: Could not find 'images/' and 'Annotations/' directories in zip")
        shutil.rmtree(temp_dir)
        return

    images_dir = data_root / "images"
    annotations_dir = data_root / "Annotations"

    # Read CSV to get tumor filenames, or fallback to using all annotation files
    csv_path = find_csv_file(data_root)
    if csv_path is not None:
        print(f"Reading {csv_path.name}...")
        tumor_filenames = get_tumor_filenames(csv_path)
        print(f"Found {len(tumor_filenames)} tumor entries in CSV")
    else:
        print("Warning: No dataset.csv found. Using all annotated images as tumors.")
        # Get all annotation files (these correspond to tumor images)
        tumor_filenames = [ann.stem for ann in sorted(annotations_dir.glob("*.json"))]
        print(f"Found {len(tumor_filenames)} tumor entries from annotation files")

    # Build valid image-mask pairs
    valid_pairs: List[Tuple[Path, Path]] = []
    skipped = 0

    for fname in tumor_filenames:
        # Extract base name (strip extension if present)
        base = Path(fname).stem

        # Find the image file
        img_path = find_image_file(images_dir, base)
        if img_path is None:
            skipped += 1
            continue

        # Find the annotation JSON
        json_path = annotations_dir / f"{base}.json"
        if not json_path.exists():
            skipped += 1
            continue

        valid_pairs.append((img_path, json_path))

    if not valid_pairs:
        print("Error: No valid image-annotation pairs found")
        shutil.rmtree(temp_dir)
        return

    print(f"Found {len(valid_pairs)} valid image-annotation pairs (skipped {skipped})")

    # Sort by filename for reproducible split
    valid_pairs.sort(key=lambda x: x[0].name)

    # Apply 80/20 split
    split_idx = int(0.8 * len(valid_pairs))
    train_pairs = valid_pairs[:split_idx]
    test_pairs = valid_pairs[split_idx:]

    print(f"Split: {len(train_pairs)} train, {len(test_pairs)} test")

    # Process training data
    print("Processing training data...")
    for idx, (img_path, json_path) in enumerate(tqdm(train_pairs)):
        case_id = f"{idx:03d}"

        try:
            # Convert image to grayscale PNG
            img = Image.open(img_path)
            if img.mode != "L":
                img = img.convert("L")
            img.save(images_tr / f"{case_id}_0000.png")

            # Rasterize COCO annotations to binary mask
            mask = rasterize_coco_annotations(json_path, fallback_image_path=img_path)
            if mask is not None:
                mask.save(labels_tr / f"{case_id}.png")
            else:
                print(f"Warning: Could not create mask for training case {case_id}")
        except Exception as e:
            print(f"Warning: Error processing training case {case_id}: {e}")

    # Process test data
    print("Processing test data...")
    for idx, (img_path, json_path) in enumerate(tqdm(test_pairs)):
        case_id = f"{(split_idx + idx):03d}"

        try:
            # Convert image to grayscale PNG
            img = Image.open(img_path)
            if img.mode != "L":
                img = img.convert("L")
            img.save(images_ts / f"{case_id}_0000.png")

            # Rasterize COCO annotations to binary mask
            mask = rasterize_coco_annotations(json_path, fallback_image_path=img_path)
            if mask is not None:
                mask.save(labels_ts / f"{case_id}.png")
            else:
                print(f"Warning: Could not create mask for test case {case_id}")
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

    print(f"\nDataset converted to nnUNet format at {dataset_dir}")
    print(f"  - Training images: {num_train}")
    print(f"  - Training labels: {len(list(labels_tr.glob('*.png')))}")
    print(f"  - Test images: {num_test}")
    print(f"  - Test labels: {len(list(labels_ts.glob('*.png')))}")
    print(f"  - dataset.json updated")


if __name__ == "__main__":
    setup_dataset()
