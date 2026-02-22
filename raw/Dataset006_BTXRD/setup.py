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

import json
import shutil
import zipfile
from pathlib import Path
from typing import List, Optional, Tuple

from PIL import Image, ImageDraw
from tqdm import tqdm


def find_data_root(root_dir: Path) -> Optional[Path]:
    """Find the root directory containing 'images' and 'Annotations' subdirectories."""
    # Check if root_dir itself has both
    if (root_dir / "images").exists() and (root_dir / "Annotations").exists():
        return root_dir

    # Search recursively for nested structure
    for item in root_dir.rglob("*"):
        if item.is_dir() and item.name in ("images", "Images"):
            parent = item.parent
            if (parent / "Annotations").exists() or (parent / "annotations").exists():
                return parent

    return None


def find_csv_file(root_dir: Path) -> Optional[Path]:
    """Find CSV file in the directory."""
    for csv_file in root_dir.rglob("*.csv"):
        return csv_file
    return None


def get_tumor_filenames(csv_path: Path) -> List[str]:
    """Read CSV and extract tumor filenames (benign + malignant, exclude normal)."""
    filenames = []
    try:
        with open(csv_path) as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith("#"):
                    continue
                # Expected format: filename,class (e.g., "image.jpg,benign")
                parts = line.split(",")
                if len(parts) >= 1:
                    filename = parts[0].strip()
                    # Filter out normal cases if class is specified
                    if len(parts) > 1:
                        class_label = parts[1].strip().lower()
                        if class_label != "normal":
                            filenames.append(filename)
                    else:
                        filenames.append(filename)
    except Exception as e:
        print(f"Warning: Error reading CSV: {e}")
    return filenames


def find_image_file(images_dir: Path, base_name: str) -> Optional[Path]:
    """Find image file matching base name (handles different extensions)."""
    for ext in [".jpg", ".jpeg", ".png", ".bmp", ".tiff"]:
        candidate = images_dir / f"{base_name}{ext}"
        if candidate.exists():
            return candidate
        # Try uppercase extensions
        candidate = images_dir / f"{base_name}{ext.upper()}"
        if candidate.exists():
            return candidate
    return None


def rasterize_coco_annotations(
    json_path: Path, fallback_image_path: Optional[Path] = None
) -> Optional[Image.Image]:
    """Rasterize COCO-style polygon annotations to binary mask."""
    import json as json_module

    try:
        with open(json_path) as f:
            coco_data = json_module.load(f)
    except Exception:
        return None

    # Determine image size
    if "images" in coco_data and len(coco_data["images"]) > 0:
        img_info = coco_data["images"][0]
        width = img_info.get("width", 512)
        height = img_info.get("height", 512)
    elif fallback_image_path is not None:
        # Use fallback image to determine size
        img = Image.open(fallback_image_path)
        width, height = img.size
    else:
        width, height = 512, 512

    # Create blank mask
    mask = Image.new("L", (width, height), 0)
    draw = ImageDraw.Draw(mask)

    # Draw polygons from annotations
    if "annotations" in coco_data:
        for ann in coco_data["annotations"]:
            if "segmentation" in ann:
                for seg in ann["segmentation"]:
                    if isinstance(seg, list) and len(seg) >= 6:
                        # Convert flat list to coordinate tuples
                        coords = [(seg[i], seg[i + 1]) for i in range(0, len(seg), 2)]
                        draw.polygon(coords, fill=255)

    return mask


def _build_valid_pairs(
    tumor_filenames: List[str], images_dir: Path, annotations_dir: Path
) -> Tuple[List[Tuple[Path, Path]], int]:
    """Build list of valid image-annotation pairs."""
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

    return valid_pairs, skipped


def _process_pairs(
    pairs: List[Tuple[Path, Path]],
    images_out: Path,
    labels_out: Path,
    offset: int = 0,
) -> None:
    """Process image-annotation pairs and save in nnUNet format."""
    for idx, (img_path, json_path) in enumerate(tqdm(pairs)):
        case_id = f"{(offset + idx):03d}"

        try:
            # Convert image to grayscale PNG
            img = Image.open(img_path)
            if img.mode != "L":
                img = img.convert("L")
            img.save(images_out / f"{case_id}_0000.png")

            # Rasterize COCO annotations to binary mask
            mask = rasterize_coco_annotations(json_path, fallback_image_path=img_path)
            if mask is not None:
                mask.save(labels_out / f"{case_id}.png")
            else:
                print(f"Warning: Could not create mask for case {case_id}")
        except Exception as e:
            print(f"Warning: Error processing case {case_id}: {e}")


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
    zip_path = zip_files[0]
    print(f"Extracting {zip_path.name}...")

    temp_dir = dataset_dir / "temp_extract"
    shutil.rmtree(temp_dir, ignore_errors=True)

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
    valid_pairs, skipped = _build_valid_pairs(tumor_filenames, images_dir, annotations_dir)

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
    _process_pairs(train_pairs, images_tr, labels_tr, offset=0)

    # Process test data
    print("Processing test data...")
    _process_pairs(test_pairs, images_ts, labels_ts, offset=split_idx)

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
    print("  - dataset.json updated")


if __name__ == "__main__":
    setup_dataset()
