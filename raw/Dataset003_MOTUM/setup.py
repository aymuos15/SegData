#!/usr/bin/env python3
"""
Setup script to convert MOTUM dataset to nnUNet format.

Prerequisites:
- Place the downloaded zip file in this directory
- Zip must contain BIDS-formatted structure:
  * sub-XXXX/anat/ - Contains 4 MRI sequences (FLAIR, T1, T2, T1ce)
  * derivatives/sub-XXXX/ - Contains segmentation labels

Processing:
- Reads BIDS-formatted dataset (sub-XXXX/anat/ and derivatives/sub-XXXX/)
- Extracts 4 MRI channels per case (FLAIR, T1, T2, T1ce)
- Merges two segmentation labels (FLAIR + T1ce) into 3-class semantic mask:
  * 0 = background
  * 1 = FLAIR lesion only
  * 2 = T1ce-enhancing (tumor/active)
- Applies 80/20 train/test split (sorted by patient ID)
- Saves organized files in nnUNet format

Usage:
    python setup.py

Output:
    imagesTr/      → Training CT images (4 channels each)
    labelsTr/      → Training segmentation masks (3-class)
    imagesTs/      → Test CT images (4 channels each)
    labelsTs/      → Test segmentation masks (3-class)
    dataset.json   → Updated metadata with actual case counts
"""

import json
import re
from pathlib import Path

import nibabel as nib
import numpy as np
from tqdm import tqdm


def extract_subject_id(subject_dir):
    """Extract numeric ID from sub-XXXX directory name."""
    return int(subject_dir.name.split('-')[1])


def setup_dataset():
    """Convert MOTUM BIDS dataset to nnUNet format."""

    dataset_dir = Path(__file__).parent

    # nnUNet structure (in-place)
    images_tr = dataset_dir / "imagesTr"
    labels_tr = dataset_dir / "labelsTr"
    images_ts = dataset_dir / "imagesTs"
    labels_ts = dataset_dir / "labelsTs"

    # Create directories
    for d in [images_tr, labels_tr, images_ts, labels_ts]:
        d.mkdir(parents=True, exist_ok=True)

    # Find all subject directories (BIDS format: sub-XXXX)
    print("Looking for BIDS subject directories...")
    subject_dirs = sorted(
        [d for d in dataset_dir.iterdir() if d.is_dir() and d.name.startswith("sub-")],
        key=extract_subject_id,
    )

    print(f"Found {len(subject_dirs)} subject directories")

    # Apply 80/20 split
    split_idx = int(0.8 * len(subject_dirs))
    train_subjects = subject_dirs[:split_idx]
    test_subjects = subject_dirs[split_idx:]

    print(f"Split: {len(train_subjects)} train, {len(test_subjects)} test")

    # Process training cases
    print("Processing training cases...")
    for idx, subject_dir in enumerate(tqdm(train_subjects)):
        case_id = f"{idx:03d}"
        process_subject(dataset_dir, subject_dir, case_id, images_tr, labels_tr)

    # Process test cases
    print("Processing test cases...")
    for idx, subject_dir in enumerate(tqdm(test_subjects)):
        case_id = f"{idx:03d}"
        process_subject(dataset_dir, subject_dir, case_id, images_ts, labels_ts)

    # Update dataset.json
    print("Updating dataset.json...")
    dataset_json_path = dataset_dir / "dataset.json"

    with open(dataset_json_path, "r") as f:
        dataset_json = json.load(f)

    dataset_json["numTraining"] = len(train_subjects)
    dataset_json["numTest"] = len(test_subjects)

    with open(dataset_json_path, "w") as f:
        json.dump(dataset_json, f, indent=2)

    print(f"✓ Dataset converted to nnUNet format at {dataset_dir}")
    print(f"  - Training cases: {len(train_subjects)}")
    print(f"  - Training images: {len(list(images_tr.glob('*_0000.nii.gz')))} cases × 4 channels")
    print(f"  - Training labels: {len(list(labels_tr.glob('*.nii.gz')))} merged masks")
    print(f"  - Test cases: {len(test_subjects)}")
    print(f"  - Test images: {len(list(images_ts.glob('*_0000.nii.gz')))} cases × 4 channels")
    print(f"  - Test labels: {len(list(labels_ts.glob('*.nii.gz')))} merged masks")
    print("  - dataset.json updated")


def process_subject(dataset_dir, subject_dir, case_id, images_dir, labels_dir):
    """
    Process a single BIDS subject: extract 4 MRI channels and merge two label files.

    Image locations: subject_dir/anat/{subject}_{modality}.nii.gz
    Label locations: dataset_dir/derivatives/{subject}/

    Channels: FLAIR, T1, T2, T1ce (saved as _0000, _0001, _0002, _0003)
    Labels: Merge flair_seg_label1 and t1ce_seg_label2 into 3-class mask
    """
    import shutil

    anat_dir = subject_dir / "anat"
    deriv_dir = dataset_dir / "derivatives" / subject_dir.name

    # Hardcode channel filenames
    subject_name = subject_dir.name
    channels = [
        anat_dir / f"{subject_name}_flair.nii.gz",
        anat_dir / f"{subject_name}_t1.nii.gz",
        anat_dir / f"{subject_name}_t2.nii.gz",
        anat_dir / f"{subject_name}_t1ce.nii.gz",
    ]

    # Copy 4 channels to nnUNet format
    for ch_idx, channel_file in enumerate(channels):
        shutil.copy(channel_file, images_dir / f"{case_id}_{ch_idx:04d}.nii.gz")

    # Hardcode label filenames
    label1_file = deriv_dir / "flair_seg_label1.nii.gz"
    label2_file = deriv_dir / "t1ce_seg_label2.nii.gz"

    # Load labels (both available case)
    label1_nib = nib.load(label1_file)
    label1 = np.array(label1_nib.get_fdata(), dtype=np.uint8)

    label2_nib = nib.load(label2_file)
    label2 = np.array(label2_nib.get_fdata(), dtype=np.uint8)

    # Merge labels: 0=bg, 1=FLAIR-only, 2=T1ce-enhancing
    label_merged = np.zeros_like(label1)
    label_merged[label1 == 1] = 1
    label_merged[label2 == 1] = 2  # Overwrites in overlap
    merged_nib = nib.Nifti1Image(label_merged, label1_nib.affine, label1_nib.header)

    # Save merged label
    nib.save(merged_nib, labels_dir / f"{case_id}.nii.gz")


if __name__ == "__main__":
    setup_dataset()
