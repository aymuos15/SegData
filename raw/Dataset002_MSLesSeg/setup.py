#!/usr/bin/env python3
"""
Setup script to convert MSLesSeg dataset to nnUNet format.
- Extracts zip
- Organizes 3D NIfTI FLAIR files and labels
- Separates train/test split
- Saves in nnUNet format (FLAIR only, no T1/T2)
"""

import json
import os
import shutil
import zipfile
from pathlib import Path

import nibabel as nib
from tqdm import tqdm


def setup_dataset():
    """Convert MSLesSeg zip to nnUNet format (FLAIR only)."""

    dataset_dir = Path(__file__).parent

    # nnUNet structure (in-place)
    images_tr = dataset_dir / "imagesTr"
    labels_tr = dataset_dir / "labelsTr"
    images_ts = dataset_dir / "imagesTs"
    labels_ts = dataset_dir / "labelsTs"

    # Create directories
    for d in [images_tr, labels_tr, images_ts, labels_ts]:
        d.mkdir(parents=True, exist_ok=True)

    # Extract outer zip
    print("Extracting zip file...")
    temp_dir = dataset_dir / "temp_extract"
    outer_zip = dataset_dir / "27919209.zip"
    with zipfile.ZipFile(outer_zip, "r") as z:
        z.extractall(temp_dir)

    # Extract nested MSLesSeg Dataset.zip
    print("Extracting nested MSLesSeg Dataset.zip...")
    dataset_zip = temp_dir / "MSLesSeg Dataset.zip"
    with zipfile.ZipFile(dataset_zip, "r") as z:
        z.extractall(temp_dir / "dataset_files")

    # Hardcode data root
    data_root = temp_dir / "dataset_files" / "MSLesSeg Dataset"

    # Get train and test case directories
    train_dir = data_root / "train"
    test_dir = data_root / "test"

    train_cases = sorted([d for d in train_dir.iterdir() if d.is_dir()])
    test_cases = sorted([d for d in test_dir.iterdir() if d.is_dir()])

    print(f"Found {len(train_cases)} train cases and {len(test_cases)} test cases")

    # Process training cases
    print("Processing training cases...")
    for idx, case_dir in enumerate(tqdm(train_cases)):
        case_id = f"{idx:03d}"
        flair_file = case_dir / f"{case_dir.name}_FLAIR.nii.gz"
        mask_file = case_dir / f"{case_dir.name}_MASK.nii.gz"
        shutil.copy(flair_file, images_tr / f"{case_id}_0000.nii.gz")
        shutil.copy(mask_file, labels_tr / f"{case_id}.nii.gz")

    # Process test cases
    print("Processing test cases...")
    for idx, case_dir in enumerate(tqdm(test_cases)):
        case_id = f"{idx:03d}"
        flair_file = case_dir / f"{case_dir.name}_FLAIR.nii.gz"
        mask_file = case_dir / f"{case_dir.name}_MASK.nii.gz"
        shutil.copy(flair_file, images_ts / f"{case_id}_0000.nii.gz")
        shutil.copy(mask_file, labels_ts / f"{case_id}.nii.gz")

    # Cleanup temp directory
    print("Cleaning up...")
    shutil.rmtree(temp_dir, ignore_errors=True)

    # Update dataset.json
    print("Updating dataset.json...")
    dataset_json_path = dataset_dir / "dataset.json"

    dataset_json = {
        "name": "MSLesSeg Multiple Sclerosis Lesion Segmentation",
        "description": "3D MRI FLAIR dataset for MS white matter lesion segmentation. 115 scan series from 75 patients (50 RRMS, 5 PPMS), preprocessed to 182×218×182 voxels at 1mm isotropic MNI152 space.",
        "reference": "https://springernature.figshare.com/articles/dataset/MSLesSeg_baseline_and_benchmarking_of_a_new_Multiple_Sclerosis_Lesion_Segmentation_dataset/27919209",
        "citation": "Guarnera, F., Rondinella, A., Crispino, E. et al. MSLesSeg: baseline and benchmarking of a new Multiple Sclerosis Lesion Segmentation dataset. Sci Data 12, 920 (2025). https://doi.org/10.1038/s41597-025-05250-y",
        "license": "CC BY 4.0",
        "release": "1.0",
        "tensorImageSize": "3D",
        "file_ending": ".nii.gz",
        "channel_names": {"0": "FLAIR"},
        "labels": {"0": "background", "1": "lesion"},
        "numTraining": len(train_cases),
        "numTest": len(test_cases),
        "numLabels": 2,
    }

    with open(dataset_json_path, "w") as f:
        json.dump(dataset_json, f, indent=2)

    print(f"✓ Dataset converted to nnUNet format at {dataset_dir}")
    print(f"  - Training images: {len(train_cases)} FLAIR files")
    print(f"  - Training labels: {len(list(labels_tr.glob('*.nii.gz')))} files")
    print(f"  - Test images: {len(test_cases)} FLAIR files")
    print(f"  - Test labels: {len(list(labels_ts.glob('*.nii.gz')))} files")
    print("  - dataset.json updated")


if __name__ == "__main__":
    setup_dataset()
