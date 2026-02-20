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


def find_matching_file(directory, patterns):
    """Find first file matching any pattern (case-insensitive)."""
    for item in directory.iterdir():
        if item.is_file():
            lower_name = item.name.lower()
            for pattern in patterns:
                if pattern.lower() in lower_name:
                    return item
    return None


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

    # Find and extract zip
    print("Finding and extracting zip file...")
    zip_files = list(dataset_dir.glob("*.zip"))
    if not zip_files:
        print("Error: No .zip file found in dataset directory")
        return

    zip_path = zip_files[0]
    print(f"Extracting {zip_path.name}...")
    with zipfile.ZipFile(zip_path, "r") as z:
        z.extractall(dataset_dir / "temp_extract")

    # Find the extracted root directory (could be nested)
    temp_dir = dataset_dir / "temp_extract"

    # Check if there's a nested "MSLesSeg Dataset.zip" that needs extraction
    dataset_zip = temp_dir / "MSLesSeg Dataset.zip"
    if dataset_zip.exists():
        print(f"Extracting nested MSLesSeg Dataset.zip...")
        with zipfile.ZipFile(dataset_zip, "r") as z:
            z.extractall(temp_dir / "dataset_files")
        extracted_root = temp_dir / "dataset_files"
    else:
        extracted_roots = [d for d in temp_dir.iterdir() if d.is_dir()]
        if len(extracted_roots) == 1:
            extracted_root = extracted_roots[0]
        else:
            extracted_root = temp_dir

    print(f"Looking for cases in {extracted_root}")

    # Find all case directories
    case_dirs = []
    for root, dirs, files in os.walk(extracted_root):
        # Look for directories containing imaging files
        nifti_files = [f for f in files if f.lower().endswith(".nii.gz") or f.lower().endswith(".nii")]
        if nifti_files:
            case_dirs.append(Path(root))

    case_dirs = sorted(case_dirs)

    if not case_dirs:
        print("Error: No NIfTI files found in extracted directory")
        return

    print(f"Found {len(case_dirs)} case directories")

    # Determine train/test split
    train_cases = []
    test_cases = []

    # Check for train/test subdirectory structure
    for case_dir in case_dirs:
        case_path_str = str(case_dir).lower()
        if "train" in case_path_str or "training" in case_path_str:
            train_cases.append(case_dir)
        elif "test" in case_path_str or "testing" in case_path_str:
            test_cases.append(case_dir)
        else:
            # Default to train if no clear split marker
            train_cases.append(case_dir)

    # If no split detected, use 80/20 split
    if not test_cases and train_cases:
        split_idx = int(0.8 * len(train_cases))
        test_cases = train_cases[split_idx:]
        train_cases = train_cases[:split_idx]

    print(f"Split: {len(train_cases)} train, {len(test_cases)} test")

    # Process training cases
    print("Processing training cases...")
    for idx, case_dir in enumerate(tqdm(train_cases)):
        case_id = f"{idx:03d}"
        process_case(case_dir, case_id, images_tr, labels_tr)

    # Process test cases
    print("Processing test cases...")
    for idx, case_dir in enumerate(tqdm(test_cases)):
        case_id = f"{idx:03d}"
        process_case(case_dir, case_id, images_ts, labels_ts)

    # Cleanup temp directory
    print("Cleaning up...")
    shutil.rmtree(temp_dir)

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
    print(f"  - dataset.json updated")


def process_case(case_dir, case_id, images_dir, labels_dir):
    """Process a single case: find and copy FLAIR and mask files only."""
    # Find FLAIR file
    flair_file = find_matching_file(case_dir, ["flair"])
    if not flair_file:
        print(f"Warning: FLAIR not found in {case_dir.name}")
        return

    # Find mask/lesion file
    mask_file = find_matching_file(case_dir, ["mask", "lesion"])
    if not mask_file:
        print(f"Warning: Mask not found in {case_dir.name}")
        return

    # Copy FLAIR and mask to nnUNet format
    try:
        shutil.copy(flair_file, images_dir / f"{case_id}_0000.nii.gz")
        shutil.copy(mask_file, labels_dir / f"{case_id}.nii.gz")
    except Exception as e:
        print(f"Error copying files for case {case_id}: {e}")
        return


if __name__ == "__main__":
    setup_dataset()
