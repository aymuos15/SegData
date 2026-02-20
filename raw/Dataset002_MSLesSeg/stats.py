#!/usr/bin/env python3
"""Dataset statistics and connected component analysis for MSLesSeg 3D segmentation.

GPU-accelerated using CuPy for fast connected component labeling with 26-connectivity (3D).

Analyzes each case in the dataset and provides:
- Number of training and test samples
- Average volume dimensions
- Connected component count per volume (lesion counts)
- Lesion size statistics (in voxels and mm³)
- Summary statistics across all cases
- One JSON file per case

Usage:
    python stats.py
    python stats.py --output-dir results
"""

import argparse
import json
from pathlib import Path
from typing import Dict, List

import nibabel as nib
import numpy as np
from tqdm import tqdm

import cupy as cp
from cupyx.scipy.ndimage import generate_binary_structure as cp_generate_binary_structure
from cupyx.scipy.ndimage import label as cp_label


class DatasetAnalyzer:
    """Analyze 3D dataset structure and connected components."""

    def __init__(self, dataset_dir: str):
        """Initialize analyzer with dataset directory."""
        self.dataset_dir = Path(dataset_dir)
        self.images_tr = self.dataset_dir / "imagesTr"
        self.labels_tr = self.dataset_dir / "labelsTr"
        self.images_ts = self.dataset_dir / "imagesTs"
        self.labels_ts = self.dataset_dir / "labelsTs"

    def count_samples(self) -> Dict[str, int]:
        """Count number of training and test samples."""
        train_count = len(list(self.images_tr.glob("*_0000.nii.gz")))
        test_count = len(list(self.images_ts.glob("*_0000.nii.gz")))
        return {
            "train": train_count,
            "test": test_count,
            "total": train_count + test_count,
        }

    def get_volume_dimensions(self, sample_ids: List[str], images_dir: Path) -> Dict:
        """Calculate average dimensions across 3D volumes."""
        shapes = []

        for sample_id in sample_ids:
            # Read first channel to get dimensions
            img_file = images_dir / f"{sample_id}_0000.nii.gz"
            if img_file.exists():
                img = nib.load(img_file)
                shape = img.shape
                shapes.append(shape)

        if not shapes:
            return {}

        shapes = np.array(shapes)
        return {
            "mean_shape": [float(np.mean(shapes[:, i])) for i in range(3)],
            "min_shape": [int(np.min(shapes[:, i])) for i in range(3)],
            "max_shape": [int(np.max(shapes[:, i])) for i in range(3)],
            "std_shape": [float(np.std(shapes[:, i])) for i in range(3)],
        }

    def count_lesions(self, label_array: np.ndarray, voxel_volume_mm3: float = 1.0) -> Dict:
        """Count connected components (lesions) in a binary mask using 26-connectivity (GPU accelerated)."""
        # For 3D, use 26-connectivity (neighboring voxels including all diagonals)
        label_gpu = cp.asarray(label_array.astype(cp.uint8))
        structure = cp_generate_binary_structure(3, 3)  # 3D, 3-connectivity = 26-connectivity

        labeled_gpu, num_components = cp_label(label_gpu, structure=structure)
        labeled_array = cp.asnumpy(labeled_gpu)

        # Free GPU memory
        del label_gpu, labeled_gpu
        cp.get_default_memory_pool().free_all_blocks()

        if num_components > 0:
            component_sizes = np.bincount(labeled_array.ravel())[1:]  # skip background
            component_sizes = component_sizes[component_sizes > 0]
        else:
            component_sizes = np.array([])

        # Convert voxel sizes to mm³
        component_sizes_mm3 = component_sizes * voxel_volume_mm3

        return {
            "num_lesions": int(num_components),
            "lesion_sizes_voxels": component_sizes.tolist(),
            "lesion_sizes_mm3": component_sizes_mm3.tolist(),
            "mean_lesion_size_voxels": float(np.mean(component_sizes)) if len(component_sizes) > 0 else 0.0,
            "median_lesion_size_voxels": float(np.median(component_sizes)) if len(component_sizes) > 0 else 0.0,
            "min_lesion_size_voxels": int(np.min(component_sizes)) if len(component_sizes) > 0 else 0,
            "max_lesion_size_voxels": int(np.max(component_sizes)) if len(component_sizes) > 0 else 0,
            "std_lesion_size_voxels": float(np.std(component_sizes)) if len(component_sizes) > 0 else 0.0,
            "mean_lesion_size_mm3": float(np.mean(component_sizes_mm3)) if len(component_sizes_mm3) > 0 else 0.0,
            "median_lesion_size_mm3": float(np.median(component_sizes_mm3)) if len(component_sizes_mm3) > 0 else 0.0,
            "min_lesion_size_mm3": float(np.min(component_sizes_mm3)) if len(component_sizes_mm3) > 0 else 0.0,
            "max_lesion_size_mm3": float(np.max(component_sizes_mm3)) if len(component_sizes_mm3) > 0 else 0.0,
            "std_lesion_size_mm3": float(np.std(component_sizes_mm3)) if len(component_sizes_mm3) > 0 else 0.0,
            "total_lesion_voxels": int(np.sum(component_sizes)),
            "total_lesion_volume_mm3": float(np.sum(component_sizes_mm3)),
        }

    def analyze_split(self, split: str, output_dir: Path | None = None) -> Dict:
        """Analyze training or test split."""
        if split == "train":
            images_dir = self.images_tr
            labels_dir = self.labels_tr
        else:
            images_dir = self.images_ts
            labels_dir = self.labels_ts

        # Get unique sample IDs (images have _0000 suffix for first channel)
        image_files = sorted(images_dir.glob("*_0000.nii.gz"))
        sample_ids = [f.name.replace("_0000.nii.gz", "") for f in image_files]

        if not sample_ids:
            return {
                "num_samples": 0,
                "volume_dims": {},
                "aggregate": {},
            }

        # Get volume dimensions
        volume_dims = self.get_volume_dimensions(sample_ids, images_dir)

        # Analyze each sample
        all_lesion_counts = []
        all_lesion_sizes = []
        all_lesion_volumes_mm3 = []

        for sample_id in tqdm(sample_ids, desc=f"Analyzing {split} split", unit="sample"):
            # Read label
            label_file = labels_dir / f"{sample_id}.nii.gz"
            if not label_file.exists():
                print(f"Warning: Missing label for {sample_id}")
                continue

            # Load NIfTI file
            label_nib = nib.load(label_file)
            label_array = label_nib.get_fdata() > 0  # Binarize

            # Get voxel dimensions (assume isotropic for simplicity)
            pixdim = label_nib.header.get("pixdim")[1]  # First spatial dimension
            voxel_volume_mm3 = float(pixdim) ** 3

            # Count lesions
            lesion_stats = self.count_lesions(label_array, voxel_volume_mm3)

            # Save individual case JSON
            if output_dir:
                case_output = {
                    "case_id": sample_id,
                    "split": split,
                    "voxel_size_mm": float(pixdim),
                    "stats": lesion_stats,
                }
                case_json = output_dir / f"{sample_id}.json"
                with open(case_json, "w") as f:
                    json.dump(case_output, f, indent=2)

            all_lesion_counts.append(lesion_stats["num_lesions"])
            all_lesion_sizes.extend(lesion_stats["lesion_sizes_voxels"])
            all_lesion_volumes_mm3.extend(lesion_stats["lesion_sizes_mm3"])

        # Aggregate statistics
        aggregate = {
            "mean_lesions_per_case": float(np.mean(all_lesion_counts)) if all_lesion_counts else 0.0,
            "median_lesions_per_case": float(np.median(all_lesion_counts)) if all_lesion_counts else 0.0,
            "min_lesions": int(np.min(all_lesion_counts)) if all_lesion_counts else 0,
            "max_lesions": int(np.max(all_lesion_counts)) if all_lesion_counts else 0,
            "std_lesions_per_case": float(np.std(all_lesion_counts)) if all_lesion_counts else 0.0,
            "total_lesions_all_cases": int(sum(all_lesion_counts)),
            "lesion_size_stats": {
                "total_lesions": int(len(all_lesion_sizes)),
                "mean_voxels": float(np.mean(all_lesion_sizes)) if all_lesion_sizes else 0.0,
                "median_voxels": float(np.median(all_lesion_sizes)) if all_lesion_sizes else 0.0,
                "min_voxels": int(np.min(all_lesion_sizes)) if all_lesion_sizes else 0,
                "max_voxels": int(np.max(all_lesion_sizes)) if all_lesion_sizes else 0,
                "std_voxels": float(np.std(all_lesion_sizes)) if all_lesion_sizes else 0.0,
                "mean_mm3": float(np.mean(all_lesion_volumes_mm3)) if all_lesion_volumes_mm3 else 0.0,
                "median_mm3": float(np.median(all_lesion_volumes_mm3)) if all_lesion_volumes_mm3 else 0.0,
                "min_mm3": float(np.min(all_lesion_volumes_mm3)) if all_lesion_volumes_mm3 else 0.0,
                "max_mm3": float(np.max(all_lesion_volumes_mm3)) if all_lesion_volumes_mm3 else 0.0,
                "std_mm3": float(np.std(all_lesion_volumes_mm3)) if all_lesion_volumes_mm3 else 0.0,
            },
        }

        return {
            "num_samples": len(sample_ids),
            "volume_dims": volume_dims,
            "aggregate": aggregate,
        }


def analyze_dataset(
    dataset_dir: str,
    output_dir: str | None = None,
) -> Dict:
    """Analyze complete dataset."""
    analyzer = DatasetAnalyzer(dataset_dir)

    # Count samples
    sample_counts = analyzer.count_samples()

    # Create output directory if specified
    output_path = None
    if output_dir:
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

    # Analyze each split
    train_analysis = analyzer.analyze_split("train", output_path)
    test_analysis = analyzer.analyze_split("test", output_path)

    # Compile output
    output = {
        "dataset_info": {
            "dataset_dir": dataset_dir,
            "sample_counts": sample_counts,
        },
        "train": train_analysis,
        "test": test_analysis,
    }

    # Save aggregate stats
    if output_path:
        aggregate_json = output_path / "aggregate_stats.json"
        with open(aggregate_json, "w") as f:
            json.dump(output, f, indent=2)
        print(f"\nAggregate stats saved to {aggregate_json}")
        print(f"Per-case stats saved to {output_path}/*.json")

    return output


def print_summary(output: Dict) -> None:
    """Print summary statistics to console."""
    counts = output["dataset_info"]["sample_counts"]

    print("\n" + "=" * 80)
    print("DATASET SUMMARY")
    print("=" * 80)
    print(f"Training samples: {counts['train']}")
    print(f"Test samples: {counts['test']}")
    print(f"Total samples: {counts['total']}")

    # Training split
    if output["train"]["num_samples"] > 0:
        print("\n" + "=" * 80)
        print("TRAINING SPLIT")
        print("=" * 80)
        train_data = output["train"]
        dims = train_data["volume_dims"]
        agg = train_data["aggregate"]

        if dims:
            print(f"\nVolume Dimensions (voxels):")
            print(f"  Mean: {dims['mean_shape']}")
            print(f"  Range: {dims['min_shape']} - {dims['max_shape']}")
            print(f"  Std Dev: {dims['std_shape']}")

        print(f"\nLesion Statistics:")
        print(f"  Avg lesions per case: {agg['mean_lesions_per_case']:.2f}")
        print(f"  Median lesions per case: {agg['median_lesions_per_case']:.0f}")
        print(f"  Lesion count range: {agg['min_lesions']}-{agg['max_lesions']}")
        print(f"  Total lesions: {agg['total_lesions_all_cases']}")

        comp_stats = agg["lesion_size_stats"]
        print(f"\nLesion Sizes:")
        print(f"  Total lesions: {comp_stats['total_lesions']}")
        print(f"  Mean: {comp_stats['mean_voxels']:.2f} voxels ({comp_stats['mean_mm3']:.2f} mm³)")
        print(f"  Median: {comp_stats['median_voxels']:.2f} voxels ({comp_stats['median_mm3']:.2f} mm³)")
        print(f"  Range: {comp_stats['min_voxels']}-{comp_stats['max_voxels']} voxels ({comp_stats['min_mm3']:.2f}-{comp_stats['max_mm3']:.2f} mm³)")
        print(f"  Std Dev: {comp_stats['std_voxels']:.2f} voxels ({comp_stats['std_mm3']:.2f} mm³)")

    # Test split
    if output["test"]["num_samples"] > 0:
        print("\n" + "=" * 80)
        print("TEST SPLIT")
        print("=" * 80)
        test_data = output["test"]
        dims = test_data["volume_dims"]
        agg = test_data["aggregate"]

        if dims:
            print(f"\nVolume Dimensions (voxels):")
            print(f"  Mean: {dims['mean_shape']}")
            print(f"  Range: {dims['min_shape']} - {dims['max_shape']}")
            print(f"  Std Dev: {dims['std_shape']}")

        print(f"\nLesion Statistics:")
        print(f"  Avg lesions per case: {agg['mean_lesions_per_case']:.2f}")
        print(f"  Median lesions per case: {agg['median_lesions_per_case']:.0f}")
        print(f"  Lesion count range: {agg['min_lesions']}-{agg['max_lesions']}")
        print(f"  Total lesions: {agg['total_lesions_all_cases']}")

        comp_stats = agg["lesion_size_stats"]
        print(f"\nLesion Sizes:")
        print(f"  Total lesions: {comp_stats['total_lesions']}")
        print(f"  Mean: {comp_stats['mean_voxels']:.2f} voxels ({comp_stats['mean_mm3']:.2f} mm³)")
        print(f"  Median: {comp_stats['median_voxels']:.2f} voxels ({comp_stats['median_mm3']:.2f} mm³)")
        print(f"  Range: {comp_stats['min_voxels']}-{comp_stats['max_voxels']} voxels ({comp_stats['min_mm3']:.2f}-{comp_stats['max_mm3']:.2f} mm³)")
        print(f"  Std Dev: {comp_stats['std_voxels']:.2f} voxels ({comp_stats['std_mm3']:.2f} mm³)")


def main():
    parser = argparse.ArgumentParser(
        description="Analyze MSLesSeg dataset statistics and lesion counts"
    )
    parser.add_argument(
        "--dataset-dir",
        default=str(Path(__file__).parent),
        help="Path to dataset directory",
    )
    parser.add_argument(
        "--output-dir",
        default="results",
        help="Directory to save JSON results (one per case)",
    )
    parser.add_argument(
        "--no-save",
        action="store_true",
        help="Don't save JSON output",
    )

    args = parser.parse_args()

    output_dir = None if args.no_save else args.output_dir

    output = analyze_dataset(
        dataset_dir=args.dataset_dir,
        output_dir=output_dir,
    )

    print_summary(output)


if __name__ == "__main__":
    main()
