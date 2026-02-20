#!/usr/bin/env python3
"""Dataset statistics and connected component analysis for MOTUM 3D multi-class segmentation.

GPU-accelerated using CuPy for fast connected component labeling with 26-connectivity (3D).

Analyzes each case in the dataset and provides:
- Number of training and test samples
- Average volume dimensions
- Connected component count per volume per label class
- Tumor region size statistics (in voxels and mm³)
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
    """Analyze 3D multi-class dataset structure and connected components."""

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

    def count_components(self, label_array: np.ndarray, class_label: int, voxel_volume_mm3: float = 1.0) -> Dict:
        """Count connected components for a specific class using 26-connectivity (GPU accelerated)."""
        # Extract binary mask for this class
        binary_mask = (label_array == class_label).astype(np.uint8)

        # For 3D, use 26-connectivity
        label_gpu = cp.asarray(binary_mask)
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
            "num_regions": int(num_components),
            "region_sizes_voxels": component_sizes.tolist(),
            "region_sizes_mm3": component_sizes_mm3.tolist(),
            "mean_region_size_voxels": float(np.mean(component_sizes)) if len(component_sizes) > 0 else 0.0,
            "median_region_size_voxels": float(np.median(component_sizes)) if len(component_sizes) > 0 else 0.0,
            "min_region_size_voxels": int(np.min(component_sizes)) if len(component_sizes) > 0 else 0,
            "max_region_size_voxels": int(np.max(component_sizes)) if len(component_sizes) > 0 else 0,
            "std_region_size_voxels": float(np.std(component_sizes)) if len(component_sizes) > 0 else 0.0,
            "mean_region_size_mm3": float(np.mean(component_sizes_mm3)) if len(component_sizes_mm3) > 0 else 0.0,
            "median_region_size_mm3": float(np.median(component_sizes_mm3)) if len(component_sizes_mm3) > 0 else 0.0,
            "min_region_size_mm3": float(np.min(component_sizes_mm3)) if len(component_sizes_mm3) > 0 else 0.0,
            "max_region_size_mm3": float(np.max(component_sizes_mm3)) if len(component_sizes_mm3) > 0 else 0.0,
            "std_region_size_mm3": float(np.std(component_sizes_mm3)) if len(component_sizes_mm3) > 0 else 0.0,
            "total_voxels": int(np.sum(component_sizes)),
            "total_volume_mm3": float(np.sum(component_sizes_mm3)),
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
        all_stats_by_class = {1: [], 2: []}  # class 1 and 2 stats

        for sample_id in tqdm(sample_ids, desc=f"Analyzing {split} split", unit="sample"):
            # Read label
            label_file = labels_dir / f"{sample_id}.nii.gz"
            if not label_file.exists():
                print(f"Warning: Missing label for {sample_id}")
                continue

            # Load NIfTI file
            label_nib = nib.load(label_file)
            label_array = np.array(label_nib.get_fdata(), dtype=np.uint8)

            # Get voxel dimensions (assume first spatial dimension for spacing)
            pixdim = label_nib.header.get("pixdim")[1]  # First spatial dimension
            voxel_volume_mm3 = float(pixdim) ** 3

            # Analyze per class (1=FLAIR-only, 2=T1ce-enhancing)
            case_stats = {
                "case_id": sample_id,
                "split": split,
                "voxel_size_mm": float(pixdim),
                "classes": {},
            }

            for class_label in [1, 2]:
                comp_stats = self.count_components(label_array, class_label, voxel_volume_mm3)
                case_stats["classes"][str(class_label)] = comp_stats
                all_stats_by_class[class_label].append(comp_stats)

            # Save individual case JSON
            if output_dir:
                case_json = output_dir / f"{sample_id}.json"
                with open(case_json, "w") as f:
                    json.dump(case_stats, f, indent=2)

        # Aggregate statistics per class
        aggregate = {}
        for class_label in [1, 2]:
            stats_list = all_stats_by_class[class_label]
            if not stats_list:
                aggregate[f"class_{class_label}"] = {"num_cases_with_class": 0}
                continue

            # Count cases where this class appears
            num_cases_with_class = sum(1 for s in stats_list if s["num_regions"] > 0)

            # Collect all region counts and sizes
            all_region_counts = [s["num_regions"] for s in stats_list]
            all_region_sizes = []
            all_region_volumes = []
            for s in stats_list:
                all_region_sizes.extend(s["region_sizes_voxels"])
                all_region_volumes.extend(s["region_sizes_mm3"])

            aggregate[f"class_{class_label}"] = {
                "num_cases_with_class": num_cases_with_class,
                "mean_regions_per_case": float(np.mean(all_region_counts)) if all_region_counts else 0.0,
                "median_regions_per_case": float(np.median(all_region_counts)) if all_region_counts else 0.0,
                "min_regions": int(np.min(all_region_counts)) if all_region_counts else 0,
                "max_regions": int(np.max(all_region_counts)) if all_region_counts else 0,
                "std_regions_per_case": float(np.std(all_region_counts)) if all_region_counts else 0.0,
                "total_regions_all_cases": int(sum(all_region_counts)),
                "region_size_stats": {
                    "total_regions": int(len(all_region_sizes)),
                    "mean_voxels": float(np.mean(all_region_sizes)) if all_region_sizes else 0.0,
                    "median_voxels": float(np.median(all_region_sizes)) if all_region_sizes else 0.0,
                    "min_voxels": int(np.min(all_region_sizes)) if all_region_sizes else 0,
                    "max_voxels": int(np.max(all_region_sizes)) if all_region_sizes else 0,
                    "std_voxels": float(np.std(all_region_sizes)) if all_region_sizes else 0.0,
                    "mean_mm3": float(np.mean(all_region_volumes)) if all_region_volumes else 0.0,
                    "median_mm3": float(np.median(all_region_volumes)) if all_region_volumes else 0.0,
                    "min_mm3": float(np.min(all_region_volumes)) if all_region_volumes else 0.0,
                    "max_mm3": float(np.max(all_region_volumes)) if all_region_volumes else 0.0,
                    "std_mm3": float(np.std(all_region_volumes)) if all_region_volumes else 0.0,
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

        # Per-class statistics
        for class_name, class_key in [("Class 1 (FLAIR-only tumor)", "class_1"), ("Class 2 (T1ce-enhancing tumor)", "class_2")]:
            if class_key in agg:
                class_agg = agg[class_key]
                cases_with_class = class_agg["num_cases_with_class"]
                print(f"\n{class_name}:")
                print(f"  Cases with region: {cases_with_class}/{counts['train']}")
                if cases_with_class > 0:
                    print(f"  Avg regions per case: {class_agg['mean_regions_per_case']:.2f}")
                    print(f"  Median regions per case: {class_agg['median_regions_per_case']:.0f}")
                    print(f"  Region count range: {class_agg['min_regions']}-{class_agg['max_regions']}")
                    print(f"  Total regions: {class_agg['total_regions_all_cases']}")

                    comp_stats = class_agg["region_size_stats"]
                    print(f"  Region sizes:")
                    print(f"    Mean: {comp_stats['mean_voxels']:.2f} voxels ({comp_stats['mean_mm3']:.2f} mm³)")
                    print(f"    Median: {comp_stats['median_voxels']:.2f} voxels ({comp_stats['median_mm3']:.2f} mm³)")
                    print(f"    Range: {comp_stats['min_voxels']}-{comp_stats['max_voxels']} voxels ({comp_stats['min_mm3']:.2f}-{comp_stats['max_mm3']:.2f} mm³)")

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

        # Per-class statistics
        for class_name, class_key in [("Class 1 (FLAIR-only tumor)", "class_1"), ("Class 2 (T1ce-enhancing tumor)", "class_2")]:
            if class_key in agg:
                class_agg = agg[class_key]
                cases_with_class = class_agg["num_cases_with_class"]
                print(f"\n{class_name}:")
                print(f"  Cases with region: {cases_with_class}/{counts['test']}")
                if cases_with_class > 0:
                    print(f"  Avg regions per case: {class_agg['mean_regions_per_case']:.2f}")
                    print(f"  Median regions per case: {class_agg['median_regions_per_case']:.0f}")
                    print(f"  Region count range: {class_agg['min_regions']}-{class_agg['max_regions']}")
                    print(f"  Total regions: {class_agg['total_regions_all_cases']}")

                    comp_stats = class_agg["region_size_stats"]
                    print(f"  Region sizes:")
                    print(f"    Mean: {comp_stats['mean_voxels']:.2f} voxels ({comp_stats['mean_mm3']:.2f} mm³)")
                    print(f"    Median: {comp_stats['median_voxels']:.2f} voxels ({comp_stats['median_mm3']:.2f} mm³)")
                    print(f"    Range: {comp_stats['min_voxels']}-{comp_stats['max_voxels']} voxels ({comp_stats['min_mm3']:.2f}-{comp_stats['max_mm3']:.2f} mm³)")


def main():
    parser = argparse.ArgumentParser(
        description="Analyze MOTUM dataset statistics and region counts"
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
