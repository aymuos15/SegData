#!/usr/bin/env python3
"""Dataset statistics and connected component analysis for BTXRD 2D segmentation.

GPU-accelerated using CuPy for fast connected component labeling.

Analyzes each case in the dataset and provides:
- Number of training and test samples
- Average image dimensions
- Connected component count per image (tumor counts)
- Tumor size statistics
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

import cupy as cp
import numpy as np
from cupyx.scipy.ndimage import (
    generate_binary_structure as cp_generate_binary_structure,
)
from cupyx.scipy.ndimage import label as cp_label
from PIL import Image
from tqdm import tqdm


class DatasetAnalyzer:
    """Analyze dataset structure and connected components."""

    def __init__(self, dataset_dir: str):
        """Initialize analyzer with dataset directory."""
        self.dataset_dir = Path(dataset_dir)
        self.images_tr = self.dataset_dir / "imagesTr"
        self.labels_tr = self.dataset_dir / "labelsTr"
        self.images_ts = self.dataset_dir / "imagesTs"
        self.labels_ts = self.dataset_dir / "labelsTs"

    def count_samples(self) -> Dict[str, int]:
        """Count number of training and test samples."""
        train_count = len(list(self.images_tr.glob("*.png")))
        test_count = len(list(self.images_ts.glob("*.png")))
        return {
            "train": train_count,
            "test": test_count,
            "total": train_count + test_count,
        }

    def get_image_dimensions(self, sample_ids: List[str], images_dir: Path) -> Dict:
        """Calculate average dimensions across images."""
        heights = []
        widths = []

        for sample_id in sample_ids:
            img_file = images_dir / f"{sample_id}_0000.png"
            if img_file.exists():
                img = Image.open(img_file)
                w, h = img.size
                widths.append(w)
                heights.append(h)

        return {
            "mean_height": float(np.mean(heights)),
            "mean_width": float(np.mean(widths)),
            "min_height": int(np.min(heights)),
            "max_height": int(np.max(heights)),
            "min_width": int(np.min(widths)),
            "max_width": int(np.max(widths)),
            "std_height": float(np.std(heights)),
            "std_width": float(np.std(widths)),
        }

    def count_tumors(self, label_array: np.ndarray) -> Dict:
        """Count connected components (tumors) in a binary mask using 8-connectivity (GPU accelerated)."""
        label_gpu = cp.asarray(label_array.astype(cp.uint8))
        structure = cp_generate_binary_structure(2, 2)

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

        return {
            "num_tumors": int(num_components),
            "tumor_sizes_pixels": component_sizes.tolist(),
            "mean_tumor_size": float(np.mean(component_sizes)) if len(component_sizes) > 0 else 0.0,
            "median_tumor_size": float(np.median(component_sizes)) if len(component_sizes) > 0 else 0.0,
            "min_tumor_size": int(np.min(component_sizes)) if len(component_sizes) > 0 else 0,
            "max_tumor_size": int(np.max(component_sizes)) if len(component_sizes) > 0 else 0,
            "std_tumor_size": float(np.std(component_sizes)) if len(component_sizes) > 0 else 0.0,
            "total_tumor_pixels": int(np.sum(component_sizes)),
        }

    def analyze_split(self, split: str, output_dir: Path | None = None) -> Dict:
        """Analyze training or test split."""
        if split == "train":
            images_dir = self.images_tr
            labels_dir = self.labels_tr
        else:
            images_dir = self.images_ts
            labels_dir = self.labels_ts

        # Get unique sample IDs
        image_files = sorted(images_dir.glob("*_0000.png"))
        sample_ids = [f.stem.replace("_0000", "") for f in image_files]

        if not sample_ids:
            return {
                "num_samples": 0,
                "image_dims": {},
                "aggregate": {},
            }

        # Get image dimensions
        image_dims = self.get_image_dimensions(sample_ids, images_dir)

        # Analyze each sample
        all_tumor_counts = []
        all_tumor_sizes = []

        for sample_id in tqdm(sample_ids, desc=f"Analyzing {split} split", unit="sample"):
            # Read label
            label_file = labels_dir / f"{sample_id}.png"
            if not label_file.exists():
                print(f"Warning: Missing label for {sample_id}")
                continue

            label_img = Image.open(label_file)
            label_array = np.array(label_img) > 0  # Binarize

            # Count tumors
            tumor_stats = self.count_tumors(label_array)

            # Save individual case JSON
            if output_dir:
                case_output = {
                    "case_id": sample_id,
                    "split": split,
                    "stats": tumor_stats,
                }
                case_json = output_dir / f"{sample_id}.json"
                with open(case_json, "w") as f:
                    json.dump(case_output, f, indent=2)

            all_tumor_counts.append(tumor_stats["num_tumors"])
            all_tumor_sizes.extend(tumor_stats["tumor_sizes_pixels"])

        # Aggregate statistics
        aggregate = {
            "mean_tumors_per_image": float(np.mean(all_tumor_counts)) if all_tumor_counts else 0.0,
            "median_tumors_per_image": float(np.median(all_tumor_counts)) if all_tumor_counts else 0.0,
            "min_tumors": int(np.min(all_tumor_counts)) if all_tumor_counts else 0,
            "max_tumors": int(np.max(all_tumor_counts)) if all_tumor_counts else 0,
            "std_tumors_per_image": float(np.std(all_tumor_counts)) if all_tumor_counts else 0.0,
            "total_tumors_all_samples": int(sum(all_tumor_counts)),
            "tumor_size_stats": {
                "total_tumors": int(len(all_tumor_sizes)),
                "mean_pixels": float(np.mean(all_tumor_sizes)) if all_tumor_sizes else 0.0,
                "median_pixels": float(np.median(all_tumor_sizes)) if all_tumor_sizes else 0.0,
                "min_pixels": int(np.min(all_tumor_sizes)) if all_tumor_sizes else 0,
                "max_pixels": int(np.max(all_tumor_sizes)) if all_tumor_sizes else 0,
                "std_pixels": float(np.std(all_tumor_sizes)) if all_tumor_sizes else 0.0,
            },
        }

        return {
            "num_samples": len(sample_ids),
            "image_dims": image_dims,
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
        dims = train_data["image_dims"]
        agg = train_data["aggregate"]

        print(f"\nImage Dimensions:")
        print(f"  Mean: {dims['mean_height']:.1f} x {dims['mean_width']:.1f} pixels")
        print(f"  Range: {dims['min_height']}-{dims['max_height']} x {dims['min_width']}-{dims['max_width']} pixels")
        print(f"  Std Dev: +/-{dims['std_height']:.1f} x +/-{dims['std_width']:.1f} pixels")

        print(f"\nTumor Statistics:")
        print(f"  Avg tumors per image: {agg['mean_tumors_per_image']:.2f}")
        print(f"  Median tumors per image: {agg['median_tumors_per_image']:.0f}")
        print(f"  Tumor count range: {agg['min_tumors']}-{agg['max_tumors']}")
        print(f"  Total tumors: {agg['total_tumors_all_samples']}")

        tumor_stats = agg["tumor_size_stats"]
        print(f"\nTumor Sizes:")
        print(f"  Total components: {tumor_stats['total_tumors']}")
        print(f"  Mean: {tumor_stats['mean_pixels']:.2f} pixels")
        print(f"  Median: {tumor_stats['median_pixels']:.2f} pixels")
        print(f"  Range: {tumor_stats['min_pixels']}-{tumor_stats['max_pixels']} pixels")
        print(f"  Std Dev: {tumor_stats['std_pixels']:.2f} pixels")

    # Test split
    if output["test"]["num_samples"] > 0:
        print("\n" + "=" * 80)
        print("TEST SPLIT")
        print("=" * 80)
        test_data = output["test"]
        dims = test_data["image_dims"]
        agg = test_data["aggregate"]

        print(f"\nImage Dimensions:")
        print(f"  Mean: {dims['mean_height']:.1f} x {dims['mean_width']:.1f} pixels")
        print(f"  Range: {dims['min_height']}-{dims['max_height']} x {dims['min_width']}-{dims['max_width']} pixels")
        print(f"  Std Dev: +/-{dims['std_height']:.1f} x +/-{dims['std_width']:.1f} pixels")

        print(f"\nTumor Statistics:")
        print(f"  Avg tumors per image: {agg['mean_tumors_per_image']:.2f}")
        print(f"  Median tumors per image: {agg['median_tumors_per_image']:.0f}")
        print(f"  Tumor count range: {agg['min_tumors']}-{agg['max_tumors']}")
        print(f"  Total tumors: {agg['total_tumors_all_samples']}")

        tumor_stats = agg["tumor_size_stats"]
        print(f"\nTumor Sizes:")
        print(f"  Total components: {tumor_stats['total_tumors']}")
        print(f"  Mean: {tumor_stats['mean_pixels']:.2f} pixels")
        print(f"  Median: {tumor_stats['median_pixels']:.2f} pixels")
        print(f"  Range: {tumor_stats['min_pixels']}-{tumor_stats['max_pixels']} pixels")
        print(f"  Std Dev: {tumor_stats['std_pixels']:.2f} pixels")


def main():
    parser = argparse.ArgumentParser(
        description="Analyze BTXRD dataset statistics and tumor counts"
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
