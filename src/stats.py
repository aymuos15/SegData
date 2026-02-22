#!/usr/bin/env python3
"""
Unified dataset statistics and connected component analysis.

Auto-detects format (PNG/NIfTI), dimensionality (2D/3D), and label classes from dataset.json.
Computes GPU-accelerated connected component analysis and saves per-case and aggregate statistics.

Usage:
    python stats.py <dataset_path>
    python stats.py raw/Dataset003_MOTUM
    python stats.py Datset001_Cellpose
"""

import json
import sys
from pathlib import Path
from typing import Dict, List, Tuple

import cupy as cp
import nibabel as nib
import numpy as np
from cupyx.scipy.ndimage import generate_binary_structure as cp_generate_binary_structure
from cupyx.scipy.ndimage import label as cp_label
from PIL import Image
from tqdm import tqdm


def load_dataset_meta(dataset_dir: Path) -> Dict:
    """Load dataset metadata from dataset.json."""
    dataset_json = dataset_dir / "dataset.json"
    with open(dataset_json) as f:
        return json.load(f)


class DatasetAnalyzer:
    """Analyze dataset structure and connected components."""

    def __init__(self, dataset_dir: Path, meta: Dict):
        """Initialize analyzer."""
        self.dataset_dir = dataset_dir
        self.meta = meta
        self.is_3d = meta.get("tensorImageSize") == "3D"
        self.file_ending = meta.get("file_ending", ".png")

        self.images_tr = dataset_dir / "imagesTr"
        self.labels_tr = dataset_dir / "labelsTr"
        self.images_ts = dataset_dir / "imagesTs"
        self.labels_ts = dataset_dir / "labelsTs"

    def count_samples(self) -> Dict[str, int]:
        """Count number of training and test samples."""
        if self.is_3d:
            pattern = f"*_0000{self.file_ending}"
        else:
            pattern = f"*_0000{self.file_ending}"

        train_count = len(list(self.images_tr.glob(pattern)))
        test_count = len(list(self.images_ts.glob(pattern)))

        return {
            "train": train_count,
            "test": test_count,
            "total": train_count + test_count,
        }

    def get_image_dimensions_2d(self, sample_ids: List[str], images_dir: Path) -> Dict:
        """Calculate average dimensions across 2D images."""
        heights = []
        widths = []

        for sample_id in sample_ids:
            img_file = images_dir / f"{sample_id}_0000{self.file_ending}"
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

    def get_image_dimensions_3d(self, sample_ids: List[str], images_dir: Path) -> Dict:
        """Calculate average dimensions across 3D volumes."""
        shapes = []

        for sample_id in sample_ids:
            img_file = images_dir / f"{sample_id}_0000{self.file_ending}"
            img = nib.load(img_file)
            shape = img.shape
            shapes.append(shape)

        shapes = np.array(shapes)
        return {
            "mean_shape": [float(np.mean(shapes[:, i])) for i in range(3)],
            "min_shape": [int(np.min(shapes[:, i])) for i in range(3)],
            "max_shape": [int(np.max(shapes[:, i])) for i in range(3)],
            "std_shape": [float(np.std(shapes[:, i])) for i in range(3)],
        }

    def count_components(
        self,
        label_array: np.ndarray,
        class_label: int,
        connectivity_struct,
        voxel_volume_mm3: float = 1.0,
    ) -> Dict:
        """Count connected components for a specific class."""
        binary_mask = (label_array == class_label).astype(np.uint8)

        label_gpu = cp.asarray(binary_mask)
        labeled_gpu, num_components = cp_label(label_gpu, structure=connectivity_struct)
        labeled_array = cp.asnumpy(labeled_gpu)

        del label_gpu, labeled_gpu
        cp.get_default_memory_pool().free_all_blocks()

        component_sizes = np.bincount(labeled_array.ravel())[1:]
        component_sizes = component_sizes[component_sizes > 0]

        if len(component_sizes) == 0:
            # No components found for this class
            return {
                "num_components": 0,
                "component_sizes": [],
                "mean_size": 0,
                "median_size": 0,
                "min_size": 0,
                "max_size": 0,
                "std_size": 0,
                "total_size": 0,
            }

        result = {
            "num_components": int(num_components),
            "component_sizes": component_sizes.tolist(),
            "mean_size": float(np.mean(component_sizes)),
            "median_size": float(np.median(component_sizes)),
            "min_size": int(np.min(component_sizes)),
            "max_size": int(np.max(component_sizes)),
            "std_size": float(np.std(component_sizes)),
            "total_size": int(np.sum(component_sizes)),
        }

        # For 3D, also compute mm³
        if self.is_3d:
            component_sizes_mm3 = component_sizes * voxel_volume_mm3
            result.update(
                {
                    "component_sizes_mm3": component_sizes_mm3.tolist(),
                    "mean_size_mm3": float(np.mean(component_sizes_mm3)),
                    "median_size_mm3": float(np.median(component_sizes_mm3)),
                    "min_size_mm3": float(np.min(component_sizes_mm3)),
                    "max_size_mm3": float(np.max(component_sizes_mm3)),
                    "std_size_mm3": float(np.std(component_sizes_mm3)),
                    "total_volume_mm3": float(np.sum(component_sizes_mm3)),
                }
            )

        return result

    def _process_sample_stats(
        self,
        sample_id: str,
        label_file: Path,
        classes_to_analyze: List[int],
        num_labels: int,
        connectivity_struct,
        labels_dict: Dict,
        output_dir: Path,
        split: str,
    ) -> Tuple[Dict, Dict]:
        """Process statistics for a single sample."""
        if self.is_3d:
            label_nib = nib.load(label_file)
            label_array = np.array(label_nib.get_fdata(), dtype=np.uint8)
            # Get voxel size
            pixdim = label_nib.header.get("pixdim")[1]
            voxel_volume_mm3 = float(pixdim) ** 3
        else:
            label_array = np.array(Image.open(label_file), dtype=np.uint8)
            voxel_volume_mm3 = 1.0

        case_stats = {
            "case_id": sample_id,
            "split": split,
            "stats": {},
        }

        # Add voxel size for 3D
        if self.is_3d:
            case_stats["voxel_size_mm"] = float(pixdim)

        # Analyze per class
        all_stats_by_class = {}
        for class_label in classes_to_analyze:
            comp_stats = self.count_components(
                label_array, class_label, connectivity_struct, voxel_volume_mm3
            )

            # Use label name or default key
            if num_labels == 2:
                key = labels_dict.get("1", "foreground")
            else:
                key = labels_dict.get(str(class_label), f"class_{class_label}")

            case_stats["stats"][key] = comp_stats
            all_stats_by_class[key] = comp_stats

        # Save individual case JSON
        case_json = output_dir / f"{sample_id}_{split}.json"
        with open(case_json, "w") as f:
            json.dump(case_stats, f, indent=2)

        return case_stats, all_stats_by_class

    def _compute_aggregate_stats(
        self,
        all_stats_by_class: Dict[str, List[Dict]],
        num_labels: int,
        labels_dict: Dict,
    ) -> Dict:
        """Compute aggregate statistics across all samples."""
        aggregate = {}

        for class_label in range(1, num_labels if num_labels > 2 else 2):
            if num_labels == 2:
                key = labels_dict.get("1", "foreground")
            else:
                key = labels_dict.get(str(class_label), f"class_{class_label}")

            stats_list = all_stats_by_class.get(key, [])

            if not stats_list:
                continue

            # Count cases with this class
            num_cases_with_class = sum(1 for s in stats_list if s["num_components"] > 0)

            # Collect all component counts and sizes
            all_component_counts = [s["num_components"] for s in stats_list]
            all_component_sizes = []
            for s in stats_list:
                all_component_sizes.extend(s["component_sizes"])

            agg = {
                "num_cases_with_class": num_cases_with_class,
                "mean_components_per_case": float(np.mean(all_component_counts)),
                "median_components_per_case": float(np.median(all_component_counts)),
                "min_components": int(np.min(all_component_counts)),
                "max_components": int(np.max(all_component_counts)),
                "std_components_per_case": float(np.std(all_component_counts)),
                "total_components_all_cases": int(sum(all_component_counts)),
            }

            if len(all_component_sizes) > 0:
                agg["component_size_stats"] = {
                    "total_components": int(len(all_component_sizes)),
                    "mean_size": float(np.mean(all_component_sizes)),
                    "median_size": float(np.median(all_component_sizes)),
                    "min_size": int(np.min(all_component_sizes)),
                    "max_size": int(np.max(all_component_sizes)),
                    "std_size": float(np.std(all_component_sizes)),
                }

                # Add mm³ stats for 3D
                if self.is_3d:
                    all_component_sizes_mm3 = []
                    for s in stats_list:
                        all_component_sizes_mm3.extend(s.get("component_sizes_mm3", []))

                    if all_component_sizes_mm3:
                        agg["component_size_stats"]["mean_size_mm3"] = float(
                            np.mean(all_component_sizes_mm3)
                        )
                        agg["component_size_stats"]["median_size_mm3"] = float(
                            np.median(all_component_sizes_mm3)
                        )
                        agg["component_size_stats"]["min_size_mm3"] = float(
                            np.min(all_component_sizes_mm3)
                        )
                        agg["component_size_stats"]["max_size_mm3"] = float(
                            np.max(all_component_sizes_mm3)
                        )
                        agg["component_size_stats"]["std_size_mm3"] = float(
                            np.std(all_component_sizes_mm3)
                        )

            aggregate[key] = agg

        return aggregate

    def analyze_split(self, split: str, output_dir: Path) -> Dict:
        """Analyze training or test split."""
        split_dirs = {
            "train": (self.images_tr, self.labels_tr),
            "test": (self.images_ts, self.labels_ts),
        }
        images_dir, labels_dir = split_dirs[split]

        # Get sample IDs
        pattern = f"*_0000{self.file_ending}"
        image_files = sorted(images_dir.glob(pattern))
        sample_ids = [f.name.replace(f"_0000{self.file_ending}", "") for f in image_files]

        # Get image dimensions
        if self.is_3d:
            image_dims = self.get_image_dimensions_3d(sample_ids, images_dir)
        else:
            image_dims = self.get_image_dimensions_2d(sample_ids, images_dir)

        # Prepare connectivity structure
        if self.is_3d:
            connectivity_struct = cp_generate_binary_structure(3, 3)  # 26-connectivity
        else:
            connectivity_struct = cp_generate_binary_structure(2, 2)  # 8-connectivity

        # Determine which classes to analyze
        num_labels = self.meta.get("numLabels", 2)
        if num_labels == 2:
            classes_to_analyze = [1]  # Binary: just foreground
        else:
            classes_to_analyze = list(range(1, num_labels))

        labels_dict = self.meta.get("labels", {})

        # Analyze each sample and collect stats by class
        all_stats_by_class: Dict[str, List[Dict]] = {
            (
                labels_dict.get("1", "foreground")
                if num_labels == 2
                else labels_dict.get(str(c), f"class_{c}")
            ): []
            for c in classes_to_analyze
        }

        for sample_id in tqdm(sample_ids, desc=f"Analyzing {split} split", unit="sample"):
            label_file = labels_dir / f"{sample_id}{self.file_ending}"
            _, class_stats = self._process_sample_stats(
                sample_id,
                label_file,
                classes_to_analyze,
                num_labels,
                connectivity_struct,
                labels_dict,
                output_dir,
                split,
            )

            # Accumulate stats by class
            for key, stats in class_stats.items():
                all_stats_by_class[key].append(stats)

        # Compute aggregate statistics
        aggregate = self._compute_aggregate_stats(all_stats_by_class, num_labels, labels_dict)

        return {
            "num_samples": len(sample_ids),
            "image_dims": image_dims,
            "aggregate": aggregate,
        }


def analyze_dataset(dataset_dir: Path, output_dir: str) -> Dict:
    """Analyze complete dataset."""
    meta = load_dataset_meta(dataset_dir)
    analyzer = DatasetAnalyzer(dataset_dir, meta)

    # Count samples
    sample_counts = analyzer.count_samples()

    # Create output directory
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # Analyze each split
    train_analysis = analyzer.analyze_split("train", output_path)
    test_analysis = analyzer.analyze_split("test", output_path)

    # Compile output
    output = {
        "dataset_info": {
            "dataset_dir": str(dataset_dir),
            "dataset_name": meta.get("name", "Unknown"),
            "sample_counts": sample_counts,
            "is_3d": analyzer.is_3d,
            "num_labels": meta.get("numLabels", 2),
        },
        "train": train_analysis,
        "test": test_analysis,
    }

    # Save aggregate stats
    aggregate_json = output_path / "aggregate_stats.json"
    with open(aggregate_json, "w") as f:
        json.dump(output, f, indent=2)

    print(f"\nAggregate stats saved to {aggregate_json}")
    print(f"Per-case stats saved to {output_path}/*_train.json and *_test.json")

    return output


def print_summary(output: Dict) -> None:
    """Print summary statistics to console."""
    dataset_info = output["dataset_info"]
    counts = dataset_info["sample_counts"]

    print("\n" + "=" * 80)
    print("DATASET SUMMARY")
    print("=" * 80)
    print(f"Dataset: {dataset_info.get('dataset_name', 'Unknown')}")
    print(f"Type: {'3D' if dataset_info['is_3d'] else '2D'}")
    print(f"Training samples: {counts['train']}")
    print(f"Test samples: {counts['test']}")
    print(f"Total samples: {counts['total']}")

    # Training split
    print("\n" + "=" * 80)
    print("TRAINING SPLIT")
    print("=" * 80)
    train_data = output["train"]

    print("\nImage Dimensions:")
    dims = train_data["image_dims"]
    if "mean_shape" in dims:
        print(f"  Mean shape: {dims['mean_shape']}")
        print(f"  Range: {dims['min_shape']} - {dims['max_shape']}")
        print(f"  Std Dev: {dims['std_shape']}")
    else:
        print(f"  Mean: {dims['mean_height']:.1f} × {dims['mean_width']:.1f} pixels")
        print(
            f"  Range: {dims['min_height']}-{dims['max_height']} × {dims['min_width']}-{dims['max_width']} pixels"
        )
        print(f"  Std Dev: ±{dims['std_height']:.1f} × ±{dims['std_width']:.1f} pixels")

    agg = train_data["aggregate"]
    for class_name, class_stats in agg.items():
        print(f"\n{class_name.replace('_', ' ').title()} Statistics:")
        print(f"  Cases with class: {class_stats['num_cases_with_class']}/{counts['train']}")
        print(f"  Avg components per case: {class_stats['mean_components_per_case']:.2f}")
        print(f"  Median components per case: {class_stats['median_components_per_case']:.0f}")
        print(
            f"  Component count range: {class_stats['min_components']}-{class_stats['max_components']}"
        )
        print(f"  Total components: {class_stats['total_components_all_cases']}")

        if "component_size_stats" in class_stats:
            comp_stats = class_stats["component_size_stats"]
            if dataset_info["is_3d"] and "mean_size_mm3" in comp_stats:
                print("  Component sizes:")
                print(
                    f"    Mean: {comp_stats['mean_size']:.2f} voxels ({comp_stats['mean_size_mm3']:.2f} mm³)"
                )
                print(
                    f"    Median: {comp_stats['median_size']:.2f} voxels ({comp_stats['median_size_mm3']:.2f} mm³)"
                )
                print(f"    Range: {comp_stats['min_size']}-{comp_stats['max_size']} voxels")
            else:
                print("  Component sizes:")
                print(f"    Mean: {comp_stats['mean_size']:.2f}")
                print(f"    Median: {comp_stats['median_size']:.2f}")
                print(f"    Range: {comp_stats['min_size']}-{comp_stats['max_size']}")

    # Test split
    print("\n" + "=" * 80)
    print("TEST SPLIT")
    print("=" * 80)
    test_data = output["test"]

    print("\nImage Dimensions:")
    dims = test_data["image_dims"]
    if "mean_shape" in dims:
        print(f"  Mean shape: {dims['mean_shape']}")
        print(f"  Range: {dims['min_shape']} - {dims['max_shape']}")
        print(f"  Std Dev: {dims['std_shape']}")
    else:
        print(f"  Mean: {dims['mean_height']:.1f} × {dims['mean_width']:.1f} pixels")
        print(
            f"  Range: {dims['min_height']}-{dims['max_height']} × {dims['min_width']}-{dims['max_width']} pixels"
        )
        print(f"  Std Dev: ±{dims['std_height']:.1f} × ±{dims['std_width']:.1f} pixels")

    agg = test_data["aggregate"]
    for class_name, class_stats in agg.items():
        print(f"\n{class_name.replace('_', ' ').title()} Statistics:")
        print(f"  Cases with class: {class_stats['num_cases_with_class']}/{counts['test']}")
        print(f"  Avg components per case: {class_stats['mean_components_per_case']:.2f}")
        print(f"  Median components per case: {class_stats['median_components_per_case']:.0f}")
        print(
            f"  Component count range: {class_stats['min_components']}-{class_stats['max_components']}"
        )
        print(f"  Total components: {class_stats['total_components_all_cases']}")

        if "component_size_stats" in class_stats:
            comp_stats = class_stats["component_size_stats"]
            if dataset_info["is_3d"] and "mean_size_mm3" in comp_stats:
                print("  Component sizes:")
                print(
                    f"    Mean: {comp_stats['mean_size']:.2f} voxels ({comp_stats['mean_size_mm3']:.2f} mm³)"
                )
                print(
                    f"    Median: {comp_stats['median_size']:.2f} voxels ({comp_stats['median_size_mm3']:.2f} mm³)"
                )
                print(f"    Range: {comp_stats['min_size']}-{comp_stats['max_size']} voxels")
            else:
                print("  Component sizes:")
                print(f"    Mean: {comp_stats['mean_size']:.2f}")
                print(f"    Median: {comp_stats['median_size']:.2f}")
                print(f"    Range: {comp_stats['min_size']}-{comp_stats['max_size']}")


def main():
    """Main entry point."""
    if len(sys.argv) < 2:
        print("Usage: python stats.py <dataset_path>")
        print("Example: python stats.py raw/Dataset003_MOTUM")
        sys.exit(1)

    dataset_arg = sys.argv[1]

    # Resolve dataset path
    dataset_dir = Path(dataset_arg)
    if not dataset_dir.is_absolute():
        dataset_dir = Path.cwd() / dataset_dir

    if not dataset_dir.exists():
        print(f"Error: Dataset directory not found: {dataset_dir}")
        sys.exit(1)

    try:
        output = analyze_dataset(dataset_dir, "results")
        print_summary(output)
    except Exception as e:
        print(f"Error: {e}")
        import traceback

        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
