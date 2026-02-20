#!/usr/bin/env python3
"""
Validate nnU-Net format datasets.

Usage:
    python verify.py Datset001_Cellpose
    python verify.py /absolute/path/to/Dataset002_Foo
"""

import json
import sys
from pathlib import Path

from PIL import Image
from tqdm import tqdm

# Color codes for output
GREEN = "\033[92m"
YELLOW = "\033[93m"
RED = "\033[91m"
RESET = "\033[0m"


class DatasetValidator:
    def __init__(self, dataset_path):
        self.dataset_path = Path(dataset_path).resolve()
        self.checks_passed = 0
        self.checks_warned = 0
        self.checks_failed = 0
        self.dataset_json = None
        self.dataset_json_data = None

    def _print_result(self, status, message):
        """Print a check result with color coding."""
        if status == "PASS":
            color = GREEN
            self.checks_passed += 1
        elif status == "WARN":
            color = YELLOW
            self.checks_warned += 1
        elif status == "FAIL":
            color = RED
            self.checks_failed += 1
        else:
            color = RESET
        print(f"{color}[{status}]{RESET} {message}")

    def run_all_checks(self):
        """Run all validation checks."""
        print(f"Validating dataset: {self.dataset_path}\n")

        # Check 1: dataset.json
        self._check_dataset_json()

        if self.dataset_json_data is None:
            print("\nCannot continue without valid dataset.json")
            return self._exit_code()

        # Check 2: Directory structure
        self._check_directory_structure()

        # Check 3: File counts
        self._check_file_counts()

        # Check 4: Case-level checks (sample subset)
        self._check_case_level()

        # Check 5: numTraining/numTest consistency
        self._check_consistency()

        return self._exit_code()

    def _exit_code(self):
        """Print summary and return exit code."""
        print(f"\n{'=' * 60}")
        print(
            f"SUMMARY: {self.checks_passed} passed, {self.checks_warned} warnings, {self.checks_failed} failures"
        )
        return 1 if self.checks_failed > 0 else 0

    def _check_dataset_json(self):
        """Check 1: Validate dataset.json file and structure."""
        print("--- Check 1: dataset.json ---")

        dataset_json_path = self.dataset_path / "dataset.json"
        if not dataset_json_path.exists():
            self._print_result("FAIL", "dataset.json not found")
            return
        self._print_result("PASS", "dataset.json exists")

        try:
            with open(dataset_json_path) as f:
                self.dataset_json_data = json.load(f)
        except json.JSONDecodeError as e:
            self._print_result("FAIL", f"dataset.json is not valid JSON: {e}")
            return
        self._print_result("PASS", "dataset.json is valid JSON")

        # Check required keys
        required_keys = {
            "name",
            "description",
            "reference",
            "citation",
            "license",
            "release",
            "tensorImageSize",
            "file_ending",
            "channel_names",
            "labels",
            "numTraining",
            "numTest",
            "numLabels",
        }
        missing_keys = required_keys - set(self.dataset_json_data.keys())
        if missing_keys:
            self._print_result("FAIL", f"Missing required keys: {missing_keys}")
        else:
            self._print_result("PASS", "All required keys present")

        # Check tensorImageSize
        tensor_size = self.dataset_json_data.get("tensorImageSize")
        if tensor_size not in ("2D", "3D"):
            self._print_result("FAIL", f"tensorImageSize must be '2D' or '3D', got: {tensor_size}")
        else:
            self._print_result("PASS", f"tensorImageSize is valid: {tensor_size}")

        # Check file_ending starts with dot
        file_ending = self.dataset_json_data.get("file_ending")
        if not file_ending or not file_ending.startswith("."):
            self._print_result("FAIL", f"file_ending must start with '.', got: {file_ending}")
        else:
            self._print_result("PASS", f"file_ending is valid: {file_ending}")

        # Check channel_names keys are sequential string ints
        channel_names = self.dataset_json_data.get("channel_names", {})
        if not self._is_sequential_int_keys(channel_names):
            self._print_result(
                "FAIL", "channel_names keys must be sequential string ints starting at '0'"
            )
        else:
            self._print_result(
                "PASS", f"channel_names keys are sequential: {list(channel_names.keys())}"
            )

        # Check labels keys are sequential string ints
        labels = self.dataset_json_data.get("labels", {})
        if not self._is_sequential_int_keys(labels):
            self._print_result("FAIL", "labels keys must be sequential string ints starting at '0'")
        else:
            self._print_result("PASS", f"labels keys are sequential: {list(labels.keys())}")

        # Check numLabels matches len(labels)
        num_labels = self.dataset_json_data.get("numLabels")
        expected_num_labels = len(labels)
        if num_labels != expected_num_labels:
            self._print_result(
                "FAIL",
                f"numLabels ({num_labels}) doesn't match len(labels) ({expected_num_labels})",
            )
        else:
            self._print_result("PASS", f"numLabels matches len(labels): {num_labels}")

    def _is_sequential_int_keys(self, d):
        """Check if dict keys are sequential string ints starting at '0'."""
        if not d:
            return False
        try:
            keys = [int(k) for k in d.keys()]
            keys.sort()
            return keys == list(range(len(keys))) and keys[0] == 0
        except (ValueError, TypeError):
            return False

    def _check_directory_structure(self):
        """Check 2: Validate directory structure."""
        print("\n--- Check 2: Directory Structure ---")

        images_tr = self.dataset_path / "imagesTr"
        labels_tr = self.dataset_path / "labelsTr"

        if not images_tr.exists():
            self._print_result("FAIL", "imagesTr/ not found")
        else:
            self._print_result("PASS", "imagesTr/ exists")

        if not labels_tr.exists():
            self._print_result("FAIL", "labelsTr/ not found")
        else:
            self._print_result("PASS", "labelsTr/ exists")

        images_ts = self.dataset_path / "imagesTs"
        labels_ts = self.dataset_path / "labelsTs"

        if not images_ts.exists():
            self._print_result("WARN", "imagesTs/ not found (optional but common)")
        else:
            self._print_result("PASS", "imagesTs/ exists")

        if not labels_ts.exists():
            self._print_result("WARN", "labelsTs/ not found (optional but common)")
        else:
            self._print_result("PASS", "labelsTs/ exists")

    def _check_file_counts(self):
        """Check 3: Validate file counts."""
        print("\n--- Check 3: File Counts ---")

        if self.dataset_json_data is None:
            return

        num_training = self.dataset_json_data.get("numTraining", 0)
        num_test = self.dataset_json_data.get("numTest", 0)
        num_channels = len(self.dataset_json_data.get("channel_names", {}))
        file_ending = self.dataset_json_data.get("file_ending", "")

        images_tr = self.dataset_path / "imagesTr"
        labels_tr = self.dataset_path / "labelsTr"
        images_ts = self.dataset_path / "imagesTs"
        labels_ts = self.dataset_path / "labelsTs"

        # Check imagesTr file count
        if images_tr.exists():
            images_tr_count = len(list(images_tr.glob(f"*{file_ending}")))
            expected_tr = num_training * num_channels
            if images_tr_count != expected_tr:
                self._print_result(
                    "FAIL",
                    f"imagesTr/ has {images_tr_count} files, expected {expected_tr} ({num_training} cases × {num_channels} channels)",
                )
            else:
                self._print_result("PASS", f"imagesTr/ file count correct: {images_tr_count}")

        # Check labelsTr file count
        if labels_tr.exists():
            labels_tr_count = len(list(labels_tr.glob(f"*{file_ending}")))
            if labels_tr_count != num_training:
                self._print_result(
                    "FAIL", f"labelsTr/ has {labels_tr_count} files, expected {num_training}"
                )
            else:
                self._print_result("PASS", f"labelsTr/ file count correct: {labels_tr_count}")

        # Check imagesTs file count (if present)
        if images_ts.exists():
            images_ts_count = len(list(images_ts.glob(f"*{file_ending}")))
            expected_ts = num_test * num_channels
            if images_ts_count != expected_ts:
                self._print_result(
                    "FAIL",
                    f"imagesTs/ has {images_ts_count} files, expected {expected_ts} ({num_test} cases × {num_channels} channels)",
                )
            else:
                self._print_result("PASS", f"imagesTs/ file count correct: {images_ts_count}")

        # Check labelsTs file count (if present)
        if labels_ts.exists():
            labels_ts_count = len(list(labels_ts.glob(f"*{file_ending}")))
            if labels_ts_count != num_test:
                self._print_result(
                    "FAIL", f"labelsTs/ has {labels_ts_count} files, expected {num_test}"
                )
            else:
                self._print_result("PASS", f"labelsTs/ file count correct: {labels_ts_count}")

    def _check_case_level(self):
        """Check 4: Case-level checks (all cases)."""
        print("\n--- Check 4: Case-Level Checks ---")

        if self.dataset_json_data is None:
            return

        file_ending = self.dataset_json_data.get("file_ending", "")
        channel_names = self.dataset_json_data.get("channel_names", {})
        labels_dict = self.dataset_json_data.get("labels", {})
        num_channels = len(channel_names)

        images_tr = self.dataset_path / "imagesTr"
        labels_tr = self.dataset_path / "labelsTr"

        if not images_tr.exists() or not labels_tr.exists():
            self._print_result(
                "FAIL", "Cannot perform case-level checks: imagesTr/ or labelsTr/ missing"
            )
            return

        # Get unique case IDs by finding *_0000{file_ending} files
        case_ids = self._get_unique_case_ids(images_tr, file_ending)

        if not case_ids:
            self._print_result("FAIL", "No cases found in imagesTr/")
            return

        case_ids = sorted(case_ids)

        # Check all cases with progress bar
        results = self._validate_all_cases(
            case_ids, images_tr, labels_tr, file_ending, num_channels, labels_dict
        )

        # Print summary results
        self._print_case_results(len(case_ids), results)

    def _validate_all_cases(
        self, case_ids, images_tr, labels_tr, file_ending, num_channels, labels_dict
    ):
        """Validate all cases and return results dict."""
        channels_failed = []
        labels_failed = []
        dims_failed = []
        channel_dims_failed = []
        invalid_label_values = []

        for case_id in tqdm(case_ids, desc="Checking cases", disable=False):
            self._validate_single_case(
                case_id,
                images_tr,
                labels_tr,
                file_ending,
                num_channels,
                labels_dict,
                channels_failed,
                labels_failed,
                dims_failed,
                channel_dims_failed,
                invalid_label_values,
            )

        return {
            "channels_failed": channels_failed,
            "labels_failed": labels_failed,
            "dims_failed": dims_failed,
            "channel_dims_failed": channel_dims_failed,
            "invalid_label_values": invalid_label_values,
        }

    def _validate_single_case(
        self,
        case_id,
        images_tr,
        labels_tr,
        file_ending,
        num_channels,
        labels_dict,
        channels_failed,
        labels_failed,
        dims_failed,
        channel_dims_failed,
        invalid_label_values,
    ):
        """Validate a single case."""
        # Check all channels exist
        for ch in range(num_channels):
            channel_file = images_tr / f"{case_id}_{ch:04d}{file_ending}"
            if not channel_file.exists():
                channels_failed.append((case_id, ch))

        # Check matching label exists
        label_file = labels_tr / f"{case_id}{file_ending}"
        if not label_file.exists():
            labels_failed.append(case_id)
            return

        # Check dimensions and label values
        try:
            self._check_case_dimensions(
                case_id,
                images_tr,
                label_file,
                file_ending,
                num_channels,
                dims_failed,
                channel_dims_failed,
            )
            self._check_label_values(label_file, case_id, labels_dict, invalid_label_values)
        except Exception as e:
            self._print_result("FAIL", f"Case {case_id}: error reading image/label: {e}")

    def _check_case_dimensions(
        self,
        case_id,
        images_tr,
        label_file,
        file_ending,
        num_channels,
        dims_failed,
        channel_dims_failed,
    ):
        """Check dimensions for a case."""
        label_img = Image.open(label_file)
        label_shape = label_img.size

        first_channel_file = images_tr / f"{case_id}_0000{file_ending}"
        if not first_channel_file.exists():
            return

        first_img = Image.open(first_channel_file)
        img_shape = first_img.size

        if img_shape != label_shape:
            dims_failed.append((case_id, img_shape, label_shape))

        # Check all channels have same dimensions
        for ch in range(num_channels):
            ch_file = images_tr / f"{case_id}_{ch:04d}{file_ending}"
            if ch_file.exists():
                ch_img = Image.open(ch_file)
                if ch_img.size != img_shape:
                    channel_dims_failed.append((case_id, ch, ch_img.size, img_shape))

    def _check_label_values(self, label_file, case_id, labels_dict, invalid_label_values):
        """Check label pixel values."""
        label_img = Image.open(label_file)
        label_array = label_img.get_flattened_data()
        max_label = max(label_array) if label_array else 0
        if max_label >= len(labels_dict) and max_label != 255:
            invalid_label_values.append((case_id, max_label, len(labels_dict) - 1))

    def _print_case_results(self, num_cases, results):
        """Print summary of case-level validation results."""
        if not results["channels_failed"]:
            self._print_result("PASS", f"All declared channels present (checked {num_cases} cases)")
        else:
            self._print_result(
                "FAIL", f"Missing channels in {len(results['channels_failed'])} case(s)"
            )

        if not results["labels_failed"]:
            self._print_result(
                "PASS", f"All cases have matching labels (checked {num_cases} cases)"
            )
        else:
            self._print_result(
                "FAIL", f"{len(results['labels_failed'])} case(s) have no matching label"
            )

        if not results["dims_failed"]:
            self._print_result(
                "PASS", f"Image and label dimensions match (checked {num_cases} cases)"
            )
        else:
            self._print_result(
                "FAIL", f"Dimension mismatch in {len(results['dims_failed'])} case(s)"
            )

        if not results["channel_dims_failed"]:
            self._print_result(
                "PASS", f"All channels have same dimensions (checked {num_cases} cases)"
            )
        else:
            self._print_result(
                "FAIL",
                f"Channel dimension inconsistency in {len(results['channel_dims_failed'])} case(s)",
            )

        if not results["invalid_label_values"]:
            self._print_result("PASS", f"Label pixel values are valid (checked {num_cases} cases)")

    def _get_unique_case_ids(self, directory, file_ending):
        """Extract unique case IDs by finding *_0000{file_ending} files."""
        case_ids = set()
        for file in directory.glob(f"*_0000{file_ending}"):
            # Remove _0000{file_ending} to get case_id
            case_id = file.name[: -len(f"_0000{file_ending}")]
            case_ids.add(case_id)
        return case_ids

    def _check_consistency(self):
        """Check 5: numTraining/numTest consistency."""
        print("\n--- Check 5: Consistency Check ---")

        if self.dataset_json_data is None:
            return

        num_training = self.dataset_json_data.get("numTraining", 0)
        num_test = self.dataset_json_data.get("numTest", 0)
        file_ending = self.dataset_json_data.get("file_ending", "")

        images_tr = self.dataset_path / "imagesTr"
        images_ts = self.dataset_path / "imagesTs"

        # Check training cases
        if images_tr.exists():
            training_cases = len(self._get_unique_case_ids(images_tr, file_ending))
            if training_cases != num_training:
                self._print_result(
                    "FAIL",
                    f"Unique training cases ({training_cases}) != numTraining ({num_training})",
                )
            else:
                self._print_result(
                    "PASS", f"Unique training cases match numTraining: {training_cases}"
                )

        # Check test cases (if present)
        if images_ts.exists():
            test_cases = len(self._get_unique_case_ids(images_ts, file_ending))
            if test_cases != num_test:
                self._print_result(
                    "FAIL", f"Unique test cases ({test_cases}) != numTest ({num_test})"
                )
            else:
                self._print_result("PASS", f"Unique test cases match numTest: {test_cases}")


def main():
    if len(sys.argv) < 2:
        print("Usage: python verify.py <dataset_path>")
        print("  python verify.py Datset001_Cellpose")
        print("  python verify.py /absolute/path/to/Dataset002_Foo")
        sys.exit(1)

    dataset_path = sys.argv[1]

    # Resolve path: relative to CWD or absolute
    resolved_path = Path(dataset_path).resolve()

    if not resolved_path.exists():
        print(f"Error: Dataset path does not exist: {dataset_path}")
        sys.exit(1)

    validator = DatasetValidator(resolved_path)
    exit_code = validator.run_all_checks()
    sys.exit(exit_code)


if __name__ == "__main__":
    main()
