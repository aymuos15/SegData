#!/usr/bin/env python3
"""
Test that each dataset has all the basic required files and structure.

This ensures consistency across all datasets in the collection.

Run with: pytest test_datasets.py -v
"""

import json
from pathlib import Path
import pytest


REQUIRED_FILES = [
    "dataset.json",
    "setup.py",
]

OPTIONAL_FILES = [
    "info.md",
    "get.md",
]

REQUIRED_DIRS = [
    "imagesTr",
    "labelsTr",
    "imagesTs",
    "labelsTs",
]


def get_dataset_dirs():
    """Get all dataset directories."""
    raw_dir = Path(__file__).parent.parent / "raw"
    return sorted([d for d in raw_dir.iterdir() if d.is_dir()])


@pytest.mark.parametrize("dataset_dir", get_dataset_dirs(), ids=lambda d: d.name)
class TestDatasets:
    """Test suite for dataset consistency."""

    def test_required_files_exist(self, dataset_dir):
        """Each dataset should have all required files."""
        missing = []
        for filename in REQUIRED_FILES:
            filepath = dataset_dir / filename
            if not filepath.exists():
                missing.append(filename)

        assert not missing, f"Missing required files: {missing}"

    def test_optional_files_exist(self, dataset_dir):
        """Each dataset should have at least one optional documentation file."""
        optional_exist = [f for f in OPTIONAL_FILES if (dataset_dir / f).exists()]
        assert optional_exist, f"Missing documentation files: must have one of {OPTIONAL_FILES}"

    def test_dataset_json_valid(self, dataset_dir):
        """dataset.json should be valid JSON with required keys."""
        dataset_json = dataset_dir / "dataset.json"
        assert dataset_json.exists(), "dataset.json not found"

        with open(dataset_json) as f:
            data = json.load(f)

        # Check required keys
        required_keys = {"name", "description", "tensorImageSize", "labels", "numLabels"}
        missing_keys = required_keys - set(data.keys())
        assert not missing_keys, f"dataset.json missing keys: {missing_keys}"

    def test_data_structure(self, dataset_dir):
        """If data directories exist, they should not be empty."""
        all_exist = all((dataset_dir / d).exists() for d in REQUIRED_DIRS)

        if all_exist:
            # All dirs exist - check they're not empty
            for dirname in REQUIRED_DIRS:
                dirpath = dataset_dir / dirname
                files = list(dirpath.glob("*"))
                assert files, f"Directory is empty: {dirname}"
        else:
            # If dirs don't exist, setup.py hasn't been run yet (that's ok)
            missing = [d for d in REQUIRED_DIRS if not (dataset_dir / d).exists()]
            # It's fine if none exist (setup not run yet)
            # But error if only some exist (incomplete conversion)
            partial = len(missing) > 0 and len(missing) < 4
            assert not partial, f"Incomplete data conversion: missing {missing}"
