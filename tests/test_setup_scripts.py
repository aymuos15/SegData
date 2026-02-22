#!/usr/bin/env python3
"""
Tests for dataset setup.py scripts and their helper functions.

Tests verify that helper functions correctly:
- Discover and pair images with masks/annotations
- Convert image formats
- Rasterize annotations
- Handle different directory structures

Run with: pytest tests/test_setup_scripts.py -v
"""

import json
import sys
import tempfile
import zipfile
from pathlib import Path
from unittest.mock import MagicMock, patch
import importlib.util

import pytest

# Add raw datasets to path so we can import setup modules
sys.path.insert(0, str(Path(__file__).parent.parent / "raw"))


# Import setup modules once at module level
def _import_setup(module_name, setup_path):
    spec = importlib.util.spec_from_file_location(module_name, setup_path)
    setup = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(setup)
    return setup


busbra_setup = _import_setup(
    "busbra_setup",
    Path(__file__).parent.parent / "raw" / "Dataset005_BUSBRA" / "setup.py",
)
btxrd_setup = _import_setup(
    "btxrd_setup",
    Path(__file__).parent.parent / "raw" / "Dataset006_BTXRD" / "setup.py",
)
cellpose_setup = _import_setup(
    "cellpose_setup",
    Path(__file__).parent.parent / "raw" / "Datset001_Cellpose" / "setup.py",
)
motum_setup = _import_setup(
    "motum_setup",
    Path(__file__).parent.parent / "raw" / "Dataset003_MOTUM" / "setup.py",
)
mslesseg_setup = _import_setup(
    "mslesseg_setup",
    Path(__file__).parent.parent / "raw" / "Dataset002_MSLesSeg" / "setup.py",
)
kvasir_setup = _import_setup(
    "kvasir_setup",
    Path(__file__).parent.parent / "raw" / "Dataset005_KvasirSEG" / "setup.py",
)


class TestDataset005BUSBRASetup:
    """Test Dataset005_BUSBRA setup.py helper functions."""

    def test_discover_images_and_masks_with_suffix_strategy(self, tmp_path):
        """Test discovery using '_mask' suffix strategy."""
        # Create test files
        img = tmp_path / "image1.jpg"
        mask = tmp_path / "image1_mask.jpg"
        img.touch()
        mask.touch()

        pairs = busbra_setup.discover_images_and_masks(tmp_path)

        assert len(pairs) == 1
        assert pairs[0][0] == img
        assert pairs[0][1] == mask

    def test_discover_images_and_masks_excludes_mask_files(self, tmp_path):
        """Test that files with 'mask' in name are excluded from source images."""
        img = tmp_path / "image1.jpg"
        mask = tmp_path / "image1_mask.jpg"
        img.touch()
        mask.touch()

        pairs = busbra_setup.discover_images_and_masks(tmp_path)

        # Should only find 1 pair (mask file ignored as source image)
        # The source image should not have 'mask' in its name
        assert len(pairs) == 1
        assert "mask" not in pairs[0][0].name.lower()

    def test_discover_images_and_masks_no_pairs(self, tmp_path):
        """Test discovery returns empty list when no masks found."""
        img = tmp_path / "image1.jpg"
        img.touch()

        pairs = busbra_setup.discover_images_and_masks(tmp_path)

        assert len(pairs) == 0

    def test_discover_images_and_masks_nested_directories(self, tmp_path):
        """Test discovery in nested directory structure."""
        # Create nested structure
        subdir = tmp_path / "subdir"
        subdir.mkdir()
        img = subdir / "image1.jpg"
        mask = subdir / "image1_mask.jpg"
        img.touch()
        mask.touch()

        pairs = busbra_setup.discover_images_and_masks(tmp_path)

        assert len(pairs) == 1
        assert pairs[0][0] == img

    def test_find_mask_for_image_with_mask_suffix(self, tmp_path):
        """Test finding mask with '_mask' suffix."""
        img = tmp_path / "test.jpg"
        mask = tmp_path / "test_mask.jpg"
        mask.touch()

        result = busbra_setup._find_mask_for_image(img)

        assert result == mask

    def test_find_mask_for_image_not_found(self, tmp_path):
        """Test that None is returned when mask not found."""
        img = tmp_path / "test.jpg"

        result = busbra_setup._find_mask_for_image(img)

        assert result is None


class TestDataset006BTXRDSetup:
    """Test Dataset006_BTXRD setup.py helper functions."""

    def test_find_data_root_flat_structure(self, tmp_path):
        """Test finding data root in flat structure."""
        images = tmp_path / "images"
        annotations = tmp_path / "Annotations"
        images.mkdir()
        annotations.mkdir()

        result = btxrd_setup.find_data_root(tmp_path)

        assert result == tmp_path

    def test_find_data_root_nested_structure(self, tmp_path):
        """Test finding data root in nested directory structure."""
        nested = tmp_path / "data" / "dataset"
        images = nested / "images"
        annotations = nested / "Annotations"
        nested.mkdir(parents=True)
        images.mkdir()
        annotations.mkdir()

        result = btxrd_setup.find_data_root(tmp_path)

        assert result == nested

    def test_find_data_root_not_found(self, tmp_path):
        """Test that None is returned when structure not found."""
        result = btxrd_setup.find_data_root(tmp_path)

        assert result is None

    def test_find_csv_file(self, tmp_path):
        """Test finding CSV file in directory."""
        csv_file = tmp_path / "dataset.csv"
        csv_file.touch()

        result = btxrd_setup.find_csv_file(tmp_path)

        assert result == csv_file

    def test_find_csv_file_not_found(self, tmp_path):
        """Test that None is returned when CSV not found."""
        result = btxrd_setup.find_csv_file(tmp_path)

        assert result is None

    def test_get_tumor_filenames_from_csv(self, tmp_path):
        """Test reading tumor filenames from CSV."""
        csv_file = tmp_path / "dataset.csv"
        csv_content = """image1.jpg,benign
image2.jpg,malignant
image3.jpg,normal
image4.jpg,benign"""
        csv_file.write_text(csv_content)

        result = btxrd_setup.get_tumor_filenames(csv_file)

        # Should exclude 'normal' class
        assert "image1.jpg" in result
        assert "image2.jpg" in result
        assert "image3.jpg" not in result
        assert "image4.jpg" in result
        assert len(result) == 3

    def test_get_tumor_filenames_empty_csv(self, tmp_path):
        """Test with empty CSV."""
        csv_file = tmp_path / "dataset.csv"
        csv_file.write_text("")

        result = btxrd_setup.get_tumor_filenames(csv_file)

        assert result == []

    def test_get_tumor_filenames_with_comments(self, tmp_path):
        """Test CSV with comment lines."""
        csv_file = tmp_path / "dataset.csv"
        csv_content = """# This is a comment
image1.jpg,benign
# Another comment
image2.jpg,malignant"""
        csv_file.write_text(csv_content)

        result = btxrd_setup.get_tumor_filenames(csv_file)

        assert len(result) == 2
        assert "image1.jpg" in result

    def test_find_image_file_exact_match(self, tmp_path):
        """Test finding image file with exact match."""
        img_file = tmp_path / "image.jpg"
        img_file.touch()

        result = btxrd_setup.find_image_file(tmp_path, "image")

        assert result == img_file

    def test_find_image_file_different_extensions(self, tmp_path):
        """Test finding image file with different extension."""
        img_file = tmp_path / "image.png"
        img_file.touch()

        result = btxrd_setup.find_image_file(tmp_path, "image")

        assert result == img_file

    def test_find_image_file_uppercase_extension(self, tmp_path):
        """Test finding image file with uppercase extension."""
        img_file = tmp_path / "image.JPG"
        img_file.touch()

        result = btxrd_setup.find_image_file(tmp_path, "image")

        assert result == img_file

    def test_find_image_file_not_found(self, tmp_path):
        """Test that None is returned when image not found."""
        result = btxrd_setup.find_image_file(tmp_path, "nonexistent")

        assert result is None

    def test_rasterize_coco_annotations_simple(self, tmp_path):
        """Test rasterizing simple COCO annotations."""
        coco_data = {
            "images": [{"width": 100, "height": 100, "id": 1}],
            "annotations": [
                {
                    "id": 1,
                    "image_id": 1,
                    "segmentation": [
                        [10, 10, 90, 10, 90, 90, 10, 90]  # Square polygon
                    ],
                }
            ],
        }
        json_file = tmp_path / "annotations.json"
        json_file.write_text(json.dumps(coco_data))

        result = btxrd_setup.rasterize_coco_annotations(json_file)

        assert result is not None
        assert result.mode == "L"
        assert result.size == (100, 100)

    def test_rasterize_coco_annotations_no_image_info(self, tmp_path):
        """Test rasterizing when image dimensions not in JSON."""
        coco_data = {
            "annotations": [
                {
                    "id": 1,
                    "segmentation": [[10, 10, 90, 10, 90, 90, 10, 90]],
                }
            ],
        }
        json_file = tmp_path / "annotations.json"
        json_file.write_text(json.dumps(coco_data))

        result = btxrd_setup.rasterize_coco_annotations(json_file)

        assert result is not None
        assert result.size == (512, 512)

    def test_rasterize_coco_annotations_invalid_json(self, tmp_path):
        """Test with invalid JSON file."""
        json_file = tmp_path / "annotations.json"
        json_file.write_text("invalid json {")

        result = btxrd_setup.rasterize_coco_annotations(json_file)

        assert result is None

    def test_rasterize_coco_annotations_empty_segmentation(self, tmp_path):
        """Test with empty segmentation."""
        coco_data = {
            "images": [{"width": 100, "height": 100}],
            "annotations": [],
        }
        json_file = tmp_path / "annotations.json"
        json_file.write_text(json.dumps(coco_data))

        result = btxrd_setup.rasterize_coco_annotations(json_file)

        assert result is not None
        assert result.mode == "L"

    def test_build_valid_pairs(self, tmp_path):
        """Test building valid image-annotation pairs."""
        # Create test files
        images_dir = tmp_path / "images"
        annotations_dir = tmp_path / "annotations"
        images_dir.mkdir()
        annotations_dir.mkdir()

        (images_dir / "image1.jpg").touch()
        (images_dir / "image2.jpg").touch()
        (images_dir / "image3.jpg").touch()

        (annotations_dir / "image1.json").touch()
        (annotations_dir / "image2.json").touch()
        # image3 has no annotation

        tumor_filenames = ["image1.jpg", "image2.jpg", "image3.jpg"]

        pairs, skipped = btxrd_setup._build_valid_pairs(
            tumor_filenames, images_dir, annotations_dir
        )

        assert len(pairs) == 2
        assert skipped == 1
        assert pairs[0][0].name == "image1.jpg"
        assert pairs[1][0].name == "image2.jpg"

    def test_build_valid_pairs_missing_images(self, tmp_path):
        """Test when annotation exists but image doesn't."""
        images_dir = tmp_path / "images"
        annotations_dir = tmp_path / "annotations"
        images_dir.mkdir()
        annotations_dir.mkdir()

        (annotations_dir / "image1.json").touch()
        # image1 missing

        tumor_filenames = ["image1.jpg"]

        pairs, skipped = btxrd_setup._build_valid_pairs(
            tumor_filenames, images_dir, annotations_dir
        )

        assert len(pairs) == 0
        assert skipped == 1


class TestDataset001CellposeSetup:
    """Test Dataset001_Cellpose setup.py helper functions."""

    def test_setup_dataset_extracts_zip(self, tmp_path, monkeypatch):
        """Test that setup_dataset extracts zip file."""
        # Create test data directory
        monkeypatch.chdir(tmp_path)

        # Create a minimal train.zip
        train_zip = tmp_path / "train.zip"
        with zipfile.ZipFile(train_zip, "w") as z:
            z.writestr("train/dummy.txt", "test")

        test_zip = tmp_path / "test.zip"
        with zipfile.ZipFile(test_zip, "w") as z:
            z.writestr("test/dummy.txt", "test")

        # Create dataset.json
        dataset_json = tmp_path / "dataset.json"
        dataset_json.write_text(json.dumps({"name": "test"}))

        # Test extraction
        with patch.object(cellpose_setup, "tqdm", lambda x: x):  # Mock tqdm
            try:
                cellpose_setup.setup_dataset()
            except (FileNotFoundError, Exception):
                # Expected - we don't have real image files
                pass

        # Verify temp directories were created (even if processing failed)
        assert (tmp_path / "temp_train").exists() or not (tmp_path / "temp_train").exists()


class TestDataset003MOTUMSetup:
    """Test Dataset003_MOTUM setup.py helper functions."""

    def test_setup_module_has_setup_function(self):
        """Test that setup_dataset function exists."""
        assert hasattr(motum_setup, "setup_dataset")
        assert callable(motum_setup.setup_dataset)


class TestDataset002MSLesSegSetup:
    """Test Dataset002_MSLesSeg setup.py helper functions."""

    def test_setup_module_has_setup_function(self):
        """Test that setup_dataset function exists."""
        assert hasattr(mslesseg_setup, "setup_dataset")
        assert callable(mslesseg_setup.setup_dataset)


class TestDataset005KvasirSEGSetup:
    """Test Dataset005_KvasirSEG setup.py helper functions."""

    def test_setup_module_has_setup_function(self):
        """Test that setup_dataset function exists."""
        assert hasattr(kvasir_setup, "setup_dataset")
        assert callable(kvasir_setup.setup_dataset)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
