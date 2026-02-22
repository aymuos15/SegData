#!/usr/bin/env python3
"""
Quick tests for src/plot.py, src/stats.py, and src/verify.py.

Verifies that all three scripts work correctly with all existing datasets.

Run with: pytest tests/test_src_scripts.py -v
"""

import sys
from pathlib import Path

import pytest

# Add src to path so we can import the scripts
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import plot
import stats
import verify


def get_dataset_dirs():
    """Get all dataset directories."""
    raw_dir = Path(__file__).parent.parent / "raw"
    return sorted([d for d in raw_dir.iterdir() if d.is_dir()])


@pytest.mark.parametrize("dataset_dir", get_dataset_dirs(), ids=lambda d: d.name)
class TestPlotScript:
    """Test plot.py functionality."""

    def test_load_metadata(self, dataset_dir):
        """Test that metadata can be loaded."""
        meta = plot.load_dataset_meta(dataset_dir)
        assert isinstance(meta, dict)
        assert "tensorImageSize" in meta
        assert "file_ending" in meta
        assert meta["tensorImageSize"] in ("2D", "3D")

    def test_list_case_ids(self, dataset_dir):
        """Test that case IDs can be listed."""
        meta = plot.load_dataset_meta(dataset_dir)
        file_ending = meta.get("file_ending", ".png")

        train_ids = plot.list_case_ids(dataset_dir, file_ending, "train")
        assert isinstance(train_ids, list)
        assert len(train_ids) > 0
        assert all(isinstance(cid, str) for cid in train_ids)

    def test_normalize_image(self, dataset_dir):
        """Test image normalization."""
        import numpy as np

        # Create dummy image
        img = np.random.randint(0, 255, (100, 100), dtype=np.uint8)
        normalized = plot.normalize_image(img)

        assert isinstance(normalized, np.ndarray)
        assert normalized.min() >= 0 and normalized.max() <= 1

    def test_get_mid_slices_3d(self, dataset_dir):
        """Test mid-slice extraction for 3D volumes."""
        import numpy as np

        # Create dummy 3D volume
        volume = np.random.randint(0, 255, (50, 60, 70), dtype=np.uint8)
        axial, coronal, sagittal = plot.get_mid_slices(volume)

        assert axial.shape == (60, 70)
        assert coronal.shape == (50, 70)
        assert sagittal.shape == (50, 60)


@pytest.mark.parametrize("dataset_dir", get_dataset_dirs(), ids=lambda d: d.name)
class TestStatsScript:
    """Test stats.py functionality."""

    def test_load_metadata(self, dataset_dir):
        """Test that metadata can be loaded."""
        meta = stats.load_dataset_meta(dataset_dir)
        assert isinstance(meta, dict)
        assert "tensorImageSize" in meta
        assert "file_ending" in meta

    def test_analyzer_initialization(self, dataset_dir):
        """Test that analyzer can be initialized."""
        meta = stats.load_dataset_meta(dataset_dir)
        analyzer = stats.DatasetAnalyzer(dataset_dir, meta)

        assert analyzer.dataset_dir == dataset_dir
        assert analyzer.is_3d == (meta["tensorImageSize"] == "3D")
        assert analyzer.file_ending == meta["file_ending"]

    def test_count_samples(self, dataset_dir):
        """Test sample counting."""
        meta = stats.load_dataset_meta(dataset_dir)
        analyzer = stats.DatasetAnalyzer(dataset_dir, meta)

        counts = analyzer.count_samples()
        assert isinstance(counts, dict)
        assert "train" in counts
        assert "test" in counts
        assert "total" in counts
        assert counts["train"] >= 0
        assert counts["test"] >= 0
        assert counts["total"] == counts["train"] + counts["test"]

    def test_get_image_dimensions_2d(self, dataset_dir):
        """Test 2D image dimension calculation."""
        meta = stats.load_dataset_meta(dataset_dir)

        if meta["tensorImageSize"] != "2D":
            pytest.skip("Not a 2D dataset")

        analyzer = stats.DatasetAnalyzer(dataset_dir, meta)
        counts = analyzer.count_samples()

        if counts["train"] == 0:
            pytest.skip("No training samples")

        # Get sample IDs
        pattern = f"*_0000{meta['file_ending']}"
        image_files = sorted(analyzer.images_tr.glob(pattern))
        sample_ids = [f.name.replace(f"_0000{meta['file_ending']}", "") for f in image_files]

        if not sample_ids:
            pytest.skip("Could not extract sample IDs")

        dims = analyzer.get_image_dimensions_2d(sample_ids, analyzer.images_tr)

        assert "mean_height" in dims
        assert "mean_width" in dims
        assert dims["mean_height"] > 0
        assert dims["mean_width"] > 0

    def test_get_image_dimensions_3d(self, dataset_dir):
        """Test 3D volume dimension calculation."""
        meta = stats.load_dataset_meta(dataset_dir)

        if meta["tensorImageSize"] != "3D":
            pytest.skip("Not a 3D dataset")

        analyzer = stats.DatasetAnalyzer(dataset_dir, meta)
        counts = analyzer.count_samples()

        if counts["train"] == 0:
            pytest.skip("No training samples")

        # Get sample IDs
        pattern = f"*_0000{meta['file_ending']}"
        image_files = sorted(analyzer.images_tr.glob(pattern))
        sample_ids = [f.name.replace(f"_0000{meta['file_ending']}", "") for f in image_files]

        if not sample_ids:
            pytest.skip("Could not extract sample IDs")

        dims = analyzer.get_image_dimensions_3d(sample_ids, analyzer.images_tr)

        assert "mean_shape" in dims
        assert len(dims["mean_shape"]) == 3
        assert all(x > 0 for x in dims["mean_shape"])


@pytest.mark.parametrize("dataset_dir", get_dataset_dirs(), ids=lambda d: d.name)
class TestVerifyScript:
    """Test verify.py functionality."""

    def test_validator_initialization(self, dataset_dir):
        """Test that validator can be initialized."""
        validator = verify.DatasetValidator(dataset_dir)
        assert validator.dataset_path == dataset_dir.resolve()

    def test_dataset_json_loading(self, dataset_dir):
        """Test dataset.json can be loaded by validator."""
        validator = verify.DatasetValidator(dataset_dir)
        validator._check_dataset_json()

        assert validator.dataset_json_data is not None
        assert isinstance(validator.dataset_json_data, dict)

    def test_validator_checks_pass(self, dataset_dir):
        """Test that validator can run all checks without crashing."""
        validator = verify.DatasetValidator(dataset_dir)

        # Run all checks - should not raise exception
        exit_code = validator.run_all_checks()

        # Exit code should be 0 (pass) or 1 (fail), not error
        assert exit_code in (0, 1)

    def test_dataset_json_metadata_keys(self, dataset_dir):
        """Test that dataset.json has all required metadata keys."""
        validator = verify.DatasetValidator(dataset_dir)
        validator._check_dataset_json()

        data = validator.dataset_json_data
        assert data is not None, "dataset.json data should not be None"

        required_keys = {
            "name",
            "description",
            "tensorImageSize",
            "file_ending",
            "channel_names",
            "labels",
            "numTraining",
            "numTest",
            "numLabels",
        }

        assert required_keys.issubset(set(data.keys()))
        assert data["tensorImageSize"] in ("2D", "3D")
        assert data["file_ending"].startswith(".")


class TestScriptsIntegration:
    """Integration tests across all scripts."""

    def test_all_datasets_have_metadata(self):
        """Test that all datasets have valid metadata."""
        for dataset_dir in get_dataset_dirs():
            meta = plot.load_dataset_meta(dataset_dir)
            assert "tensorImageSize" in meta
            assert "file_ending" in meta
            assert "labels" in meta

    def test_metadata_consistency_across_scripts(self):
        """Test that metadata loading is consistent across scripts."""
        for dataset_dir in get_dataset_dirs():
            plot_meta = plot.load_dataset_meta(dataset_dir)
            stats_meta = stats.load_dataset_meta(dataset_dir)

            # Key metadata should match
            assert plot_meta["tensorImageSize"] == stats_meta["tensorImageSize"]
            assert plot_meta["file_ending"] == stats_meta["file_ending"]
            assert plot_meta["numLabels"] == stats_meta["numLabels"]

    def test_case_id_listing_consistency(self):
        """Test that case IDs are consistent with sample counts."""
        for dataset_dir in get_dataset_dirs():
            meta = plot.load_dataset_meta(dataset_dir)
            file_ending = meta["file_ending"]

            analyzer = stats.DatasetAnalyzer(dataset_dir, meta)
            counts = analyzer.count_samples()

            train_ids = plot.list_case_ids(dataset_dir, file_ending, "train")
            test_ids = plot.list_case_ids(dataset_dir, file_ending, "test")

            assert len(train_ids) == counts["train"]
            assert len(test_ids) == counts["test"]
