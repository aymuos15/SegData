# SegData Repository

This repo has (some opinionated) code to setup common opensource datasets in the **nnU-Net format**. Only those datasts will be added which are openly accessible and do not have a request form to fill out which bars accessibility. The goal is to make it easy for researchers to quickly get up and running with a variety of datasets for segmentation tasks, without needing to worry about data formatting or preprocessing.

## Repository Structure

Each dataset directory follows the nnU-Net standard format with the following structure:

```
Dataset###_Name/
├── imagesTr/           # Training images
├── labelsTr/           # Training labels (segmentation masks)
├── imagesTs/           # Test images
├── labelsTs/           # Test labels (optional)
├── results/            # Analysis and statistics output
├── dataset.json        # Dataset metadata (auto-generated)
├── setup.py            # Setup script to prepare the dataset
├── get.md              # Instructions for obtaining the dataset
├── stats.py            # Dataset statistics and analysis
├── plot.py             # Visualization tool (optional)
└── info.md             # Dataset-specific information (optional)
```

## Results Directory

The `results/` folder contains output from `stats.py`:
- Per-case analysis JSON files
- Dataset-wide statistics (dimensions, counts, class distributions)
- Summary reports

## Optional: Visualization (plot.py)

Each dataset can optionally include a `plot.py` script for visualizing images and labels side by side.

**Requirements:** `matplotlib` (install with `uv sync --all-extras` or `pip install matplotlib`)

**Usage:**
```bash
python3 plot.py 000                # Show case 000 as RGB
python3 plot.py 000 --channel 0    # Show specific channel
python3 plot.py 000 --test         # Show from test set
```

## Optional: info.md

Each dataset can optionally include an `info.md` file with dataset-specific information such as:
- Conversion methods or preprocessing details
- Notes on format or structure

**Note:** Source and citation information should be stored in `dataset.json`.

## nnU-Net Format

All datasets conform to the nnU-Net data format:
- Images and labels stored as PNG files
- Naming convention: `{case_id}_{channel_id}.png`
- Metadata stored in `dataset.json`

## Usage

For each dataset:

```bash
cd Dataset###_Name
python setup.py    # Prepare the dataset
python stats.py    # View dataset statistics
```

Then use with nnU-Net or other segmentation frameworks.

# Contributing

Coming soon. Although open to PRs which clearly follow the current system to add datasets.
