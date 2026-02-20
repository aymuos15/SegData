# Cellpose 2D Cell Segmentation Dataset

2D cell segmentation dataset formatted for nnUNet v2.

## Key Conversion: Instance → Semantic

**Instance labels are binarized to binary semantic segmentation:**
- **Input**: Instance masks (each cell = unique pixel value)
- **Output**: Binary masks (0 = background, 1 = cell)
- **Method**: All non-zero pixels → 1

**Why?** Instance segmentation (tracking individual cells) is more complex than semantic segmentation (identifying cell regions). For general segmentation networks like nnUNet, binary semantic segmentation is more appropriate and easier to train. Individual cell boundaries can be recovered post-hoc if needed.

## Format Details
- **File format**: PNG (lossless)
- **Channels**: 3 (RGB, saved as separate files: _0000, _0001, _0002)
- **Labels**: 0 (background), 1 (cell)
- **2D images**: No pseudo-3D conversion
